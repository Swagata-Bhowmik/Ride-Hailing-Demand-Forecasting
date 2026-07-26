"""Prophet forecasting model family (Requirement 5.4).

Prophet (facebook/meta ``prophet``) is an additive time-series model that
decomposes a series into trend + seasonalities + holidays. It is robust to
missing data and shifts in trend, needs little tuning, and takes a simple
two-column frame - ``ds`` (datestamp) and ``y`` (value) - which makes it a
natural "modern" entry in the Candidate_Model_Set alongside the classical
(SARIMA), multivariate (VAR) and ML (XGBoost) families.

This module implements :class:`ProphetModel`, conforming to the
:class:`~src.models.base.Forecaster` protocol so :func:`~src.models.base.train_all`
can train and score it exactly like every other candidate.

Region handling
---------------
Prophet is a **univariate** model: it forecasts a single ``y`` series, whereas the
:data:`~src.preparation.DemandSeries` is long-format with one row per
``(period, region)``. This wrapper therefore reduces the multi-region series to a
single series before fitting, controlled by the ``region`` constructor argument:

* ``region=None`` (default): **total demand** - sum demand across all regions for
  each ``period``, giving one project-level daily total series. This is the
  natural univariate view for a single-series model and keeps Prophet comparable
  to the other univariate candidates. Joint multi-region forecasting is the job
  of the VAR/VARMAX family (task 7.4), not Prophet.
* ``region="Manhattan"`` (a specific borough): fit only that region's series, so
  Prophet can be pointed at one borough when a per-region forecast is wanted.

Either way the model produces **one** :class:`~src.models.base.Forecast` whose
values align to the Holdout_Set periods (R5.7), consistent with how the
Evaluation_Framework compares models.

Prophet import
--------------
``prophet`` (and its ``cmdstanpy``/compiler backend) is a heavy optional
dependency. It is imported **lazily inside** :meth:`ProphetModel.fit` rather than
at module import time, so this module always imports even in an environment where
Prophet is not installed. If Prophet is missing, ``fit`` raises a clear
``ImportError`` which :func:`~src.models.base.train_all` catches and turns into an
:class:`~src.models.base.ExclusionRecord` (R5.8) - one missing dependency never
aborts the other candidates.

Design references:
- Components and Interfaces -> Forecasting_System (`src/models/`)
- Data Models -> Forecast
- Requirements 5.4, 5.7, 5.8
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

import pandas as pd

from src.models.base import Forecast
from src.preparation import DEMAND_COLUMN, PERIOD_COLUMN, REGION_COLUMN

if TYPE_CHECKING:  # pragma: no cover - imported only for type checking
    from src.config import ScopeConfig
    from src.preparation import DemandSeries


#: Time_Grain -> the pandas frequency alias Prophet uses to build future periods.
#: Only the daily grain is supported for now (the project default), matching the
#: preparation pipeline's supported grains.
_TIME_GRAIN_FREQ: dict[str, str] = {
    "daily": "D",
}


class ProphetModel:
    """A Prophet forecaster over the project's demand series (Requirement 5.4).

    Conforms to the :class:`~src.models.base.Forecaster` protocol: :meth:`fit`
    collapses the long-format :class:`~src.preparation.DemandSeries` to a single
    ``ds``/``y`` frame (see module docstring "Region handling"), fits a Prophet
    model, and :meth:`predict` forecasts ``horizon`` future daily periods,
    returning a :class:`~src.models.base.Forecast` aligned to the Holdout_Set.

    Attributes:
        name: Human-readable model name used as the key in the evaluation table.
        region: The region to forecast. ``None`` (default) means total demand
            summed across all regions; a borough name means that region only.
        prophet_kwargs: Extra keyword arguments forwarded to the ``Prophet``
            constructor (e.g. ``seasonality_mode``), so callers can tune the model
            without subclassing.
    """

    def __init__(
        self,
        region: Optional[str] = None,
        *,
        name: Optional[str] = None,
        **prophet_kwargs: Any,
    ) -> None:
        """Create a Prophet forecaster.

        Args:
            region: Which region to forecast. ``None`` (default) sums demand
                across all regions into one total series; a borough name restricts
                the fit to that region's series.
            name: Optional override of the model name. Defaults to ``"Prophet"``
                (total) or ``"Prophet (<region>)"`` when a region is given.
            **prophet_kwargs: Extra keyword arguments forwarded to the ``Prophet``
                constructor.
        """
        self.region = region
        if name is not None:
            self.name = name
        elif region is not None:
            self.name = f"Prophet ({region})"
        else:
            self.name = "Prophet"
        self.prophet_kwargs = prophet_kwargs

        # Populated by fit(); used by predict().
        self._model: Any = None
        self._freq: str = "D"
        self._last_train_period: Optional[pd.Timestamp] = None

    def _to_ds_y(self, train: "DemandSeries") -> pd.DataFrame:
        """Collapse the long-format DemandSeries to Prophet's ``ds``/``y`` frame.

        Applies the region-handling rule (see module docstring): when
        ``self.region`` is ``None`` the demand is summed across regions per period
        (total demand); otherwise only the selected region's rows are used. The
        result is sorted by ``ds`` and has exactly one row per period, matching
        Prophet's expectation of a single univariate series.

        Args:
            train: The training :class:`~src.preparation.DemandSeries`.

        Returns:
            A DataFrame with columns ``ds`` (datetime period) and ``y`` (demand).

        Raises:
            KeyError: If the required period/region/demand columns are missing.
            ValueError: If the selected region is absent or the series is empty.
        """
        for col in (PERIOD_COLUMN, REGION_COLUMN, DEMAND_COLUMN):
            if col not in train.columns:
                raise KeyError(
                    f"Column '{col}' not found in training series. "
                    f"Available columns: {list(train.columns)}."
                )

        work = train[[PERIOD_COLUMN, REGION_COLUMN, DEMAND_COLUMN]].copy()
        work[PERIOD_COLUMN] = pd.to_datetime(work[PERIOD_COLUMN])

        if self.region is not None:
            work = work[work[REGION_COLUMN] == self.region]
            if work.empty:
                available = sorted(map(str, pd.unique(train[REGION_COLUMN].dropna())))
                raise ValueError(
                    f"Region '{self.region}' not present in the training series. "
                    f"Available regions: {available}."
                )

        # Sum across regions (total demand) or collapse any duplicate periods
        # within the single selected region into one row per period.
        collapsed = (
            work.groupby(PERIOD_COLUMN, as_index=False, sort=True)[DEMAND_COLUMN]
            .sum()
            .rename(columns={PERIOD_COLUMN: "ds", DEMAND_COLUMN: "y"})
        )

        if collapsed.empty:
            raise ValueError(
                "Training series is empty after region handling; Prophet needs at "
                "least two periods to fit."
            )

        collapsed["y"] = collapsed["y"].astype("float64")
        return collapsed.reset_index(drop=True)

    def fit(self, train: "DemandSeries", scope: "ScopeConfig") -> None:
        """Fit Prophet on the training series (Requirement 5.4).

        Imports ``prophet`` lazily (see module docstring) so the module still
        imports when Prophet is not installed; a missing dependency raises a clear
        ``ImportError`` here that :func:`~src.models.base.train_all` records as an
        exclusion (R5.8). The long-format series is reduced to a ``ds``/``y`` frame
        per the region-handling rule, then handed to a freshly constructed
        ``Prophet`` model.

        Args:
            train: The training :class:`~src.preparation.DemandSeries` (periods
                before the Holdout_Set).
            scope: The project :class:`~src.config.ScopeConfig`; its Time_Grain
                selects the future-period frequency used by :meth:`predict`.

        Raises:
            ImportError: If the ``prophet`` package is not installed.
            ValueError: If the Time_Grain is unsupported or the series is empty.
            KeyError: If required DemandSeries columns are missing.
        """
        try:
            from prophet import Prophet  # noqa: PLC0415 - lazy, optional heavy dep
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise ImportError(
                "The 'prophet' package is required to train the Prophet model but "
                "is not installed. Install it with `pip install prophet` (see "
                "requirements-full.txt). Until then this model is excluded from the "
                "comparison."
            ) from exc

        grain = scope.time_grain.lower()
        if grain not in _TIME_GRAIN_FREQ:
            raise ValueError(
                f"Unsupported time_grain '{scope.time_grain}' for Prophet. "
                f"Supported grains: {sorted(_TIME_GRAIN_FREQ)}."
            )
        self._freq = _TIME_GRAIN_FREQ[grain]

        ds_y = self._to_ds_y(train)
        self._last_train_period = pd.Timestamp(ds_y["ds"].iloc[-1])

        model = Prophet(**self.prophet_kwargs)
        model.fit(ds_y)
        self._model = model

    def predict(self, horizon: int) -> Forecast:
        """Forecast ``horizon`` future daily periods over the Holdout_Set (R5.7).

        Builds a future frame extending ``horizon`` periods past the last training
        period at the Time_Grain frequency, runs Prophet's ``predict``, and returns
        the last ``horizon`` ``yhat`` values as a :class:`~src.models.base.Forecast`
        whose ``index`` is the corresponding future periods - the Holdout_Set the
        Evaluation_Framework compares against.

        Args:
            horizon: Number of future periods to forecast (the Holdout_Set length).

        Returns:
            A :class:`~src.models.base.Forecast` with ``horizon`` values indexed by
            the forecast periods.

        Raises:
            RuntimeError: If called before :meth:`fit`.
            ValueError: If ``horizon`` is not a positive integer.
        """
        if self._model is None:
            raise RuntimeError("ProphetModel.predict called before fit.")
        if horizon <= 0:
            raise ValueError(f"horizon must be a positive integer; got {horizon}.")

        future = self._model.make_future_dataframe(periods=horizon, freq=self._freq)
        forecast_df = self._model.predict(future)

        # The most-recent `horizon` rows are the out-of-sample future periods that
        # line up with the Holdout_Set; earlier rows are in-sample fitted values.
        tail = forecast_df.iloc[-horizon:]
        index = pd.DatetimeIndex(pd.to_datetime(tail["ds"].to_numpy()))
        values = tail["yhat"].to_numpy()

        return Forecast(model_name=self.name, values=values, index=index)
