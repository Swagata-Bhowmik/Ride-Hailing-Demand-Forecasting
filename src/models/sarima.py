"""SARIMA / SARIMAX classical-univariate forecasters (Requirement 5.2).

This module provides the ARIMA-family entry in the Candidate_Model_Set: a
seasonal ARIMA model (:class:`Sarima`) and its exogenous-regressor variant
(:class:`Sarimax`), both built on ``statsmodels.tsa.statespace.SARIMAX`` and both
implementing the shared :class:`~src.models.base.Forecaster` interface so the
Evaluation_Framework can train and score them exactly like every other family.

Region-handling choice (documented per task 7.3)
------------------------------------------------
SARIMA/SARIMAX are *univariate* time-series models: each one describes a single
scalar series through time. The project's :data:`~src.preparation.DemandSeries`,
however, is *long format* - one demand value per ``(period, region)`` - covering
several boroughs at once (the Geographic_Grain).

The choice made here is **one independent SARIMAX per region**: at ``fit`` the
series is split by ``region`` and a separate seasonal model is estimated on each
borough's own daily demand history; at ``predict`` each region is forecast
forward ``horizon`` periods and the per-region forecasts are concatenated back
into a single long-format :class:`~src.models.base.Forecast` aligned to the
holdout grid (sorted by ``period`` then ``region``, matching how
:func:`~src.preparation.fill_missing_periods` orders the series).

Rationale: boroughs have distinct demand levels and seasonal shapes, so a shared
univariate model would be mis-specified. Modelling each region independently
keeps SARIMA a faithful *classical univariate* baseline (joint multi-region
dependence is the job of the VAR/VARMAX family, task 7.4). To keep the golden
rule - one model failing never aborts the comparison (Requirement 5.8) - a region
whose SARIMAX cannot be estimated (too little history, non-convergence) falls back
to a documented seasonal-naive forecast for that region only, recorded in
:attr:`Sarima.fit_notes`, rather than failing the whole model.

Exogenous-variable choice for SARIMAX
-------------------------------------
:class:`Sarimax` adds exogenous regressors to the SARIMA structure. Because a
forecast needs the exogenous values *over the future horizon*, the default uses
**deterministic day-of-week calendar indicators** derived straight from the
``period`` timestamps - these are fully known for any future date, which is the
textbook-safe way to use SARIMAX. Callers may instead pass ``exog_columns`` to
use existing :class:`~src.preparation.DemandSeries` columns (e.g. lag features);
in that case future exogenous values are extended by a documented seasonal-naive
rule (the value from ``seasonal_periods`` steps earlier, falling back to the last
observed value), since such columns are not otherwise known ahead of time.

Design references:
- Components and Interfaces -> Forecasting_System (`src/models/`)
- Data Models -> Forecast, DemandSeries (long format)
- Error Handling -> a candidate model failing is isolated (R5.8)
- Requirements 5.2, 5.7
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Optional, Sequence

import numpy as np
import pandas as pd

from src.models.base import Forecast

if TYPE_CHECKING:  # pragma: no cover - imported only for type checking
    from src.config import ScopeConfig
    from src.preparation import DemandSeries

# --- Canonical DemandSeries column names (mirror src.preparation) ------------

#: Temporal key of the DemandSeries (pickup timestamp at the Time_Grain).
PERIOD_COLUMN = "period"

#: Geographic_Grain key of the DemandSeries (borough).
REGION_COLUMN = "region"

#: The forecasting target column (trip count per bucket, >= 0).
DEMAND_COLUMN = "demand"

# --- Time_Grain wiring -------------------------------------------------------

#: Time_Grain -> pandas frequency alias, used to build the future period index.
#: Only the daily grain (the project default) is supported for now.
_TIME_GRAIN_FREQ: dict[str, str] = {"daily": "D"}

#: Time_Grain -> default seasonal period ``s``. Daily demand repeats weekly, so
#: the natural seasonal cycle is 7 (Key Design Decisions: weekly seasonality).
_TIME_GRAIN_SEASONAL_PERIODS: dict[str, int] = {"daily": 7}

#: Default non-seasonal ARIMA order ``(p, d, q)``. A first difference with one AR
#: and one MA term is a sensible, widely-used starting point for daily demand.
DEFAULT_ORDER: tuple[int, int, int] = (1, 1, 1)

#: Default seasonal order ``(P, D, Q)``; the seasonal period ``s`` is appended
#: from the Time_Grain (7 for daily) unless overridden.
DEFAULT_SEASONAL_ORDER: tuple[int, int, int] = (1, 1, 1)


def _freq_for_grain(time_grain: str) -> str:
    """Return the pandas frequency alias for a Time_Grain, or raise if unsupported."""
    grain = time_grain.lower()
    if grain not in _TIME_GRAIN_FREQ:
        raise ValueError(
            f"Unsupported time_grain '{time_grain}'. "
            f"Supported grains: {sorted(_TIME_GRAIN_FREQ)}."
        )
    return _TIME_GRAIN_FREQ[grain]


def _seasonal_periods_for_grain(time_grain: str) -> int:
    """Return the default seasonal period ``s`` for a Time_Grain."""
    grain = time_grain.lower()
    if grain not in _TIME_GRAIN_SEASONAL_PERIODS:
        raise ValueError(
            f"Unsupported time_grain '{time_grain}'. "
            f"Supported grains: {sorted(_TIME_GRAIN_SEASONAL_PERIODS)}."
        )
    return _TIME_GRAIN_SEASONAL_PERIODS[grain]


def _calendar_exog(periods: pd.DatetimeIndex) -> pd.DataFrame:
    """Build deterministic day-of-week exogenous indicators for ``periods``.

    Returns a one-hot encoding of the day of week with the first level dropped to
    avoid perfect collinearity with the model's constant. These features are fully
    determined by the calendar date, so they are known over any future horizon -
    which is exactly what SARIMAX needs from its exogenous regressors.

    Args:
        periods: The timestamps to derive calendar features for.

    Returns:
        A DataFrame indexed by ``periods`` with columns ``dow_1``..``dow_6``
        (Monday is dropped as the reference level).
    """
    idx = pd.DatetimeIndex(periods)
    dow = idx.dayofweek  # Monday=0 .. Sunday=6
    frame = pd.DataFrame(index=idx)
    # Drop the first level (Monday=0) as the reference category.
    for day in range(1, 7):
        frame[f"dow_{day}"] = (dow == day).astype(float)
    return frame


class Sarima:
    """Seasonal ARIMA forecaster, one independent model per region (R5.2).

    Implements the :class:`~src.models.base.Forecaster` interface using
    ``statsmodels`` SARIMAX with a seasonal order suited to the Time_Grain
    (weekly, ``s = 7``, for the daily grain). See the module docstring for the
    region-handling choice: one univariate model is estimated per borough and the
    per-region forecasts are recombined into a single long-format
    :class:`~src.models.base.Forecast`.

    Orders are configurable with sensible defaults (:data:`DEFAULT_ORDER`,
    :data:`DEFAULT_SEASONAL_ORDER`); ``seasonal_periods`` defaults to the value
    implied by the scope's Time_Grain.

    Attributes:
        name: Model name used across the evaluation comparison (``"SARIMA"``).
        order: The non-seasonal ``(p, d, q)`` order.
        seasonal_order: The seasonal ``(P, D, Q)`` order (``s`` appended at fit).
        seasonal_periods: Override for the seasonal period ``s``; ``None`` means
            derive it from the Time_Grain.
        fit_notes: Per-region notes recorded during ``fit`` (e.g. a region that
            fell back to a seasonal-naive forecast because SARIMAX did not fit).
    """

    name: str = "SARIMA"

    def __init__(
        self,
        order: tuple[int, int, int] = DEFAULT_ORDER,
        seasonal_order: tuple[int, int, int] = DEFAULT_SEASONAL_ORDER,
        seasonal_periods: Optional[int] = None,
        *,
        enforce_stationarity: bool = False,
        enforce_invertibility: bool = False,
    ) -> None:
        self.order = tuple(order)
        self.seasonal_order = tuple(seasonal_order)
        self.seasonal_periods = seasonal_periods
        self.enforce_stationarity = enforce_stationarity
        self.enforce_invertibility = enforce_invertibility

        # Populated at fit():
        self._freq: Optional[str] = None
        self._s: Optional[int] = None
        self._regions: list[str] = []
        # region -> dict(result, last_period, endog, exog_history)
        self._models: dict[str, dict] = {}
        self.fit_notes: dict[str, str] = {}

    # --- exogenous hooks (overridden by Sarimax) -----------------------------

    def _uses_exog(self) -> bool:
        """Whether this model supplies exogenous regressors to SARIMAX."""
        return False

    def _fit_exog(self, region_df: pd.DataFrame, periods: pd.DatetimeIndex):
        """Exogenous design matrix for training; ``None`` for plain SARIMA."""
        return None

    def _future_exog(self, region: str, future_periods: pd.DatetimeIndex):
        """Exogenous design matrix over the forecast horizon; ``None`` for SARIMA."""
        return None

    # --- Forecaster interface ------------------------------------------------

    def fit(self, train: "DemandSeries", scope: "ScopeConfig") -> None:
        """Fit one seasonal model per region on the training DemandSeries.

        The training series is split by ``region``; each borough's demand is
        ordered by ``period`` and fit with SARIMAX using the configured orders and
        the seasonal period implied by the Time_Grain. A region whose model cannot
        be estimated is recorded in :attr:`fit_notes` and served by a seasonal-naive
        fallback at predict time, so a single problematic region never aborts the
        whole model (Requirement 5.8).

        Args:
            train: The training :class:`~src.preparation.DemandSeries` (long
                format with at least ``period``, ``region``, ``demand``).
            scope: The project :class:`~src.config.ScopeConfig` (Time_Grain drives
                the frequency and default seasonal period).

        Raises:
            KeyError: If required columns are missing from ``train``.
            ValueError: If the scope's Time_Grain is unsupported, or the series has
                no usable rows.
        """
        for col in (PERIOD_COLUMN, REGION_COLUMN, DEMAND_COLUMN):
            if col not in train.columns:
                raise KeyError(
                    f"Column '{col}' not found in train. "
                    f"Available columns: {list(train.columns)}."
                )

        self._freq = _freq_for_grain(scope.time_grain)
        self._s = (
            int(self.seasonal_periods)
            if self.seasonal_periods is not None
            else _seasonal_periods_for_grain(scope.time_grain)
        )

        work = train.copy()
        work[PERIOD_COLUMN] = pd.to_datetime(work[PERIOD_COLUMN])
        work = work.dropna(subset=[PERIOD_COLUMN, REGION_COLUMN])
        if work.empty:
            raise ValueError("Cannot fit SARIMA: training series has no usable rows.")

        self._regions = sorted(work[REGION_COLUMN].unique())
        self._models = {}
        self.fit_notes = {}

        seasonal_order_full = (*self.seasonal_order, self._s)

        for region in self._regions:
            region_df = (
                work[work[REGION_COLUMN] == region]
                .sort_values(PERIOD_COLUMN, kind="stable")
            )
            periods = pd.DatetimeIndex(region_df[PERIOD_COLUMN].to_numpy())
            endog = pd.Series(
                region_df[DEMAND_COLUMN].to_numpy(dtype=float), index=periods
            )
            # Attach a frequency where the periods are regular (they are after
            # zero-fill); this keeps statsmodels from warning about the index.
            inferred = pd.infer_freq(endog.index) if len(endog) > 2 else None
            if inferred is not None:
                endog = endog.asfreq(inferred)
                endog = endog.fillna(0.0)

            exog = self._fit_exog(region_df, endog.index) if self._uses_exog() else None
            last_period = endog.index[-1]

            entry: dict = {
                "last_period": last_period,
                "endog": endog.to_numpy(dtype=float),
                "result": None,
                "exog_history": exog,
            }

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model = self._build_sarimax(
                        endog, exog, self.order, seasonal_order_full
                    )
                    entry["result"] = model.fit(disp=False)
            except Exception as exc:  # noqa: BLE001 - isolate a per-region failure
                self.fit_notes[region] = (
                    f"SARIMAX fit failed ({type(exc).__name__}: {exc}); "
                    "using seasonal-naive fallback for this region."
                )

            self._models[region] = entry

    def predict(self, horizon: int) -> Forecast:
        """Forecast ``horizon`` periods ahead for every region, recombined long.

        For each region the fitted model forecasts ``horizon`` future periods
        (starting one Time_Grain step after that region's last training period).
        Regions that fell back at fit time are forecast with a seasonal-naive rule.
        Forecasts are clipped at zero (demand cannot be negative) and stacked into
        a single :class:`~src.models.base.Forecast` whose ``index`` is a
        ``(period, region)`` MultiIndex sorted by ``period`` then ``region`` -
        aligning to the holdout grid every model shares (Requirement 5.7).

        Args:
            horizon: Number of periods to forecast per region (the Holdout_Set
                length).

        Returns:
            A :class:`~src.models.base.Forecast` with ``values`` (one non-negative
            forecast per ``(period, region)``) and a matching ``(period, region)``
            MultiIndex.

        Raises:
            RuntimeError: If called before :meth:`fit`.
            ValueError: If ``horizon`` is not a positive integer.
        """
        if self._freq is None or not self._models:
            raise RuntimeError("Sarima.predict called before fit.")
        if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon <= 0:
            raise ValueError(f"horizon must be a positive integer; got {horizon!r}.")

        offset = pd.tseries.frequencies.to_offset(self._freq)
        frames: list[pd.DataFrame] = []

        for region in self._regions:
            entry = self._models[region]
            future_periods = pd.date_range(
                start=entry["last_period"] + offset,
                periods=horizon,
                freq=self._freq,
            )

            result = entry["result"]
            if result is not None:
                exog_future = (
                    self._future_exog(region, future_periods)
                    if self._uses_exog()
                    else None
                )
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    forecast_obj = result.get_forecast(steps=horizon, exog=exog_future)
                    values = np.asarray(forecast_obj.predicted_mean, dtype=float)
            else:
                values = self._seasonal_naive(entry["endog"], horizon)

            # Demand is a non-negative count; clip any negative point forecasts.
            values = np.clip(values, a_min=0.0, a_max=None)

            frames.append(
                pd.DataFrame(
                    {
                        PERIOD_COLUMN: future_periods,
                        REGION_COLUMN: region,
                        DEMAND_COLUMN: values,
                    }
                )
            )

        combined = pd.concat(frames, ignore_index=True)
        combined = combined.sort_values(
            [PERIOD_COLUMN, REGION_COLUMN], kind="stable"
        ).reset_index(drop=True)

        index = pd.MultiIndex.from_arrays(
            [combined[PERIOD_COLUMN], combined[REGION_COLUMN]],
            names=[PERIOD_COLUMN, REGION_COLUMN],
        )
        return Forecast(
            model_name=self.name,
            values=combined[DEMAND_COLUMN].to_numpy(dtype=float),
            index=index,
        )

    # --- helpers -------------------------------------------------------------

    def _build_sarimax(self, endog, exog, order, seasonal_order):
        """Construct a ``statsmodels`` SARIMAX model (imported lazily)."""
        # Imported here so merely importing this module does not require the heavy
        # statsmodels state-space machinery until a model is actually fit.
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        return SARIMAX(
            endog,
            exog=exog,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=self.enforce_stationarity,
            enforce_invertibility=self.enforce_invertibility,
        )

    def _seasonal_naive(self, endog: np.ndarray, horizon: int) -> np.ndarray:
        """Seasonal-naive fallback forecast for a region that could not be fit.

        Repeats the last full seasonal cycle (``s`` values) forward; when there is
        less than one season of history it carries the last observed value (or the
        series mean for an empty tail). Documented fallback so the model still
        emits a complete, aligned forecast (Requirement 5.8).
        """
        s = self._s or 1
        endog = np.asarray(endog, dtype=float)
        if endog.size == 0:
            return np.zeros(horizon, dtype=float)
        if endog.size >= s:
            season = endog[-s:]
            reps = int(np.ceil(horizon / s))
            return np.tile(season, reps)[:horizon]
        return np.full(horizon, endog[-1], dtype=float)


class Sarimax(Sarima):
    """SARIMA with exogenous regressors (R5.2).

    Extends :class:`Sarima` by supplying an exogenous design matrix to each
    per-region SARIMAX model. By default the exogenous variables are deterministic
    day-of-week calendar indicators derived from the ``period`` timestamps, which
    are known for any future date (see the module docstring). Alternatively,
    ``exog_columns`` names existing :class:`~src.preparation.DemandSeries` columns
    (e.g. lag features) to use as exogenous inputs; those are extended over the
    forecast horizon with a seasonal-naive rule since they are not otherwise known
    ahead of time.

    Attributes:
        name: Model name (``"SARIMAX"``).
        exog_columns: Optional list of DemandSeries columns to use as exogenous
            regressors. When ``None`` (default), day-of-week calendar indicators
            are generated from the period timestamps.
    """

    name: str = "SARIMAX"

    def __init__(
        self,
        order: tuple[int, int, int] = DEFAULT_ORDER,
        seasonal_order: tuple[int, int, int] = DEFAULT_SEASONAL_ORDER,
        seasonal_periods: Optional[int] = None,
        *,
        exog_columns: Optional[Sequence[str]] = None,
        enforce_stationarity: bool = False,
        enforce_invertibility: bool = False,
    ) -> None:
        super().__init__(
            order,
            seasonal_order,
            seasonal_periods,
            enforce_stationarity=enforce_stationarity,
            enforce_invertibility=enforce_invertibility,
        )
        self.exog_columns: Optional[list[str]] = (
            list(exog_columns) if exog_columns is not None else None
        )

    def _uses_exog(self) -> bool:
        return True

    def _fit_exog(self, region_df: pd.DataFrame, periods: pd.DatetimeIndex):
        """Build the training exogenous matrix for one region.

        Uses the named ``exog_columns`` when provided, else deterministic
        day-of-week calendar indicators derived from ``periods``. The result is
        indexed by ``periods`` so it aligns with the region's endogenous series.
        """
        if self.exog_columns is None:
            return _calendar_exog(periods)

        missing = [c for c in self.exog_columns if c not in region_df.columns]
        if missing:
            raise KeyError(
                f"Exogenous column(s) {missing} not found in the DemandSeries. "
                f"Available columns: {list(region_df.columns)}."
            )
        exog = region_df[self.exog_columns].to_numpy(dtype=float)
        frame = pd.DataFrame(exog, columns=list(self.exog_columns), index=periods)
        # Lag features are NaN for the first k periods; fill so SARIMAX can fit.
        return frame.fillna(0.0)

    def _future_exog(self, region: str, future_periods: pd.DatetimeIndex):
        """Build the exogenous matrix over the forecast horizon for one region.

        For calendar exogenous variables the values are computed directly from the
        future timestamps (fully known). For named ``exog_columns`` the future
        values are extended by a seasonal-naive rule from the training history.
        """
        horizon = len(future_periods)
        if self.exog_columns is None:
            return _calendar_exog(future_periods)

        history = self._models[region].get("exog_history")
        if history is None or len(history) == 0:
            # No history to extend from: contribute zeros (documented fallback).
            return pd.DataFrame(
                np.zeros((horizon, len(self.exog_columns))),
                columns=list(self.exog_columns),
                index=future_periods,
            )

        s = self._s or 1
        hist_values = np.asarray(history.to_numpy(dtype=float))
        n = hist_values.shape[0]
        rows = []
        for h in range(horizon):
            if n >= s:
                src = hist_values[-s + (h % s)]
            else:
                src = hist_values[-1]
            rows.append(src)
        return pd.DataFrame(
            np.vstack(rows), columns=list(self.exog_columns), index=future_periods
        )
