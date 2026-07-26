"""VAR/VARMAX multivariate forecaster for joint multi-region demand (R5.3).

Requirement 5.3 asks the Forecasting_System to train a multivariate model from
the VAR/VARMAX family for *joint* multi-region forecasting. Where the univariate
families (Holt-Winters, SARIMA) forecast one region's series in isolation, a
Vector Autoregression treats every region's demand as one component of a single
vector-valued series and forecasts them together, so cross-region dynamics (a
surge in Manhattan leading demand in Brooklyn, say) are captured.

This module implements :class:`VarVarmax`, a thin :class:`~src.models.base.Forecaster`
around ``statsmodels`` ``VAR``. It:

* pivots the long-format :class:`~src.preparation.DemandSeries` (``period`` /
  ``region`` / ``demand``) to a wide matrix with one column per region and one
  row per period - the natural shape for a vector-valued series;
* fits a VAR jointly across all region columns;
* forecasts ``horizon`` periods ahead for every region at once, returning a
  single :class:`~src.models.base.Forecast` whose ``values`` is a wide
  ``DataFrame`` (rows = holdout periods, columns = regions) aligned to the
  holdout index.

**Single-region guard.** A VAR needs at least two endogenous series; there is no
"vector" to model with one region. Rather than fit a degenerate model,
:meth:`VarVarmax.fit` raises a clear :class:`ValueError` when only one region is
present. ``train_all`` (see :mod:`src.models.base`) catches that and records an
:class:`~src.models.base.ExclusionRecord`, so VAR is excluded gracefully from the
comparison instead of aborting the run (R5.8).

Design references:
- Components and Interfaces -> Forecasting_System (`src/models/`)
- Data Models -> Forecast (values aligned to the Holdout_Set index, R5.7)
- Error Handling -> VAR on a single region becomes an ExclusionRecord (R5.8)
- Requirements 5.3
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import pandas as pd

from src.models.base import Forecast
from src.preparation import (
    DEMAND_COLUMN,
    PERIOD_COLUMN,
    REGION_COLUMN,
    _TIME_GRAIN_FREQ,
)

if TYPE_CHECKING:  # pragma: no cover - imported only for type checking
    from src.config import ScopeConfig
    from src.preparation import DemandSeries


def _pivot_to_wide(
    train: "DemandSeries",
    *,
    period_column: str = PERIOD_COLUMN,
    region_column: str = REGION_COLUMN,
    demand_column: str = DEMAND_COLUMN,
) -> pd.DataFrame:
    """Reshape a long DemandSeries into a wide (period x region) demand matrix.

    A VAR models a *vector* of series, one component per region, indexed by time.
    The long DemandSeries (one row per ``(period, region)``) is pivoted so each
    region becomes a column and each period a row - exactly that vector-valued
    layout. Any missing ``(period, region)`` cell is filled with ``0`` (the same
    convention the zero-fill preparation step uses: an absent bucket means no
    trips), and the rows are sorted by period so the series is chronologically
    ordered.

    Args:
        train: The training :class:`~src.preparation.DemandSeries`.
        period_column: Name of the period column. Defaults to ``"period"``.
        region_column: Name of the region column. Defaults to ``"region"``.
        demand_column: Name of the demand column. Defaults to ``"demand"``.

    Returns:
        A wide ``DataFrame`` indexed by ``period`` (a ``DatetimeIndex``) with one
        float column per region, sorted chronologically.

    Raises:
        KeyError: If any of the required columns is absent from ``train``.
    """
    for col in (period_column, region_column, demand_column):
        if col not in train.columns:
            raise KeyError(
                f"Column '{col}' not found in training series. "
                f"Available columns: {list(train.columns)}."
            )

    wide = (
        train.pivot_table(
            index=period_column,
            columns=region_column,
            values=demand_column,
            aggfunc="sum",
            fill_value=0,
        )
        .sort_index()
    )
    # Drop the columns' name ("region") so the frame's columns are just regions.
    wide.columns.name = None
    wide.index = pd.to_datetime(wide.index)
    return wide.astype(float)


class VarVarmax:
    """Joint multi-region forecaster backed by ``statsmodels`` VAR (R5.3).

    Fits a single Vector Autoregression across every region's demand series at
    once, so forecasts for all regions are produced jointly and cross-region
    dependencies are modeled. Implements the
    :class:`~src.models.base.Forecaster` interface (``name`` / ``fit`` /
    ``predict``) so the Evaluation_Framework can train and score it uniformly with
    the other families.

    ``predict`` returns a single :class:`~src.models.base.Forecast` covering *all*
    regions: its ``values`` is a wide ``pandas.DataFrame`` with one row per
    holdout period and one column per region (same column order as seen at fit
    time), and its ``index`` is the holdout ``DatetimeIndex`` those rows align to.
    This "one Forecast, many region columns" convention reflects that VAR is a
    joint model - the regions are not forecast independently.

    Attributes:
        name: ``"VAR"`` - the model name used in the comparison table and in any
            :class:`~src.models.base.ExclusionRecord`.
        maxlags: Upper bound on the VAR lag order to consider. ``None`` (default)
            picks a conservative bound from the training data's size and width.
        ic: Information criterion used to select the lag order up to ``maxlags``
            (e.g. ``"aic"``); when ``None`` the model is fit at exactly ``maxlags``.
    """

    def __init__(self, maxlags: Optional[int] = None, ic: Optional[str] = "aic") -> None:
        self.name = "VAR"
        self.maxlags = maxlags
        self.ic = ic
        # Populated by fit():
        self._results = None
        self._columns: Optional[pd.Index] = None
        self._lag_order: int = 0
        self._last_obs = None
        self._last_period: Optional[pd.Timestamp] = None
        self._freq: str = "D"

    def fit(self, train: "DemandSeries", scope: "ScopeConfig") -> None:
        """Fit a VAR jointly across all region series in ``train`` (R5.3).

        Pivots the long training series to a wide (period x region) matrix and
        fits a ``statsmodels`` ``VAR`` over all region columns together. The lag
        order is chosen by the configured information criterion up to a
        conservative ``maxlags`` bound (kept small on short fixtures so the model
        stays estimable), and is forced to at least 1 so :meth:`predict` always
        has lagged observations to roll forward from.

        Args:
            train: The training :class:`~src.preparation.DemandSeries` (periods
                before the Holdout_Set).
            scope: The project :class:`~src.config.ScopeConfig`, supplying the
                Time_Grain used to derive the forecast period frequency.

        Raises:
            ValueError: If fewer than two regions are present (a VAR needs a
                vector of at least two series - this becomes an
                :class:`~src.models.base.ExclusionRecord` via ``train_all``, R5.8),
                or if there is not enough history to estimate even a lag-1 VAR.
            KeyError: If ``train`` is missing the period/region/demand columns.
        """
        # Imported lazily so importing this module never hard-requires statsmodels
        # until a VAR is actually trained.
        from statsmodels.tsa.api import VAR

        wide = _pivot_to_wide(train)
        n_obs, n_vars = wide.shape

        if n_vars < 2:
            raise ValueError(
                "VAR requires at least two regions for joint multi-region "
                f"forecasting; got {n_vars} region(s) "
                f"({list(wide.columns)}). With a single region use a univariate "
                "model (Holt-Winters / SARIMA) instead."
            )

        # A VAR of lag p over k series consumes p observations and estimates
        # k * (k*p + 1) coefficients, so it needs comfortably more rows than
        # lags. Pick a conservative upper bound that keeps the model estimable on
        # short fixtures, then let the information criterion choose within it.
        if self.maxlags is not None:
            maxlags = max(1, int(self.maxlags))
        else:
            maxlags = max(1, min(7, (n_obs - 1) // (n_vars + 1)))

        if n_obs <= maxlags + 1:
            raise ValueError(
                f"Not enough history to fit a VAR: {n_obs} periods for "
                f"{n_vars} regions with lag order up to {maxlags}."
            )

        model = VAR(wide)
        if self.ic is not None:
            results = model.fit(maxlags=maxlags, ic=self.ic)
            # An IC can select order 0 (no dynamics); forecasting then has no
            # lagged state to roll forward, so force at least a lag-1 fit.
            if getattr(results, "k_ar", 0) < 1:
                results = model.fit(1)
        else:
            results = model.fit(maxlags)

        self._results = results
        self._columns = wide.columns
        self._lag_order = int(getattr(results, "k_ar", 1)) or 1
        # VAR.forecast needs the last ``lag_order`` observations to seed the roll.
        self._last_obs = wide.values[-self._lag_order :]
        self._last_period = wide.index[-1]

        grain = scope.time_grain.lower()
        self._freq = _TIME_GRAIN_FREQ.get(grain, "D")

    def predict(self, horizon: int) -> Forecast:
        """Forecast ``horizon`` periods ahead for every region jointly (R5.7).

        Rolls the fitted VAR forward ``horizon`` steps from the last observed
        window, producing a joint forecast for all regions. The result is a single
        :class:`~src.models.base.Forecast` whose ``values`` is a wide
        ``DataFrame`` (rows = the ``horizon`` holdout periods, columns = regions
        in fit-time order) and whose ``index`` is the holdout ``DatetimeIndex``
        those rows align to - the same horizon every other model forecasts over.

        Args:
            horizon: Number of periods to forecast, equal to the Holdout_Set
                length.

        Returns:
            A :class:`~src.models.base.Forecast` covering all regions jointly.

        Raises:
            RuntimeError: If called before :meth:`fit`.
            ValueError: If ``horizon`` is not a positive integer.
        """
        if self._results is None:
            raise RuntimeError("VarVarmax.predict called before fit.")
        if horizon <= 0:
            raise ValueError(f"horizon must be a positive integer; got {horizon}.")

        raw = self._results.forecast(self._last_obs, steps=horizon)

        index = pd.date_range(
            start=self._last_period + pd.tseries.frequencies.to_offset(self._freq),
            periods=horizon,
            freq=self._freq,
        )
        values = pd.DataFrame(raw, index=index, columns=self._columns)

        return Forecast(model_name=self.name, values=values, index=index)


#: Alias so callers can refer to the family by the shorter ``Var`` name; the two
#: refer to the same joint VAR-based implementation.
Var = VarVarmax
