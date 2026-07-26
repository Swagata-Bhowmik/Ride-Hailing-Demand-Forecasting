"""Holt-Winters (Exponential Smoothing) baseline forecaster (Requirement 5.1).

This is the simplest candidate in the model set: classical Exponential Smoothing
(Holt-Winters) via ``statsmodels``. It is the *baseline* every other model is
measured against, so it deliberately stays simple and dependency-light.

Univariate design choice
-------------------------
Holt-Winters is a **univariate** method - it models one series at a time. The
project's :data:`~src.preparation.DemandSeries` is long-format and (at the
Borough Geographic_Grain) carries several parallel regional series. This module
handles that by **fitting one independent Holt-Winters model per region** and
producing a combined long-format :class:`~src.models.base.Forecast`. This keeps
the baseline consistent with the rest of the pipeline: the same long ``(period,
region, demand)`` shape flows in and a matching long forecast flows out, so the
Evaluation_Framework can line the forecast up against the multi-region
Holdout_Set exactly as it does for the multivariate models. (Fitting a single
model on demand aggregated across regions was the alternative, but that would
throw away the per-region signal every downstream component - evaluation,
error-by-period, the driver-positioning recommendation - is built around.)

Seasonality
-----------
At the daily Time_Grain the dominant cycle is weekly, so seasonal periods default
to 7. Additive trend and seasonality are used (demand can legitimately be 0 after
zero-fill, so multiplicative seasonality - which requires strictly positive data
- is avoided). When a region has too little history to estimate a full seasonal
cycle (fewer than two complete seasons), that region falls back to
non-seasonal Exponential Smoothing so ``fit`` never raises on short fixtures.

Design references:
- Components and Interfaces -> Forecasting_System (`src/models/`)
- Data Models -> Forecast (aligned to the Holdout_Set)
- Requirements 5.1, 5.7
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from src.models.base import Forecast
from src.preparation import DEMAND_COLUMN, PERIOD_COLUMN, REGION_COLUMN

if TYPE_CHECKING:  # pragma: no cover - imported only for type checking
    from src.config import ScopeConfig
    from src.preparation import DemandSeries

#: Time_Grain -> pandas date-offset alias used to generate the forecast periods.
#: Mirrors ``src.preparation._TIME_GRAIN_FREQ``; only the daily grain (the project
#: default) is supported for now.
_TIME_GRAIN_FREQ: dict[str, str] = {"daily": "D"}

#: Time_Grain -> dominant seasonal cycle length. Daily demand is driven by a
#: weekly rhythm (weekday vs weekend), so 7 periods per season.
_TIME_GRAIN_SEASONAL_PERIODS: dict[str, int] = {"daily": 7}


class HoltWinters:
    """Exponential Smoothing (Holt-Winters) baseline conforming to ``Forecaster``.

    Fits one univariate Holt-Winters model per region (see module docstring) and
    forecasts every region over the same horizon, returning a single long-format
    :class:`~src.models.base.Forecast`.

    Attributes:
        name: Model name used throughout evaluation (``"Holt-Winters"``).
    """

    name = "Holt-Winters"

    def __init__(self, *, seasonal_periods: int | None = None) -> None:
        """Create an unfitted Holt-Winters baseline.

        Args:
            seasonal_periods: Optional override for the seasonal cycle length. When
                ``None`` (default) it is derived from the scope's Time_Grain in
                :meth:`fit` (daily -> 7). Set explicitly for non-default grains or
                to force a particular season length on a fixture.
        """
        self._seasonal_periods_override = seasonal_periods

        # Populated by fit():
        self._fitted: dict[str, object] = {}
        self._regions: list[str] = []
        self._last_period: pd.Timestamp | None = None
        self._freq: str | None = None
        self._seasonal_periods: int | None = None

    def fit(self, train: "DemandSeries", scope: "ScopeConfig") -> None:
        """Fit one Holt-Winters model per region on the training series.

        Args:
            train: Training :class:`~src.preparation.DemandSeries` (long format
                with ``period``/``region``/``demand`` columns, the periods before
                the Holdout_Set).
            scope: Project :class:`~src.config.ScopeConfig` supplying the
                Time_Grain (which fixes the forecast frequency and the seasonal
                cycle length).

        Raises:
            KeyError: If ``train`` is missing a required column.
            ValueError: If the Time_Grain is unsupported or the series is empty.
        """
        for col in (PERIOD_COLUMN, REGION_COLUMN, DEMAND_COLUMN):
            if col not in train.columns:
                raise KeyError(
                    f"Column '{col}' not found in train series. "
                    f"Available columns: {list(train.columns)}."
                )

        grain = scope.time_grain.lower()
        if grain not in _TIME_GRAIN_FREQ:
            raise ValueError(
                f"Unsupported time_grain '{scope.time_grain}'. "
                f"Supported grains: {sorted(_TIME_GRAIN_FREQ)}."
            )
        self._freq = _TIME_GRAIN_FREQ[grain]
        self._seasonal_periods = (
            self._seasonal_periods_override
            if self._seasonal_periods_override is not None
            else _TIME_GRAIN_SEASONAL_PERIODS[grain]
        )

        work = train[[PERIOD_COLUMN, REGION_COLUMN, DEMAND_COLUMN]].copy()
        work[PERIOD_COLUMN] = pd.to_datetime(work[PERIOD_COLUMN])
        work = work.sort_values([PERIOD_COLUMN, REGION_COLUMN], kind="stable")

        if work.empty:
            raise ValueError("Cannot fit Holt-Winters on an empty training series.")

        self._regions = sorted(work[REGION_COLUMN].dropna().unique().tolist())
        self._last_period = work[PERIOD_COLUMN].max()

        self._fitted = {}
        for region in self._regions:
            region_series = (
                work.loc[work[REGION_COLUMN] == region]
                .set_index(PERIOD_COLUMN)[DEMAND_COLUMN]
                .astype(float)
                .sort_index()
            )
            self._fitted[region] = self._fit_one(region_series)

    def _fit_one(self, series: pd.Series):
        """Fit a single univariate Holt-Winters model, degrading gracefully.

        Uses additive trend + additive weekly seasonality when the region has at
        least two full seasons of history; otherwise falls back to non-seasonal
        Exponential Smoothing so short fixtures still fit. A regularly-spaced
        period index is attached so ``statsmodels`` treats the data as an evenly
        sampled series.

        Args:
            series: The region's demand indexed by period, ascending.

        Returns:
            A fitted ``statsmodels`` results object exposing ``forecast(steps)``.
        """
        values = series.to_numpy(dtype=float)
        n = len(values)
        seasonal_periods = self._seasonal_periods or 0

        # Attach a regular frequency index so statsmodels does not warn / guess.
        idx = pd.date_range(start=series.index.min(), periods=n, freq=self._freq)
        endog = pd.Series(values, index=idx)

        use_seasonal = seasonal_periods >= 2 and n >= 2 * seasonal_periods
        use_trend = n >= 3

        if use_seasonal:
            model = ExponentialSmoothing(
                endog,
                trend="add" if use_trend else None,
                seasonal="add",
                seasonal_periods=seasonal_periods,
                initialization_method="estimated",
            )
        else:
            model = ExponentialSmoothing(
                endog,
                trend="add" if use_trend else None,
                seasonal=None,
                initialization_method="estimated",
            )

        return model.fit()

    def predict(self, horizon: int) -> Forecast:
        """Forecast ``horizon`` periods ahead for every region.

        Produces a single long-format forecast: for each region the fitted model
        is rolled forward ``horizon`` steps, and the results are stacked into a
        forecast whose ``index`` is a ``MultiIndex`` of ``(period, region)``
        aligned to the Holdout_Set (the ``horizon`` periods immediately after the
        training data, one row per region). Ordering is period-then-region, the
        same ordering the prepared :class:`~src.preparation.DemandSeries` uses, so
        forecast and holdout line up directly.

        Args:
            horizon: Number of periods to forecast (the Holdout_Set length).

        Returns:
            A :class:`~src.models.base.Forecast` with ``values`` (float, one per
            ``(period, region)`` cell) and a ``(period, region)`` ``MultiIndex``.

        Raises:
            RuntimeError: If called before :meth:`fit`.
            ValueError: If ``horizon`` is not a positive integer.
        """
        if self._last_period is None or self._freq is None:
            raise RuntimeError("HoltWinters.predict called before fit.")
        if horizon <= 0:
            raise ValueError(f"horizon must be a positive integer, got {horizon}.")

        # The horizon periods immediately following the last training period.
        future_periods = pd.date_range(
            start=self._last_period,
            periods=horizon + 1,
            freq=self._freq,
        )[1:]

        frames = []
        for region in self._regions:
            fitted = self._fitted[region]
            forecast_values = np.asarray(fitted.forecast(horizon), dtype=float)
            frames.append(
                pd.DataFrame(
                    {
                        PERIOD_COLUMN: future_periods,
                        REGION_COLUMN: region,
                        DEMAND_COLUMN: forecast_values,
                    }
                )
            )

        combined = pd.concat(frames, ignore_index=True)
        combined = combined.sort_values(
            [PERIOD_COLUMN, REGION_COLUMN], kind="stable"
        ).reset_index(drop=True)

        index = pd.MultiIndex.from_frame(combined[[PERIOD_COLUMN, REGION_COLUMN]])
        return Forecast(
            model_name=self.name,
            values=combined[DEMAND_COLUMN].to_numpy(dtype=float),
            index=index,
        )
