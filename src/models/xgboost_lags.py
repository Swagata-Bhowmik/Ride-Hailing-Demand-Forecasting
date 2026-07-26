"""XGBoost model driven by lag (and calendar) features (Requirement 5.5).

This is the machine-learning candidate in the Candidate_Model_Set. Unlike the
classical time-series families (Holt-Winters, SARIMA, VAR, Prophet) that model
the demand series directly, :class:`XGBoostLags` treats forecasting as a
supervised regression problem: it learns a mapping from *engineered predictors*
to demand and applies that mapping period by period.

The predictors are exactly the lag features the Data_Preparation_Pipeline already
produces (``lag_1``, ``lag_7``, ``lag_14``, ... - see
:func:`src.preparation.add_lag_features`, Requirement 4.6), enriched with a few
deterministic calendar features derived from the period (day-of-week, month,
etc. - the "calendar features" row of the design's DemandSeries schema) and a
region indicator so a single model serves every borough at once.

Multi-step (recursive) forecasting strategy
--------------------------------------------
A lag-feature regressor can only predict one period ahead, because predicting
period ``t`` needs ``demand[t-k]`` for each lag ``k``. To forecast the whole
Holdout_Set we therefore roll forward **recursively**:

* Seed a per-region history with the training actuals.
* Walk the holdout periods in chronological order. For each future period ``t``
  and each region, look each lag ``k`` back to period ``t - k``:
  - if ``t - k`` falls in the training range, the lag is a real observed value;
  - if ``t - k`` is itself a already-forecast holdout period, the lag is the
    model's own earlier prediction (this is the "recursive" part);
  - if ``t - k`` predates all available history, the lag is ``NaN`` (XGBoost
    handles missing values natively via its default split direction).
* Predict every region for period ``t`` in one batch, clip to the non-negative
  demand domain, then append those predictions to the history so the next
  period's short lags can see them.

Because ``predict(horizon)`` is called with ``horizon`` equal to the Holdout_Set
length, the forecast covers the same holdout as every other model (Requirement
5.7), and its ``index`` is the ``(period, region)`` grid of that holdout so the
Evaluation_Framework can align it to the actuals.

Graceful optional dependency
-----------------------------
``xgboost`` is an optional heavyweight dependency. It is imported defensively at
module load: a missing/broken install does **not** crash the import (so the rest
of ``src.models`` keeps working). Instead the failure surfaces later, when
:meth:`XGBoostLags.fit` is called, as a clear exception - which ``train_all``
turns into an :class:`~src.models.base.ExclusionRecord` rather than aborting the
whole comparison (Requirement 5.8).

Design references:
- Components and Interfaces -> Forecasting_System (`src/models/`), XGBoostLags
- Data Models -> DemandSeries (lag features R4.6, calendar features)
- Error Handling -> per-model exclusion recorded, one failure never aborts others
- Requirements 5.5, 5.7, 5.8
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from src.models.base import Forecast
from src.preparation import (
    DEMAND_COLUMN,
    PERIOD_COLUMN,
    REGION_COLUMN,
    add_lag_features,
    lag_column_name,
)

if TYPE_CHECKING:  # pragma: no cover - imported only for type checking
    from src.config import ScopeConfig
    from src.preparation import DemandSeries

# --- Graceful optional-dependency import (Requirement 5.8) -------------------
# xgboost is a heavy optional dependency. Importing it must never crash module
# load; a missing install becomes an exclusion at fit time, not an import error.
try:  # pragma: no cover - exercised implicitly by whichever environment runs
    from xgboost import XGBRegressor as _XGBRegressor

    _XGBOOST_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # noqa: BLE001 - any import failure must degrade gracefully
    _XGBRegressor = None  # type: ignore[assignment]
    _XGBOOST_IMPORT_ERROR = exc


#: Prefix for the one-hot region-indicator columns added to the feature matrix.
_REGION_FEATURE_PREFIX = "region_"

#: Fallback step between periods when it cannot be inferred from the training
#: data (e.g. a degenerate single-period training set). One day matches the
#: project's default daily Time_Grain.
_DEFAULT_STEP = pd.Timedelta(days=1)


def _calendar_features(periods: "pd.Series | pd.DatetimeIndex") -> pd.DataFrame:
    """Derive deterministic calendar features from a period column.

    These are the "calendar features" of the design's DemandSeries schema:
    purely a function of the timestamp (no leakage, no fitting), so they can be
    computed identically for training rows and for future holdout rows. Demand is
    strongly weekly/seasonal, so day-of-week and month in particular give the
    regressor the seasonal signal the lag features alone do not encode.

    Args:
        periods: The ``period`` timestamps to derive features from.

    Returns:
        A DataFrame (indexed like ``periods``) of integer calendar features.
    """
    idx = pd.DatetimeIndex(pd.to_datetime(pd.Series(periods).to_numpy()))
    frame = pd.DataFrame(
        {
            "cal_dayofweek": idx.dayofweek,
            "cal_day": idx.day,
            "cal_month": idx.month,
            "cal_dayofyear": idx.dayofyear,
            "cal_is_weekend": (idx.dayofweek >= 5).astype(int),
        }
    )
    frame.index = pd.Index(pd.Series(periods).index)
    return frame


def _infer_step(periods: "pd.Series") -> pd.Timedelta:
    """Infer the spacing between consecutive periods from the training data.

    Uses the most common positive gap between distinct sorted periods, which is
    robust to the occasional missing period and independent of the configured
    Time_Grain (so it keeps working if new grains are added later). Falls back to
    one day when the gap cannot be determined.
    """
    unique = pd.Series(pd.to_datetime(periods)).drop_duplicates().sort_values()
    if len(unique) < 2:
        return _DEFAULT_STEP
    diffs = unique.diff().dropna()
    positive = diffs[diffs > pd.Timedelta(0)]
    if positive.empty:
        return _DEFAULT_STEP
    # The modal gap is the natural period step (e.g. 1 day at the daily grain).
    return positive.mode().iloc[0]


class XGBoostLags:
    """Gradient-boosted regressor over lag + calendar features (Requirement 5.5).

    Implements the :class:`~src.models.base.Forecaster` interface. A single
    :class:`xgboost.XGBRegressor` is trained on every region's rows at once, with
    a one-hot region indicator so the model can specialize per borough while still
    sharing statistical strength across them. Forecasting over the Holdout_Set is
    done recursively (see the module docstring).

    Attributes:
        name: Model name used as the key in ``TrainedModel``/``ExclusionRecord``
            and the evaluation comparison table.
    """

    def __init__(self, name: str = "XGBoost", **xgb_params: Any) -> None:
        """Create an (unfitted) XGBoost lag-feature forecaster.

        Constructing this object never imports or requires ``xgboost`` - the
        dependency is only needed at :meth:`fit`. Any keyword arguments are
        forwarded to :class:`xgboost.XGBRegressor`, over sensible defaults tuned
        for small demand series.

        Args:
            name: Human-readable model name (defaults to ``"XGBoost"``).
            **xgb_params: Extra parameters passed through to ``XGBRegressor``.
        """
        self.name = name
        # Modest defaults: enough trees to fit weekly/annual structure without
        # overfitting a short daily series; callers can override any of them.
        self._xgb_params: dict[str, Any] = {
            "n_estimators": 300,
            "max_depth": 4,
            "learning_rate": 0.05,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "random_state": 0,
            "n_jobs": 1,
            "objective": "reg:squarederror",
        }
        self._xgb_params.update(xgb_params)

        # Populated by fit():
        self.model: Any = None
        self._lags: list[int] = []
        self._regions: list[str] = []
        self._feature_columns: list[str] = []
        self._step: pd.Timedelta = _DEFAULT_STEP
        self._last_period: pd.Timestamp | None = None
        # region -> {period -> demand} history seeded from the training actuals.
        self._train_history: dict[str, dict[pd.Timestamp, float]] = {}

    # -- feature engineering --------------------------------------------------

    def _build_feature_matrix(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Assemble the model's feature matrix from a long-format frame.

        The frame must carry ``period``, ``region`` and one ``lag_{k}`` column per
        fitted lag. The returned matrix has exactly ``self._feature_columns`` in
        the same order used at fit time, so training and prediction always present
        identical features to the model.
        """
        features: dict[str, np.ndarray] = {}

        for k in self._lags:
            col = lag_column_name(k)
            features[col] = pd.to_numeric(frame[col], errors="coerce").to_numpy(
                dtype="float64"
            )

        calendar = _calendar_features(frame[PERIOD_COLUMN])
        for col in calendar.columns:
            features[col] = calendar[col].to_numpy(dtype="float64")

        # One-hot region indicator using the region set seen at fit time, so the
        # column layout is stable even if a prediction batch omits some regions.
        region_values = frame[REGION_COLUMN].to_numpy()
        for region in self._regions:
            features[f"{_REGION_FEATURE_PREFIX}{region}"] = (
                region_values == region
            ).astype("float64")

        matrix = pd.DataFrame(features, index=frame.index)
        return matrix.reindex(columns=self._feature_columns)

    # -- Forecaster interface -------------------------------------------------

    def fit(self, train: "DemandSeries", scope: "ScopeConfig") -> None:
        """Train the regressor on the training portion of the series.

        Consumes the lag features produced by the preparation pipeline (recomputing
        them from ``scope.lags`` if they are not already present, so the model is
        robust to being handed a bare ``DemandSeries``), derives calendar features
        and the region indicator, and fits an :class:`xgboost.XGBRegressor` to
        predict ``demand``. Rows whose lag features are ``NaN`` (the first ``k``
        periods of each region) are kept - XGBoost routes missing values on its
        own - so no training history is discarded.

        Args:
            train: The training :class:`~src.preparation.DemandSeries` (periods
                before the Holdout_Set), with ``period``/``region``/``demand`` and,
                ideally, ``lag_{k}`` columns.
            scope: The project :class:`~src.config.ScopeConfig` supplying ``lags``.

        Raises:
            RuntimeError: If ``xgboost`` is not importable (so ``train_all`` records
                an exclusion instead of crashing - Requirement 5.8).
            KeyError: If ``train`` lacks the required period/region/demand columns.
            ValueError: If ``train`` has no rows to learn from.
        """
        if _XGBRegressor is None:
            raise RuntimeError(
                "xgboost is not available, so the XGBoost model is excluded: "
                f"{_XGBOOST_IMPORT_ERROR!r}. Install 'xgboost' to enable it."
            )

        for col in (PERIOD_COLUMN, REGION_COLUMN, DEMAND_COLUMN):
            if col not in train.columns:
                raise KeyError(
                    f"Column '{col}' not found in training series. "
                    f"Available columns: {list(train.columns)}."
                )
        if train.empty:
            raise ValueError("Cannot fit XGBoostLags on an empty training series.")

        self._lags = list(dict.fromkeys(int(k) for k in scope.lags))
        # Sorted, stable region ordering for a deterministic one-hot layout.
        self._regions = sorted(pd.unique(train[REGION_COLUMN].dropna()).tolist())

        work = train.copy()
        work[PERIOD_COLUMN] = pd.to_datetime(work[PERIOD_COLUMN])

        # Ensure every requested lag column exists; recompute any that are missing
        # so the model works whether it is handed a fully-prepared series or a bare
        # (period, region, demand) one.
        missing_lags = [k for k in self._lags if lag_column_name(k) not in work.columns]
        if missing_lags:
            work = add_lag_features(work, self._lags)

        # Fix the feature-column order once, here, so predict() reproduces it.
        self._feature_columns = (
            [lag_column_name(k) for k in self._lags]
            + ["cal_dayofweek", "cal_day", "cal_month", "cal_dayofyear", "cal_is_weekend"]
            + [f"{_REGION_FEATURE_PREFIX}{r}" for r in self._regions]
        )

        X = self._build_feature_matrix(work)
        y = pd.to_numeric(work[DEMAND_COLUMN], errors="coerce").to_numpy(dtype="float64")

        self.model = _XGBRegressor(**self._xgb_params)
        self.model.fit(X, y)

        # Record the period spacing, the last training period, and per-region
        # history so predict() can roll lag features forward recursively.
        self._step = _infer_step(work[PERIOD_COLUMN])
        self._last_period = pd.Timestamp(work[PERIOD_COLUMN].max())
        self._train_history = {region: {} for region in self._regions}
        for period, region, demand in zip(
            work[PERIOD_COLUMN], work[REGION_COLUMN], y
        ):
            if region in self._train_history:
                self._train_history[region][pd.Timestamp(period)] = float(demand)

    def predict(self, horizon: int) -> Forecast:
        """Recursively forecast ``horizon`` periods over the Holdout_Set.

        Rolls the model forward one period at a time (see the module docstring):
        each future period's lag features are read from a history that blends
        training actuals with the model's own earlier predictions, so lags shorter
        than the horizon stay populated. Predictions are clipped to the
        non-negative demand domain. The returned :class:`Forecast` carries a
        ``(period, region)`` MultiIndex covering the holdout grid, sorted by period
        then region (matching the preparation pipeline's ordering) so it aligns to
        the actual holdout values.

        Args:
            horizon: Number of periods to forecast (the Holdout_Set length, R5.7).

        Returns:
            A :class:`Forecast` with one value per ``(period, region)`` holdout
            cell and a matching MultiIndex.

        Raises:
            RuntimeError: If called before :meth:`fit`.
            ValueError: If ``horizon`` is not a positive integer.
        """
        if self.model is None or self._last_period is None:
            raise RuntimeError("XGBoostLags.predict called before fit.")
        if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon <= 0:
            raise ValueError(f"horizon must be a positive integer; got {horizon!r}.")

        # Fresh copy of the seed history so repeated predict() calls are pure.
        history: dict[str, dict[pd.Timestamp, float]] = {
            region: dict(values) for region, values in self._train_history.items()
        }

        future_periods = [
            self._last_period + (i + 1) * self._step for i in range(horizon)
        ]

        records: list[tuple[pd.Timestamp, str, float]] = []
        for period in future_periods:
            # Build one row per region for this period, reading lags from history.
            rows: list[dict[str, Any]] = []
            for region in self._regions:
                row: dict[str, Any] = {PERIOD_COLUMN: period, REGION_COLUMN: region}
                region_hist = history[region]
                for k in self._lags:
                    prior = period - k * self._step
                    row[lag_column_name(k)] = region_hist.get(prior, np.nan)
                rows.append(row)

            batch = pd.DataFrame(rows)
            X = self._build_feature_matrix(batch)
            preds = np.clip(np.asarray(self.model.predict(X), dtype="float64"), 0.0, None)

            # Commit this period's predictions to history before advancing, so the
            # next period's short lags can reference them (recursive strategy).
            for region, pred in zip(self._regions, preds):
                history[region][period] = float(pred)
                records.append((period, region, float(pred)))

        result = pd.DataFrame(
            records, columns=[PERIOD_COLUMN, REGION_COLUMN, "prediction"]
        ).sort_values(by=[PERIOD_COLUMN, REGION_COLUMN], kind="stable").reset_index(
            drop=True
        )

        index = pd.MultiIndex.from_arrays(
            [result[PERIOD_COLUMN], result[REGION_COLUMN]],
            names=[PERIOD_COLUMN, REGION_COLUMN],
        )
        return Forecast(
            model_name=self.name,
            values=result["prediction"].to_numpy(dtype="float64"),
            index=index,
        )
