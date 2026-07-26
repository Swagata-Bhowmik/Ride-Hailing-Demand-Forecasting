"""Integration tests for each forecasting model family (Requirement 5).

Design reference:
- Testing Strategy -> Integration tests: "each model family (Holt-Winters,
  SARIMA/SARIMAX, VAR/VARMAX, Prophet, XGBoost, LSTM/GRU) fits on a small fixture
  series and produces a forecast of expected length (R5.1-5.6) - 1-3 examples per
  model, not property tests, because fitting is stochastic and expensive."
- Components and Interfaces -> Forecasting_System (`src/models/`)
- Error Handling -> a candidate model failing is isolated as an ExclusionRecord.

These are **example / integration** tests (not property tests): each candidate
family is fit on a small, deterministic, in-memory multi-region daily fixture
(never the real ~1 GB NYC TLC dataset) and asserted to produce a forecast whose
length and index align to the Holdout_Set (R5.7). Optional heavy dependencies
(prophet, xgboost, tensorflow) are skipped gracefully with ``pytest.importorskip``
so the suite stays green in a minimal environment. A final test drives
``train_all`` and asserts an :class:`ExclusionRecord` is recorded on a forced
failure (R5.8) without aborting the other candidates.

Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import default_scope
from src.models.base import ExclusionRecord, Forecast, TrainedModel, train_all
from src.preparation import (
    DEMAND_COLUMN,
    PERIOD_COLUMN,
    REGION_COLUMN,
    add_lag_features,
)

# --- Fixture parameters ------------------------------------------------------

#: Small multi-region fixture: 3 boroughs over ~90 contiguous daily periods -
#: enough history for every family (incl. SARIMA's weekly season and VAR's lags)
#: to fit, but tiny enough to keep the suite fast. Never the real dataset.
FIXTURE_REGIONS = ["Manhattan", "Brooklyn", "Queens"]
FIXTURE_DAYS = 90
FIXTURE_START = pd.Timestamp("2025-01-01")
HORIZON = 14  # Holdout_Set length (periods every model forecasts over).


def _make_demand_series() -> pd.DataFrame:
    """Build a small, valid, deterministic multi-region daily DemandSeries.

    Each region gets a contiguous daily series with a gentle trend, a weekly
    seasonal wave, a per-region level, and small deterministic noise, then rounded
    to a non-negative integer trip count - the shape a real prepared
    DemandSeries has (period / region / demand, complete and contiguous). Lag
    features (from the default scope) are appended so the ML model can consume
    them directly, matching the design's DemandSeries schema.
    """
    rng = np.random.default_rng(20240517)
    periods = pd.date_range(start=FIXTURE_START, periods=FIXTURE_DAYS, freq="D")
    day_index = np.arange(FIXTURE_DAYS)

    frames = []
    for offset, region in enumerate(FIXTURE_REGIONS):
        level = 200.0 + 60.0 * offset
        trend = 0.5 * day_index
        weekly = 30.0 * np.sin(2 * np.pi * day_index / 7.0)
        noise = rng.normal(0.0, 5.0, size=FIXTURE_DAYS)
        demand = np.clip(np.rint(level + trend + weekly + noise), 0, None).astype(
            "int64"
        )
        frames.append(
            pd.DataFrame(
                {
                    PERIOD_COLUMN: periods,
                    REGION_COLUMN: region,
                    DEMAND_COLUMN: demand,
                }
            )
        )

    series = pd.concat(frames, ignore_index=True)
    series = series.sort_values(
        [PERIOD_COLUMN, REGION_COLUMN], kind="stable"
    ).reset_index(drop=True)

    # Append lag features (default scope lags: 1, 7, 14) so the XGBoost family can
    # consume them; the first k periods of each region are NaN by construction.
    series = add_lag_features(series, list(default_scope().lags))
    return series


@pytest.fixture(scope="module")
def demand_series() -> pd.DataFrame:
    """The full fixture DemandSeries (module-scoped: built once, reused)."""
    return _make_demand_series()


@pytest.fixture(scope="module")
def split(demand_series: pd.DataFrame):
    """Split the fixture into (train, holdout, holdout_periods, regions, scope).

    The Holdout_Set is the most-recent ``HORIZON`` contiguous periods; training is
    everything earlier. Because the series is contiguous daily, the periods each
    model forecasts (the ``HORIZON`` days after the last training period) equal the
    holdout periods exactly, which is what R5.7 alignment is checked against.
    """
    unique_periods = np.sort(demand_series[PERIOD_COLUMN].unique())
    holdout_periods = pd.DatetimeIndex(unique_periods[-HORIZON:])
    train_periods = set(unique_periods[:-HORIZON])

    train = demand_series[demand_series[PERIOD_COLUMN].isin(train_periods)].copy()
    holdout = demand_series[
        demand_series[PERIOD_COLUMN].isin(set(holdout_periods))
    ].copy()

    scope = default_scope()  # daily Time_Grain; lags [1, 7, 14].
    return train, holdout, holdout_periods, list(FIXTURE_REGIONS), scope


# --- Alignment assertion helpers --------------------------------------------


def _assert_per_region_forecast(forecast: Forecast, holdout_periods, regions) -> None:
    """Assert a per-region long forecast aligns to the (period, region) holdout.

    Holt-Winters, SARIMA/SARIMAX, XGBoost and LSTM/GRU return one value per
    ``(period, region)`` cell with a ``(period, region)`` MultiIndex. The forecast
    must cover exactly the holdout periods x regions grid (R5.7).
    """
    assert isinstance(forecast, Forecast)
    assert isinstance(forecast.index, pd.MultiIndex)
    assert forecast.index.names == [PERIOD_COLUMN, REGION_COLUMN]

    values = np.asarray(forecast.values, dtype="float64")
    assert len(values) == len(holdout_periods) * len(regions)
    assert len(forecast.index) == len(values)
    assert np.all(np.isfinite(values))
    assert np.all(values >= 0.0)  # demand is a non-negative count

    got_periods = forecast.index.get_level_values(PERIOD_COLUMN).unique()
    got_regions = forecast.index.get_level_values(REGION_COLUMN).unique()
    assert set(pd.DatetimeIndex(got_periods)) == set(holdout_periods)
    assert set(map(str, got_regions)) == set(map(str, regions))


def _assert_wide_forecast(forecast: Forecast, holdout_periods, regions) -> None:
    """Assert VAR's joint wide forecast aligns to the holdout (rows) x regions.

    VAR forecasts every region jointly, so ``values`` is a wide DataFrame with one
    row per holdout period and one column per region, and ``index`` is the holdout
    DatetimeIndex (R5.7).
    """
    assert isinstance(forecast, Forecast)
    assert isinstance(forecast.values, pd.DataFrame)
    assert forecast.values.shape == (len(holdout_periods), len(regions))
    assert set(map(str, forecast.values.columns)) == set(map(str, regions))
    assert list(pd.DatetimeIndex(forecast.index)) == list(holdout_periods)
    assert np.all(np.isfinite(forecast.values.to_numpy(dtype="float64")))


def _assert_single_series_forecast(forecast: Forecast, holdout_periods) -> None:
    """Assert a single-series forecast (Prophet total) aligns to the holdout index."""
    assert isinstance(forecast, Forecast)
    values = np.asarray(forecast.values, dtype="float64")
    assert len(values) == len(holdout_periods)
    assert list(pd.DatetimeIndex(forecast.index)) == list(holdout_periods)
    assert np.all(np.isfinite(values))


# --- Per-family integration tests -------------------------------------------


def test_holt_winters_fits_and_forecasts_holdout(split):
    """Holt-Winters baseline fits and forecasts the holdout (R5.1, R5.7)."""
    pytest.importorskip("statsmodels")
    from src.models.holt_winters import HoltWinters

    train, _holdout, holdout_periods, regions, scope = split

    model = HoltWinters()
    model.fit(train, scope)
    forecast = model.predict(HORIZON)

    assert model.name == "Holt-Winters"
    _assert_per_region_forecast(forecast, holdout_periods, regions)


def test_sarima_fits_and_forecasts_holdout(split):
    """SARIMA classical-univariate fits and forecasts the holdout (R5.2, R5.7)."""
    pytest.importorskip("statsmodels")
    from src.models.sarima import Sarima

    train, _holdout, holdout_periods, regions, scope = split

    model = Sarima()
    model.fit(train, scope)
    forecast = model.predict(HORIZON)

    assert model.name == "SARIMA"
    _assert_per_region_forecast(forecast, holdout_periods, regions)


def test_sarimax_fits_and_forecasts_holdout(split):
    """SARIMAX (exogenous variant) fits and forecasts the holdout (R5.2, R5.7)."""
    pytest.importorskip("statsmodels")
    from src.models.sarima import Sarimax

    train, _holdout, holdout_periods, regions, scope = split

    # Default calendar (day-of-week) exogenous regressors - known for any future
    # date, so no exogenous-forecast leakage.
    model = Sarimax()
    model.fit(train, scope)
    forecast = model.predict(HORIZON)

    assert model.name == "SARIMAX"
    _assert_per_region_forecast(forecast, holdout_periods, regions)


def test_var_fits_and_forecasts_holdout_jointly(split):
    """VAR/VARMAX joint multivariate model fits and forecasts the holdout (R5.3, R5.7)."""
    pytest.importorskip("statsmodels")
    from src.models.var import VarVarmax

    train, _holdout, holdout_periods, regions, scope = split

    model = VarVarmax()
    model.fit(train, scope)
    forecast = model.predict(HORIZON)

    assert model.name == "VAR"
    _assert_wide_forecast(forecast, holdout_periods, regions)


def test_prophet_fits_and_forecasts_holdout(split):
    """Prophet fits and forecasts the holdout total series (R5.4, R5.7)."""
    pytest.importorskip("prophet")
    from src.models.prophet_model import ProphetModel

    train, _holdout, holdout_periods, _regions, scope = split

    model = ProphetModel()  # region=None -> total demand, single series.
    model.fit(train, scope)
    forecast = model.predict(HORIZON)

    assert model.name == "Prophet"
    _assert_single_series_forecast(forecast, holdout_periods)


def test_xgboost_fits_and_forecasts_holdout(split):
    """XGBoost lag-feature model fits and forecasts the holdout (R5.5, R5.7)."""
    pytest.importorskip("xgboost")
    from src.models.xgboost_lags import XGBoostLags

    train, _holdout, holdout_periods, regions, scope = split

    # Small tree count keeps the fixture fit fast without changing behavior.
    model = XGBoostLags(n_estimators=40)
    model.fit(train, scope)
    forecast = model.predict(HORIZON)

    assert model.name == "XGBoost"
    _assert_per_region_forecast(forecast, holdout_periods, regions)


@pytest.mark.parametrize("layer", ["LSTM", "GRU"])
def test_recurrent_models_fit_and_forecast_holdout(split, layer):
    """LSTM/GRU deep-learning models fit and forecast the holdout (R5.6, R5.7).

    Skipped gracefully when tensorflow is not installed; epochs kept tiny so the
    fixture fit stays fast.
    """
    pytest.importorskip("tensorflow")
    from src.models.lstm_gru import GRUModel, LSTMModel

    train, _holdout, holdout_periods, regions, scope = split

    model = LSTMModel(epochs=2) if layer == "LSTM" else GRUModel(epochs=2)
    model.fit(train, scope)
    forecast = model.predict(HORIZON)

    assert model.name == layer
    _assert_per_region_forecast(forecast, holdout_periods, regions)


# --- Forced-failure exclusion (R5.8) ----------------------------------------


class _BrokenForecaster:
    """A deliberately broken Forecaster whose ``fit`` always raises (test double)."""

    name = "BrokenModel"

    def fit(self, train, scope) -> None:  # noqa: D401 - test double
        raise RuntimeError("intentional fit failure for exclusion test")

    def predict(self, horizon: int) -> Forecast:  # pragma: no cover - never reached
        raise AssertionError("predict should not be called after fit failed")


def test_var_on_single_region_is_excluded(split):
    """VAR on a single-region series is recorded as an ExclusionRecord (R5.8)."""
    pytest.importorskip("statsmodels")
    from src.models.var import VarVarmax

    train, _holdout, _holdout_periods, _regions, scope = split
    single_region = train[train[REGION_COLUMN] == FIXTURE_REGIONS[0]].copy()

    results = train_all([VarVarmax()], single_region, scope, HORIZON)

    assert len(results) == 1
    record = results[0]
    assert isinstance(record, ExclusionRecord)
    assert record.model_name == "VAR"
    assert record.reason  # a non-empty reason is recorded


def test_train_all_isolates_failure_from_other_models(split):
    """A broken model yields an ExclusionRecord without aborting the others (R5.8)."""
    pytest.importorskip("statsmodels")
    from src.models.holt_winters import HoltWinters

    train, _holdout, holdout_periods, regions, scope = split

    # Broken model first, a healthy Holt-Winters second: the failure must be
    # isolated and the healthy model must still train and forecast.
    results = train_all(
        [_BrokenForecaster(), HoltWinters()], train, scope, HORIZON
    )

    assert len(results) == 2

    broken, healthy = results
    assert isinstance(broken, ExclusionRecord)
    assert broken.model_name == "BrokenModel"
    assert "intentional fit failure" in broken.reason

    assert isinstance(healthy, TrainedModel)
    assert healthy.model_name == "Holt-Winters"
    _assert_per_region_forecast(healthy.forecast, holdout_periods, regions)
