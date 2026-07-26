"""Example-based unit tests for the EDA_Module wrappers (src/eda.py).

Per the design's Testing Strategy ("Example-based unit tests"), the EDA
functions wrap third-party statistical tools (statsmodels' seasonal
decomposition, the ADF test, and pandas correlation) that are already
property-tested upstream, so here we assert their *contract* on a concrete,
realistic synthetic series rather than generating inputs:

- ``seasonal_decompose_demand`` returns three components of correct length (R3.2),
- ``adf_test`` returns a statistic and a p-value in the valid ``[0, 1]`` range (R3.3),
- ``demand_correlations`` returns a matrix with all values within ``[-1, 1]`` (R3.6).

Two supporting checks (anomaly detection flags an injected spike, ACF/PACF
returns value arrays) round out the coverage of the module's remaining wrappers.

The module forces the non-interactive matplotlib Agg backend at import time, so
these tests are headless-safe. A fixture closes all figures after each test to
keep matplotlib's figure registry clean and avoid "too many open figures"
warnings.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from src.eda import (
    ADFResult,
    Anomaly,
    EDAResult,
    acf_pacf,
    adf_test,
    demand_correlations,
    detect_anomalies,
    seasonal_decompose_demand,
)
from src.preparation import DEMAND_COLUMN, PERIOD_COLUMN, REGION_COLUMN

# Number of days in the synthetic series: comfortably more than 90 days and more
# than two full weekly cycles (2 * period = 14) so seasonal_decompose can run.
N_DAYS = 120
PERIOD = 7  # weekly seasonality for daily data


@pytest.fixture(autouse=True)
def _close_figures():
    """Close every matplotlib figure after each test to keep the registry clean."""
    yield
    plt.close("all")


@pytest.fixture
def daily_demand() -> pd.Series:
    """A synthetic daily demand series with trend + weekly seasonality + noise.

    Returns a period-indexed ``pandas.Series`` of length ``N_DAYS`` so component
    lengths and index alignment are unambiguous. The construction is deterministic
    (fixed seed) so the tests are stable across runs.
    """
    rng = np.random.default_rng(42)
    index = pd.date_range("2025-05-01", periods=N_DAYS, freq="D")
    t = np.arange(N_DAYS)
    trend = 1000 + 5 * t  # gentle upward trend
    weekly = 200 * np.sin(2 * np.pi * t / PERIOD)  # weekly seasonal swing
    noise = rng.normal(0, 25, N_DAYS)
    demand = trend + weekly + noise
    return pd.Series(demand, index=index, name=DEMAND_COLUMN)


@pytest.fixture
def daily_demand_frame(daily_demand: pd.Series) -> pd.DataFrame:
    """The synthetic series as a long-format DemandSeries DataFrame (one region).

    Confirms the wrappers also accept the long format the rest of the pipeline
    speaks, reducing it to the system-wide total per period.
    """
    return pd.DataFrame(
        {
            PERIOD_COLUMN: daily_demand.index,
            REGION_COLUMN: "Manhattan",
            DEMAND_COLUMN: daily_demand.to_numpy(),
        }
    )


# --------------------------------------------------------------------------- #
# R3.2 - seasonal decomposition returns three components of correct length
# --------------------------------------------------------------------------- #


class TestSeasonalDecompose:
    def test_returns_edaresult_with_figure_and_interpretation(self, daily_demand):
        result = seasonal_decompose_demand(daily_demand, period=PERIOD)
        assert isinstance(result, EDAResult)
        assert result.figure is not None
        assert isinstance(result.interpretation, str)
        assert result.interpretation.strip() != ""

    def test_components_have_input_length(self, daily_demand):
        """R3.2: trend, seasonal and residual each match the input series length."""
        result = seasonal_decompose_demand(daily_demand, period=PERIOD)
        stats = result.stats
        assert stats is not None
        for component in ("trend", "seasonal", "residual"):
            assert component in stats
            assert len(stats[component]) == len(daily_demand)
        # The observed component and metadata are surfaced too.
        assert len(stats["observed"]) == len(daily_demand)
        assert stats["period"] == PERIOD
        assert stats["model"] == "additive"

    def test_accepts_long_format_frame(self, daily_demand_frame, daily_demand):
        """The wrapper reduces a DemandSeries DataFrame to one period-indexed series."""
        result = seasonal_decompose_demand(daily_demand_frame, period=PERIOD)
        for component in ("trend", "seasonal", "residual"):
            assert len(result.stats[component]) == len(daily_demand)

    def test_multiplicative_model_supported(self, daily_demand):
        result = seasonal_decompose_demand(daily_demand, period=PERIOD, model="multiplicative")
        assert result.stats["model"] == "multiplicative"
        assert len(result.stats["seasonal"]) == len(daily_demand)

    def test_rejects_period_below_two(self, daily_demand):
        with pytest.raises(ValueError):
            seasonal_decompose_demand(daily_demand, period=1)

    def test_rejects_series_shorter_than_two_cycles(self):
        short = pd.Series(
            np.arange(10, dtype=float),
            index=pd.date_range("2025-05-01", periods=10, freq="D"),
        )
        with pytest.raises(ValueError):
            seasonal_decompose_demand(short, period=PERIOD)


# --------------------------------------------------------------------------- #
# R3.3 - ADF returns statistic and p-value in valid range
# --------------------------------------------------------------------------- #


class TestADFTest:
    def test_returns_statistic_and_pvalue_in_range(self, daily_demand):
        """R3.3: statistic is a float and the p-value lies within [0, 1]."""
        result = adf_test(daily_demand)
        assert isinstance(result, ADFResult)
        assert isinstance(result.statistic, float)
        assert isinstance(result.p_value, float)
        assert 0.0 <= result.p_value <= 1.0

    def test_stationary_is_bool_and_interpretation_present(self, daily_demand):
        result = adf_test(daily_demand)
        assert isinstance(result.stationary, bool)
        assert isinstance(result.interpretation, str)
        assert result.interpretation.strip() != ""

    def test_stationary_flag_matches_alpha_convention(self, daily_demand):
        result = adf_test(daily_demand, alpha=0.05)
        assert result.stationary == (result.p_value < 0.05)

    def test_critical_values_reported(self, daily_demand):
        result = adf_test(daily_demand)
        assert isinstance(result.critical_values, dict)
        # statsmodels reports the standard 1%/5%/10% critical values.
        assert {"1%", "5%", "10%"}.issubset(set(result.critical_values.keys()))
        for value in result.critical_values.values():
            assert isinstance(value, float)

    def test_accepts_long_format_frame(self, daily_demand_frame):
        result = adf_test(daily_demand_frame)
        assert 0.0 <= result.p_value <= 1.0

    def test_rejects_too_short_series(self):
        short = pd.Series([1.0, 2.0, 3.0])
        with pytest.raises(ValueError):
            adf_test(short)


# --------------------------------------------------------------------------- #
# R3.6 - correlation matrix within [-1, 1]
# --------------------------------------------------------------------------- #


class TestDemandCorrelations:
    def test_values_within_unit_interval_with_calendar_features(self, daily_demand):
        """R3.6: with derived calendar features, all correlations lie in [-1, 1]."""
        corr = demand_correlations(daily_demand)
        assert isinstance(corr, pd.DataFrame)
        values = corr.to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        assert finite.size > 0
        assert np.all(finite >= -1.0 - 1e-9)
        assert np.all(finite <= 1.0 + 1e-9)

    def test_demand_is_row_and_column(self, daily_demand):
        corr = demand_correlations(daily_demand)
        assert "demand" in corr.index
        assert "demand" in corr.columns
        # A variable's correlation with itself is 1.
        assert corr.loc["demand", "demand"] == pytest.approx(1.0)

    def test_interpretation_attached_to_attrs(self, daily_demand):
        """R3.7: the plain-language interpretation travels on the frame's attrs."""
        corr = demand_correlations(daily_demand)
        assert "interpretation" in corr.attrs
        assert isinstance(corr.attrs["interpretation"], str)
        assert corr.attrs["interpretation"].strip() != ""

    def test_correlated_exog_variable_scores_high(self, daily_demand):
        """A variable built from demand should correlate strongly with demand,
        and every matrix entry still lies within [-1, 1] (R3.6)."""
        rng = np.random.default_rng(7)
        exog = pd.DataFrame(
            {
                # Strongly correlated with demand (demand + small noise).
                "correlated": daily_demand.to_numpy() + rng.normal(0, 10, len(daily_demand)),
                # Independent random noise as a control.
                "unrelated": rng.normal(0, 100, len(daily_demand)),
            },
            index=daily_demand.index,
        )
        corr = demand_correlations(daily_demand, exog=exog)

        # All values remain within the valid correlation range.
        values = corr.to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        assert np.all(finite >= -1.0 - 1e-9)
        assert np.all(finite <= 1.0 + 1e-9)

        # The correlated variable is present and strongly related to demand.
        assert "correlated" in corr.columns
        assert corr.loc["demand", "correlated"] > 0.9

    def test_accepts_long_format_frame(self, daily_demand_frame):
        corr = demand_correlations(daily_demand_frame)
        assert "demand" in corr.columns
        values = corr.to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        assert np.all((finite >= -1.0 - 1e-9) & (finite <= 1.0 + 1e-9))


# --------------------------------------------------------------------------- #
# Supporting coverage: anomaly detection and ACF/PACF arrays
# --------------------------------------------------------------------------- #


class TestDetectAnomalies:
    def test_flags_injected_spike(self, daily_demand):
        """An injected extreme spike is detected and reported as a 'spike'."""
        spiked = daily_demand.copy()
        spike_period = spiked.index[60]
        spiked.loc[spike_period] = spiked.max() * 5  # unmistakable spike

        anomalies = detect_anomalies(spiked, window=PERIOD, threshold=3.5)
        assert isinstance(anomalies, list)
        assert all(isinstance(a, Anomaly) for a in anomalies)
        flagged_periods = {a.period for a in anomalies}
        assert pd.Timestamp(spike_period) in flagged_periods
        spike = next(a for a in anomalies if a.period == pd.Timestamp(spike_period))
        assert spike.direction == "spike"
        assert spike.description.strip() != ""

    def test_rejects_invalid_arguments(self, daily_demand):
        with pytest.raises(ValueError):
            detect_anomalies(daily_demand, window=1)
        with pytest.raises(ValueError):
            detect_anomalies(daily_demand, threshold=0)


class TestAcfPacf:
    def test_returns_acf_and_pacf_arrays(self, daily_demand):
        result = acf_pacf(daily_demand)
        assert isinstance(result, EDAResult)
        assert result.figure is not None
        stats = result.stats
        assert stats is not None
        assert isinstance(stats["acf"], np.ndarray)
        assert isinstance(stats["pacf"], np.ndarray)
        # lag 0 plus `lags` lags -> arrays of length lags + 1.
        assert len(stats["acf"]) == stats["lags"] + 1
        assert len(stats["pacf"]) == stats["lags"] + 1
        # ACF at lag 0 is 1 by construction.
        assert stats["acf"][0] == pytest.approx(1.0)
