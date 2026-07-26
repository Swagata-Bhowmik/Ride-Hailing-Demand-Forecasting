"""Unit tests for driver-positioning recommendation generation (src/business.py).

Covers Requirement 7.1: the Business_Module derives a driver-positioning
recommendation from the selected forecast at the defined Time_Grain and
Geographic_Grain. These are example-based unit tests that construct small
sample :class:`~src.models.base.Forecast` objects and assert
:func:`~src.business.positioning_recommendation` produces the expected
recommendation.

Three shapes of forecast are exercised, matching the two index shapes the
modeling layer produces plus the empty edge case:

* a multi-region ``(period, region)`` MultiIndex forecast (the primary case);
* a univariate forecast over a plain ``DatetimeIndex`` (region is ``None``); and
* an empty forecast (no values at all).

Validates: Requirements 7.1
"""

from __future__ import annotations

import pandas as pd

from src.business import Placement, Recommendation, positioning_recommendation
from src.config import default_scope
from src.models.base import Forecast


# --------------------------------------------------------------------------- #
# R7.1 - multi-region forecast: (period, region) MultiIndex
# --------------------------------------------------------------------------- #


def _multi_region_forecast() -> Forecast:
    """A 2-period, 2-region forecast at (period, region) grain.

    Period P1: Manhattan=100, Brooklyn=50  -> top region Manhattan
    Period P2: Manhattan=80,  Brooklyn=200 -> top region Brooklyn (global peak)
    """
    p1 = pd.Timestamp("2026-04-01")
    p2 = pd.Timestamp("2026-04-02")
    index = pd.MultiIndex.from_tuples(
        [(p1, "Manhattan"), (p1, "Brooklyn"), (p2, "Manhattan"), (p2, "Brooklyn")],
        names=["period", "region"],
    )
    values = [100.0, 50.0, 80.0, 200.0]
    return Forecast(model_name="SARIMAX", values=values, index=index)


class TestMultiRegionRecommendation:
    """R7.1: a recommendation is produced for a multi-region forecast."""

    def test_produces_recommendation_instance(self) -> None:
        rec = positioning_recommendation(_multi_region_forecast(), default_scope())
        assert isinstance(rec, Recommendation)

    def test_identifies_global_peak_region_period(self) -> None:
        rec = positioning_recommendation(_multi_region_forecast(), default_scope())
        # Brooklyn on P2 has the highest predicted demand (200) across the horizon.
        assert rec.region == "Brooklyn"
        assert rec.period == pd.Timestamp("2026-04-02")
        assert rec.predicted_demand == 200.0

    def test_one_placement_per_period_with_top_region(self) -> None:
        rec = positioning_recommendation(_multi_region_forecast(), default_scope())
        # Two distinct periods -> two placements, each the top region for its period.
        assert len(rec.placements) == 2
        assert all(isinstance(p, Placement) for p in rec.placements)

        by_period = {p.period: p for p in rec.placements}
        p1 = pd.Timestamp("2026-04-01")
        p2 = pd.Timestamp("2026-04-02")
        assert by_period[p1].region == "Manhattan"
        assert by_period[p1].predicted_demand == 100.0
        assert by_period[p2].region == "Brooklyn"
        assert by_period[p2].predicted_demand == 200.0

    def test_grains_populated_from_scope(self) -> None:
        scope = default_scope()
        rec = positioning_recommendation(_multi_region_forecast(), scope)
        assert rec.time_grain == scope.time_grain == "daily"
        assert rec.geographic_grain == scope.geographic_grain == "borough"

    def test_action_is_non_empty_and_mentions_peak_region(self) -> None:
        rec = positioning_recommendation(_multi_region_forecast(), default_scope())
        assert isinstance(rec.action, str) and rec.action.strip()
        assert "Brooklyn" in rec.action

    def test_impact_not_yet_computed(self) -> None:
        # positioning_recommendation leaves impact for quantify_impact to fill in.
        rec = positioning_recommendation(_multi_region_forecast(), default_scope())
        assert rec.impact is None


# --------------------------------------------------------------------------- #
# R7.1 - univariate forecast: plain DatetimeIndex, region is None
# --------------------------------------------------------------------------- #


class TestUnivariateRecommendation:
    """R7.1: a recommendation is produced for a univariate (region-less) forecast."""

    def _univariate_forecast(self) -> Forecast:
        index = pd.DatetimeIndex(
            [pd.Timestamp("2026-04-01"), pd.Timestamp("2026-04-02"), pd.Timestamp("2026-04-03")]
        )
        return Forecast(model_name="Holt-Winters", values=[10.0, 20.0, 15.0], index=index)

    def test_produces_recommendation_with_no_region(self) -> None:
        rec = positioning_recommendation(self._univariate_forecast(), default_scope())
        assert isinstance(rec, Recommendation)
        assert rec.region is None

    def test_identifies_peak_period(self) -> None:
        rec = positioning_recommendation(self._univariate_forecast(), default_scope())
        # Highest value (20) is on the second period.
        assert rec.period == pd.Timestamp("2026-04-02")
        assert rec.predicted_demand == 20.0

    def test_one_placement_per_period_regionless(self) -> None:
        rec = positioning_recommendation(self._univariate_forecast(), default_scope())
        assert len(rec.placements) == 3
        assert all(p.region is None for p in rec.placements)

    def test_grains_still_populated_from_scope(self) -> None:
        scope = default_scope()
        rec = positioning_recommendation(self._univariate_forecast(), scope)
        assert rec.time_grain == "daily"
        assert rec.geographic_grain == "borough"


# --------------------------------------------------------------------------- #
# R7.1 - empty forecast edge case
# --------------------------------------------------------------------------- #


class TestEmptyForecast:
    """R7.1: an empty forecast yields a well-defined, empty recommendation."""

    def test_empty_forecast_zero_demand_no_placements(self) -> None:
        empty = Forecast(model_name="SARIMA", values=[], index=[])
        rec = positioning_recommendation(empty, default_scope())

        assert isinstance(rec, Recommendation)
        assert rec.region is None
        assert rec.period is None
        assert rec.predicted_demand == 0.0
        assert rec.placements == []
        # Grains are still recorded from scope even with no data.
        assert rec.time_grain == "daily"
        assert rec.geographic_grain == "borough"
        # An informative action is still provided.
        assert isinstance(rec.action, str) and rec.action.strip()
