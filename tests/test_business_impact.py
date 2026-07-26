"""Property-based test for impact calculation reproducibility (Property 12).

Design reference:
- Correctness Properties -> Property 12 (Impact calculation reproducibility)
- Requirements 7.3 (WHERE a quantified benefit is stated, the Business_Module
  SHALL show the assumptions and calculation used to derive it).

This module implements the *single* Hypothesis property test the design assigns
to Property 12, at 100+ iterations, per the Testing Strategy. It exercises
``src.business.quantify_impact``: given a :class:`~src.business.Recommendation`
carrying a ``predicted_demand`` and an explicit set of planning ``assumptions``,
the quantified benefit it reports must equal the value obtained by *independently*
recomputing the documented formula (``IMPACT_FORMULA``) from those same effective
assumptions.

The documented formula (from ``src.business.IMPACT_FORMULA``) is::

    drivers_positioned        = predicted_demand / trips_per_driver
    rider_wait_minutes_saved  = predicted_demand * baseline_wait_minutes * wait_reduction_pct
    driver_idle_minutes_saved = drivers_positioned * baseline_idle_minutes * idle_reduction_pct
    total_minutes_saved       = rider_wait_minutes_saved + driver_idle_minutes_saved

``quantify_impact`` merges caller-supplied assumptions over
``DEFAULT_IMPACT_ASSUMPTIONS``. The generator draws an *arbitrary subset* of the
assumption keys (sometimes none) so the test also covers the documented default
fallback: the "effective" assumptions the benefit must match are computed here,
independently, as ``defaults <- generated overrides``.

The generator constrains inputs to the valid space:

* ``predicted_demand`` >= 0 (a trip count);
* ``trips_per_driver`` strictly positive (the function raises otherwise);
* ``wait_reduction_pct`` / ``idle_reduction_pct`` in [0, 1] (fractions);
* ``baseline_wait_minutes`` / ``baseline_idle_minutes`` >= 0 within sensible
  minute ranges.

All values are drawn as finite floats (no NaN/inf) so the arithmetic - and the
``pytest.approx`` comparison - is well defined.
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.business import (
    DEFAULT_IMPACT_ASSUMPTIONS,
    IMPACT_FORMULA,
    Recommendation,
    quantify_impact,
)

# Finite, sensible ranges for each assumption. Rates are fractions in [0, 1];
# baselines are non-negative minute counts; trips_per_driver is strictly
# positive (quantify_impact raises for non-positive values).
_FINITE = dict(allow_nan=False, allow_infinity=False)


@st.composite
def recommendation_and_assumptions(draw: st.DrawFn):
    """Generate ``(recommendation, assumptions)`` for the impact calculation.

    * ``predicted_demand`` is a non-negative finite float (trip count), including
      the ``0.0`` edge case;
    * ``assumptions`` is an arbitrary subset of the five documented keys, each
      drawn from its valid range, so unspecified keys exercise the default
      fallback. ``trips_per_driver`` (when present) is strictly positive.
    """
    predicted_demand = draw(
        st.floats(min_value=0.0, max_value=1_000_000.0, **_FINITE)
    )

    # A pool of valid values keyed by assumption name; each key is independently
    # included or omitted so we cover full, partial, and empty override sets.
    value_strategies = {
        "baseline_wait_minutes": st.floats(min_value=0.0, max_value=120.0, **_FINITE),
        "wait_reduction_pct": st.floats(min_value=0.0, max_value=1.0, **_FINITE),
        "baseline_idle_minutes": st.floats(min_value=0.0, max_value=120.0, **_FINITE),
        "idle_reduction_pct": st.floats(min_value=0.0, max_value=1.0, **_FINITE),
        # Strictly positive: exclude 0 so quantify_impact does not raise.
        "trips_per_driver": st.floats(
            min_value=0.1, max_value=100.0, exclude_min=False, **_FINITE
        ),
    }

    assumptions: dict[str, float] = {}
    for key, strategy in value_strategies.items():
        if draw(st.booleans()):
            assumptions[key] = draw(strategy)

    recommendation = Recommendation(
        region="Manhattan",
        period="2026-04-15",
        predicted_demand=predicted_demand,
        action="test placement",
    )
    return recommendation, assumptions


# Feature: ride-hailing-demand-forecasting, Property 12: Impact calculation
# reproducibility - the quantified benefit equals recomputing the documented
# formula from the same assumptions
#
# Validates: Requirements 7.3
@settings(max_examples=200)
@given(data=recommendation_and_assumptions())
def test_impact_calculation_reproducibility(data) -> None:
    recommendation, assumptions = data

    statement = quantify_impact(recommendation, assumptions)

    # --- Independent recomputation of the documented formula --------------- #
    # Effective assumptions: documented defaults overridden by any supplied keys.
    effective = dict(DEFAULT_IMPACT_ASSUMPTIONS)
    effective.update(assumptions)

    predicted_demand = float(recommendation.predicted_demand)
    trips_per_driver = effective["trips_per_driver"]

    expected_drivers = predicted_demand / trips_per_driver
    expected_wait = (
        predicted_demand
        * effective["baseline_wait_minutes"]
        * effective["wait_reduction_pct"]
    )
    expected_idle = (
        expected_drivers
        * effective["baseline_idle_minutes"]
        * effective["idle_reduction_pct"]
    )
    expected_total = expected_wait + expected_idle

    # --- Each component matches the independent recomputation -------------- #
    assert statement.drivers_positioned == pytest.approx(expected_drivers)
    assert statement.rider_wait_minutes_saved == pytest.approx(expected_wait)
    assert statement.driver_idle_minutes_saved == pytest.approx(expected_idle)
    assert statement.total_minutes_saved == pytest.approx(expected_total)

    # --- The total is internally consistent with its own components -------- #
    assert statement.total_minutes_saved == pytest.approx(
        statement.rider_wait_minutes_saved + statement.driver_idle_minutes_saved
    )

    # --- Assumptions and formula are surfaced for reproducibility (R7.3) --- #
    assert statement.assumptions == effective
    assert statement.formula == IMPACT_FORMULA
    assert statement.predicted_demand == pytest.approx(predicted_demand)
