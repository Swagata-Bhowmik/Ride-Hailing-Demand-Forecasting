"""Property-based test for zero-fill completeness (Property 4).

Design reference:
- Correctness Properties -> Property 4 (Zero-fill completeness)
- Requirements 4.3 (empty periods represented explicitly as zero demand rather
  than omitted) and 13.4 (transformation output re-validated against intent).

This module implements the *single* Hypothesis property test the design assigns
to Property 4, at 100 iterations, per the Testing Strategy. It exercises
``src.preparation.fill_missing_periods``: given a demand series holding some
observed ``(period, region)`` buckets and an explicit region set, the function
must materialize the complete, contiguous grid across the Analysis_Window.

The property has three parts:

1. **Completeness** - the result contains *exactly one* row for every
   ``(period, region)`` combination in the window (every calendar day from
   ``scope.window_start`` to ``scope.window_end`` inclusive, crossed with every
   region), none omitted and none duplicated. Concretely: the row count equals
   ``n_periods * n_regions`` and the set of ``(period, region)`` keys equals the
   full cartesian product.
2. **Zero-fill correctness** - ``demand`` is exactly ``0`` for every
   ``(period, region)`` that had no input record, and equals the observed demand
   wherever an input record existed.
3. **Integer dtype** - the ``demand`` column is an integer type.

Tractability: ``ScopeConfig`` validates the Analysis_Window to 12-24 months
(``__post_init__``), so a short window cannot be constructed directly. The test
therefore uses the real ~12-month default scope (365 periods) but constrains the
region set to 1-2 regions, keeping each example at 365-730 rows so 100 iterations
run quickly. Observed buckets are a small random subset of the full grid with
non-zero demand, so the "zero where no input existed" assertion is meaningful.
"""

from __future__ import annotations

import pandas as pd
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from src.config import default_scope
from src.preparation import (
    DEMAND_COLUMN,
    PERIOD_COLUMN,
    REGION_COLUMN,
    fill_missing_periods,
)

# The Geographic_Grain values (NYC boroughs + EWR). The test constrains the
# region set to a 1-2 element prefix of this pool so the 365-day grid stays at
# 365-730 rows per example, keeping 100 iterations fast.
BOROUGHS = ["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island", "EWR"]


@st.composite
def observed_buckets(draw: st.DrawFn):
    """Generate ``(regions, observed)`` where ``observed`` is a random subset of grid.

    * ``regions`` is a 1-2 borough prefix (single-region series is a key edge
      case for zero-fill).
    * ``observed`` maps a small, unique set of ``(period, region)`` buckets drawn
      from *within* the default scope window to a non-zero demand value. Buckets
      not chosen represent the empty periods that must be zero-filled.

    Uniqueness of the ``(period, region)`` keys is enforced so each observed
    bucket has an unambiguous expected demand (no duplicate-bucket summing to
    reason about here - that is covered by the aggregation property).
    """
    scope = default_scope()
    all_periods = list(
        pd.date_range(
            start=pd.Timestamp(scope.window_start),
            end=pd.Timestamp(scope.window_end),
            freq="D",
        )
    )
    n_periods = len(all_periods)

    n_regions = draw(st.integers(min_value=1, max_value=2))
    regions = BOROUGHS[:n_regions]

    # Draw a modest number of unique (period_index, region) keys from the grid,
    # including the empty case (no observed buckets -> entire grid zero-filled).
    keys = draw(
        st.lists(
            st.tuples(
                st.integers(min_value=0, max_value=n_periods - 1),
                st.sampled_from(regions),
            ),
            min_size=0,
            max_size=40,
            unique=True,
        )
    )

    observed: dict[tuple[pd.Timestamp, str], int] = {}
    for period_idx, region in keys:
        demand = draw(st.integers(min_value=1, max_value=5000))
        observed[(all_periods[period_idx], region)] = demand

    return regions, observed


def _observed_to_series(observed: dict[tuple[pd.Timestamp, str], int]) -> pd.DataFrame:
    """Build a long-format DemandSeries from the observed-bucket mapping."""
    if not observed:
        return pd.DataFrame(
            {
                PERIOD_COLUMN: pd.Series([], dtype="datetime64[ns]"),
                REGION_COLUMN: pd.Series([], dtype="object"),
                DEMAND_COLUMN: pd.Series([], dtype="int64"),
            }
        )
    periods = [p for (p, _r) in observed]
    regions = [r for (_p, r) in observed]
    demands = list(observed.values())
    return pd.DataFrame(
        {PERIOD_COLUMN: periods, REGION_COLUMN: regions, DEMAND_COLUMN: demands}
    )


# Feature: ride-hailing-demand-forecasting, Property 4: Zero-fill completeness -
# exactly one row per (period, region) in the window, none omitted, demand 0
# where no input existed
#
# Validates: Requirements 4.3, 13.4
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
@given(data=observed_buckets())
def test_zero_fill_completeness(data) -> None:
    regions, observed = data
    scope = default_scope()
    series = _observed_to_series(observed)

    filled = fill_missing_periods(series, scope, regions=regions)

    # Ground-truth full grid: every calendar day in the window x every region.
    all_periods = list(
        pd.date_range(
            start=pd.Timestamp(scope.window_start),
            end=pd.Timestamp(scope.window_end),
            freq="D",
        )
    )
    n_periods = len(all_periods)
    expected_keys = {(p, r) for p in all_periods for r in regions}

    # --- Part 1: completeness - exactly one row per (period, region) ------- #
    assert len(filled) == n_periods * len(regions)

    result_keys = list(zip(filled[PERIOD_COLUMN], filled[REGION_COLUMN]))
    # No (period, region) appears more than once.
    assert len(result_keys) == len(set(result_keys))
    # Exactly the full cartesian product - none omitted, none extra.
    assert set(result_keys) == expected_keys

    # --- Part 3: demand dtype is integer ----------------------------------- #
    assert pd.api.types.is_integer_dtype(filled[DEMAND_COLUMN])

    # --- Part 2: 0 where no input existed, observed value where present ---- #
    actual_demand = {
        (period, region): int(demand)
        for period, region, demand in zip(
            filled[PERIOD_COLUMN], filled[REGION_COLUMN], filled[DEMAND_COLUMN]
        )
    }
    for key in expected_keys:
        expected = observed.get(key, 0)
        assert actual_demand[key] == expected
