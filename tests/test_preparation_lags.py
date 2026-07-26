"""Property-based test for lag feature correctness (Property 6).

Design reference:
- Correctness Properties -> Property 6 (Lag feature correctness)
- Requirements 4.6 (produce lag features required by the ML models).

This module implements the *single* Hypothesis property test the design assigns
to Property 6, at 100+ iterations, per the Testing Strategy. It exercises
``src.preparation.add_lag_features``: given a long-format demand series spanning
one or more regions, for every requested lag ``k`` the function must add a
``lag_{k}`` column where, *within each region* (rows ordered by period), the
value at period ``t`` equals the demand at period ``t - k`` and is ``NaN`` for
the first ``k`` periods of that region.

The property has four parts:

1. **Lag equality within region** - ordering each region's rows by period, the
   ``lag_{k}`` value at position ``i`` equals the demand at position ``i - k``
   for ``i >= k``.
2. **NaN prefix** - the first ``k`` periods of every region are ``NaN`` in
   ``lag_{k}``.
3. **No cross-region leakage** - because the expected values are computed strictly
   per region, a match proves a region's earliest periods never borrow demand
   from another region (regions deliberately share overlapping calendar dates).
4. **Structure preserved** - original columns (period, region, demand) are kept
   unchanged, row order/index is preserved, and the row count is unchanged.

The function takes no ``ScopeConfig``, so the generator is free to build small,
per-region contiguous daily series directly (1-4 regions, short lengths), varying
region count, per-region length, demand values, and the lag set, and optionally
shuffling the input row order to prove the function sorts internally.
"""

from __future__ import annotations

import math

import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st

from src.preparation import (
    DEMAND_COLUMN,
    PERIOD_COLUMN,
    REGION_COLUMN,
    add_lag_features,
    lag_column_name,
)

# The Geographic_Grain values (NYC boroughs + EWR). The generator draws a subset
# so region count varies from a single-region series (a key edge case) up to 4.
BOROUGHS = ["Manhattan", "Brooklyn", "Queens", "Bronx"]

# The candidate lag offsets. Every k >= 1; the generator draws a non-empty subset.
LAG_POOL = [1, 2, 3, 7]

# A common base date. Regions start within a small window of this date so their
# calendar periods overlap - if the shift ever leaked across regions, overlapping
# dates would make the leak observable.
BASE_DATE = pd.Timestamp("2025-01-01")


@st.composite
def demand_series_and_lags(draw: st.DrawFn):
    """Generate ``(series, lags)`` for a multi-region long-format demand series.

    * region count 1-4 (single region is an important edge case);
    * each region gets a contiguous daily sequence of 1-20 periods starting at a
      small offset from :data:`BASE_DATE`, so regions' dates overlap;
    * demand is a non-negative integer per (period, region);
    * ``lags`` is a non-empty unique subset of :data:`LAG_POOL` (all k >= 1);
    * the assembled rows are optionally shuffled to prove the function sorts
      internally rather than relying on input order.

    Within a region periods are unique (contiguous daily), so each (region,
    period) key maps to an unambiguous demand value.
    """
    n_regions = draw(st.integers(min_value=1, max_value=len(BOROUGHS)))
    regions = BOROUGHS[:n_regions]

    frames = []
    for region in regions:
        length = draw(st.integers(min_value=1, max_value=20))
        start_offset = draw(st.integers(min_value=0, max_value=10))
        start = BASE_DATE + pd.Timedelta(days=start_offset)
        periods = pd.date_range(start=start, periods=length, freq="D")
        demands = draw(
            st.lists(
                st.integers(min_value=0, max_value=5000),
                min_size=length,
                max_size=length,
            )
        )
        frames.append(
            pd.DataFrame(
                {
                    PERIOD_COLUMN: periods,
                    REGION_COLUMN: region,
                    DEMAND_COLUMN: demands,
                }
            )
        )

    series = pd.concat(frames, ignore_index=True)

    if draw(st.booleans()):
        # Shuffle rows (deterministically, driven by Hypothesis) to prove the
        # function orders by (region, period) internally.
        order = draw(st.permutations(list(range(len(series)))))
        series = series.iloc[list(order)].reset_index(drop=True)

    lags = draw(
        st.lists(st.sampled_from(LAG_POOL), min_size=1, max_size=len(LAG_POOL), unique=True)
    )

    return series, lags


# Feature: ride-hailing-demand-forecasting, Property 6: Lag feature correctness -
# within each region lag_k at t equals demand at t-k and is NaN for the first k
# periods
#
# Validates: Requirements 4.6
@settings(max_examples=200)
@given(data=demand_series_and_lags())
def test_lag_feature_correctness(data) -> None:
    series, lags = data
    original = series.copy()

    result = add_lag_features(series, lags)

    # --- Part 4a: row count and original columns preserved unchanged ------- #
    assert len(result) == len(original)
    for col in (PERIOD_COLUMN, REGION_COLUMN, DEMAND_COLUMN):
        assert col in result.columns
        # Same values in the same row order / index (only new columns added).
        assert list(result[col]) == list(original[col])
    assert list(result.index) == list(original.index)

    # --- Ground truth: per-region expected lag values --------------------- #
    # For each region, order its rows by period and shift demand by k. This is
    # computed strictly within a region, so a match proves no cross-region
    # leakage (Part 3) even though regions share overlapping calendar dates.
    expected: dict[tuple, dict[int, float]] = {}
    for region, group in original.groupby(REGION_COLUMN):
        ordered = group.sort_values(by=PERIOD_COLUMN, kind="stable")
        periods = list(ordered[PERIOD_COLUMN])
        demands = list(ordered[DEMAND_COLUMN])
        for i, period in enumerate(periods):
            key = (region, period)
            expected[key] = {}
            for k in lags:
                if i >= k:
                    expected[key][k] = float(demands[i - k])
                else:
                    # Part 2: the first k periods of each region are NaN.
                    expected[key][k] = math.nan

    # --- Verify each requested lag column exists and matches ground truth -- #
    for k in lags:
        col = lag_column_name(k)
        assert col in result.columns

    for _, row in result.iterrows():
        key = (row[REGION_COLUMN], row[PERIOD_COLUMN])
        for k in lags:
            actual = row[lag_column_name(k)]
            exp = expected[key][k]
            if math.isnan(exp):
                assert pd.isna(actual), (
                    f"expected NaN for region={key[0]} period={key[1]} lag={k}, "
                    f"got {actual!r}"
                )
            else:
                # Parts 1 & 3: lag_k at t equals demand at t-k, within region.
                assert not pd.isna(actual)
                assert float(actual) == exp, (
                    f"region={key[0]} period={key[1]} lag={k}: "
                    f"expected {exp}, got {actual}"
                )
