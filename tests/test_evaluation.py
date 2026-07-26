"""Property-based test for holdout split correctness (Property 7).

Design reference:
- Correctness Properties -> Property 7 (Holdout split has no leakage)
- Requirements 6.1 (reserve the most recent contiguous portion of the demand
  series as the Holdout_Set and exclude it from all model training).

This module implements the *single* Hypothesis property test the design assigns
to Property 7, at 100+ iterations, per the Testing Strategy. It exercises
``src.evaluation.split_holdout``: given a long-format demand series spanning one
or more regions and a holdout size ``n``, the function must partition the rows
by *period* such that:

1. **Holdout equals the most-recent n periods** - every row whose ``period`` is
   among the ``n`` most recent distinct periods lands in ``holdout`` and no
   other row does.
2. **Train is the earlier remainder** - ``train`` holds exactly the rows in the
   earlier ``d - n`` distinct periods.
3. **Disjoint / no leakage** - ``train`` and ``holdout`` share no periods (and no
   rows); the max train period is strictly less than the min holdout period, so
   no future period ever leaks into training.
4. **Exact reconstruction** - ``pd.concat([train, holdout])`` reproduces the
   original series exactly (same rows, same relative order, same index).

The generator builds small multi-region contiguous daily series (1-4 regions,
short lengths) sharing overlapping calendar dates, optionally shuffles the input
row order to prove the split is period-based rather than row-based, and draws a
valid ``n`` in ``[1, d - 1]`` so training stays non-empty.
"""

from __future__ import annotations

import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st

from src.evaluation import split_holdout
from src.preparation import (
    DEMAND_COLUMN,
    PERIOD_COLUMN,
    REGION_COLUMN,
)

# The Geographic_Grain values (NYC boroughs). The generator draws a subset so
# region count varies from a single-region series up to 4.
BOROUGHS = ["Manhattan", "Brooklyn", "Queens", "Bronx"]

# A common base date. All regions share the same contiguous calendar window so
# their periods overlap - if the split ever leaked across regions it would show.
BASE_DATE = pd.Timestamp("2025-01-01")


@st.composite
def demand_series_and_holdout(draw: st.DrawFn):
    """Generate ``(series, n)`` for a multi-region long-format demand series.

    * A shared contiguous daily window of ``d`` distinct periods (2-25), so a
      valid ``n`` in ``[1, d - 1]`` always exists.
    * region count 1-4 (single region is an important edge case); every region
      spans the same window so all regions overlap on every date.
    * demand is a non-negative integer per (period, region).
    * ``n`` is drawn in ``[1, d - 1]`` so training is guaranteed non-empty.
    * the assembled rows are optionally shuffled to prove the function splits on
      the *period* axis rather than trusting input row order.
    """
    num_periods = draw(st.integers(min_value=2, max_value=25))
    periods = pd.date_range(start=BASE_DATE, periods=num_periods, freq="D")

    n_regions = draw(st.integers(min_value=1, max_value=len(BOROUGHS)))
    regions = BOROUGHS[:n_regions]

    frames = []
    for region in regions:
        demands = draw(
            st.lists(
                st.integers(min_value=0, max_value=5000),
                min_size=num_periods,
                max_size=num_periods,
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
        # split partitions by period rather than by input row position.
        order = draw(st.permutations(list(range(len(series)))))
        series = series.iloc[list(order)].reset_index(drop=True)

    holdout_periods = draw(st.integers(min_value=1, max_value=num_periods - 1))

    return series, holdout_periods


# Feature: ride-hailing-demand-forecasting, Property 7: Holdout split has no leakage
#
# Validates: Requirements 6.1
@settings(max_examples=200)
@given(data=demand_series_and_holdout())
def test_holdout_split_has_no_leakage(data) -> None:
    series, n = data
    original = series.copy()

    train, holdout = split_holdout(series, n)

    # Ground truth: the n most-recent distinct periods form the holdout window.
    distinct = pd.Series(pd.unique(original[PERIOD_COLUMN])).sort_values(kind="stable")
    expected_holdout_periods = set(distinct.iloc[-n:])
    expected_train_periods = set(distinct.iloc[:-n])

    # --- Part 1 & 2: each side holds exactly its period window's rows -------- #
    assert set(pd.unique(holdout[PERIOD_COLUMN])) == expected_holdout_periods
    assert set(pd.unique(train[PERIOD_COLUMN])) == expected_train_periods

    # Every original row whose period is in the holdout window must be in holdout
    # (and vice versa) - i.e. no rows dropped or misclassified within a period.
    expected_holdout_mask = original[PERIOD_COLUMN].isin(expected_holdout_periods)
    assert len(holdout) == int(expected_holdout_mask.sum())
    assert len(train) == int((~expected_holdout_mask).sum())

    # --- Part 3: disjoint, no leakage --------------------------------------- #
    assert expected_train_periods.isdisjoint(expected_holdout_periods)
    assert set(pd.unique(train[PERIOD_COLUMN])).isdisjoint(
        set(pd.unique(holdout[PERIOD_COLUMN]))
    )
    # No shared row indices between the two frames.
    assert set(train.index).isdisjoint(set(holdout.index))
    if len(train) > 0 and len(holdout) > 0:
        # The chronological boundary holds: max train period < min holdout period,
        # so no future period ever leaks into training.
        assert train[PERIOD_COLUMN].max() < holdout[PERIOD_COLUMN].min()

    # --- Part 4: exact reconstruction on concatenation ---------------------- #
    # Concatenating the two sides recovers exactly the caller's rows. Each side
    # preserves the original row index, so restoring index order reproduces the
    # input frame identically (same rows, values, and index) - no row is dropped,
    # duplicated, or altered by the split.
    reconstructed = pd.concat([train, holdout]).sort_index(kind="stable")
    pd.testing.assert_frame_equal(reconstructed, original.sort_index(kind="stable"))
