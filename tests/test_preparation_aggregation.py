"""Property-based test for aggregation correctness and reconciliation (Property 3).

Design reference:
- Correctness Properties -> Property 3 (Aggregation correctness and reconciliation)
- Requirements 4.1 (aggregate raw records to Time_Grain/Geographic_Grain),
  4.5 / 13.4 (reconcile aggregated totals with raw valid record counts).

This module implements the *single* Hypothesis property test the design assigns
to Property 3, at 100+ iterations, per the Testing Strategy. It exercises
``src.preparation.aggregate_demand`` (the aggregation step) together with
``src.validation.revalidate_prepared`` (the reconciliation step).

The property has two parts:

1. **Per-bucket correctness** - for any set of raw trip records, the demand
   series produced by :func:`aggregate_demand` has, for every ``(period, region)``
   bucket, a ``demand`` equal to a direct group-by count of the records (computed
   here independently with a plain :class:`collections.Counter`, so the reference
   does not just re-run pandas' groupby).

2. **Conservation / reconciliation** - the total demand summed across all buckets
   equals the number of *valid* raw records (rows whose pickup timestamp parses
   to a non-null period AND whose region is non-null). This is confirmed via
   :func:`revalidate_prepared`, asserting ``reconciled is True`` and
   ``difference == 0``.

Edge cases are driven by the generators: empty frames, a single region, all
records on the same day, and injected null timestamps / null regions (which must
be excluded from both the bucket counts and the valid raw count).
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime

import pandas as pd
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from src.config import default_scope
from src.preparation import (
    DEMAND_COLUMN,
    PERIOD_COLUMN,
    REGION_COLUMN,
    aggregate_demand,
)
from src.validation import revalidate_prepared

# The Geographic_Grain values (NYC boroughs + EWR), used as the region pool. A
# small, fixed pool makes (period, region) collisions frequent so buckets
# accumulate real counts rather than every record landing in its own bucket.
BOROUGHS = ["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island", "EWR"]

# Bounds for generated pickup timestamps. A modest ~3-month span keeps many
# records falling on the same calendar day (exercising real aggregation) while
# still varying across days and months.
_MIN_DT = datetime(2025, 5, 1, 0, 0, 0)
_MAX_DT = datetime(2025, 7, 31, 23, 59, 59)


@st.composite
def raw_trip_records(draw: st.DrawFn) -> list[tuple[object, object]]:
    """Generate a list of ``(pickup_datetime, region)`` raw trip records.

    The strategy deliberately covers the Property 3 edge cases:

    * **number of rows** varies from 0 (empty-ish frame) up to 80;
    * **number of regions** varies from 1 (single-region series) up to all six
      boroughs;
    * **timestamps** are drawn from a shared ~3-month window so many records
      share a calendar day (including the "all same day" case when the window
      collapses), with ``None`` injected to represent null timestamps;
    * **regions** are drawn from the chosen borough pool with ``None`` injected to
      represent null regions.

    Rows with a null timestamp or a null region are *invalid* for aggregation and
    must be excluded from both the per-bucket counts and the valid raw count.
    """
    n_regions = draw(st.integers(min_value=1, max_value=len(BOROUGHS)))
    region_pool = BOROUGHS[:n_regions]

    # ~1-in-6 chance of a null region; otherwise a borough from the pool.
    region_strategy = st.one_of(
        st.none(),
        st.sampled_from(region_pool),
        st.sampled_from(region_pool),
        st.sampled_from(region_pool),
        st.sampled_from(region_pool),
        st.sampled_from(region_pool),
    )

    # ~1-in-6 chance of a null timestamp; otherwise a datetime in the window.
    datetime_strategy = st.one_of(
        st.none(),
        st.datetimes(min_value=_MIN_DT, max_value=_MAX_DT),
        st.datetimes(min_value=_MIN_DT, max_value=_MAX_DT),
        st.datetimes(min_value=_MIN_DT, max_value=_MAX_DT),
        st.datetimes(min_value=_MIN_DT, max_value=_MAX_DT),
        st.datetimes(min_value=_MIN_DT, max_value=_MAX_DT),
    )

    return draw(
        st.lists(
            st.tuples(datetime_strategy, region_strategy),
            min_size=0,
            max_size=80,
        )
    )


def _records_to_frame(records: list[tuple[object, object]]) -> pd.DataFrame:
    """Build a raw-records DataFrame with the columns ``aggregate_demand`` consumes.

    Provides ``region`` directly (zone mapping is skipped intentionally, since
    ``aggregate_demand`` consumes an already-present region column) alongside the
    ``pickup_datetime`` column it truncates to the daily grain.
    """
    if not records:
        return pd.DataFrame(
            {
                "pickup_datetime": pd.Series([], dtype="object"),
                "region": pd.Series([], dtype="object"),
            }
        )
    pickup, region = zip(*records)
    return pd.DataFrame({"pickup_datetime": list(pickup), "region": list(region)})


def _expected_bucket_counts(
    records: list[tuple[object, object]]
) -> tuple[dict[tuple[pd.Timestamp, object], int], int]:
    """Independently compute the ground-truth bucket counts and valid-row count.

    Mirrors ``aggregate_demand``'s contract without reusing pandas' groupby: each
    record is valid only if its timestamp parses to a non-null day-floored period
    AND its region is non-null; valid records increment their ``(period, region)``
    bucket by one. Returns ``(counts, valid_count)`` where ``valid_count`` is the
    total number of valid records (== sum of all bucket counts).
    """
    counts: Counter = Counter()
    valid_count = 0
    for ts, region in records:
        if ts is None or region is None:
            continue
        # Floor the timestamp to the daily grain, matching aggregate_demand.
        period = pd.Timestamp(ts).floor("D")
        if pd.isna(period):
            continue
        counts[(period, region)] += 1
        valid_count += 1
    return dict(counts), valid_count


# Feature: ride-hailing-demand-forecasting, Property 3: Aggregation correctness
# and reconciliation - per-bucket counts equal a direct group-by and total
# demand equals the number of valid raw records
#
# Validates: Requirements 4.1, 4.5, 13.4
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
@given(records=raw_trip_records())
def test_aggregation_correctness_and_reconciliation(
    records: list[tuple[object, object]]
) -> None:
    scope = default_scope()
    df = _records_to_frame(records)

    series = aggregate_demand(df, scope)

    # Independent ground truth: direct per-bucket counts and the valid-row count.
    expected_counts, expected_valid = _expected_bucket_counts(records)

    # --- Part 1: per-bucket counts equal the direct group-by --------------- #
    # The series must carry exactly the long-format DemandSeries columns.
    assert list(series.columns) == [PERIOD_COLUMN, REGION_COLUMN, DEMAND_COLUMN]

    # One row per observed (period, region) bucket - no duplicate buckets.
    bucket_keys = list(zip(series[PERIOD_COLUMN], series[REGION_COLUMN]))
    assert len(bucket_keys) == len(set(bucket_keys))

    actual_counts = {
        (period, region): int(demand)
        for period, region, demand in zip(
            series[PERIOD_COLUMN], series[REGION_COLUMN], series[DEMAND_COLUMN]
        )
    }
    assert actual_counts == expected_counts

    # Demand is always a non-negative integer count.
    assert (series[DEMAND_COLUMN] >= 0).all()

    # --- Part 2: conservation / reconciliation ----------------------------- #
    total_demand = int(series[DEMAND_COLUMN].sum())
    assert total_demand == expected_valid

    report = revalidate_prepared(series, expected_valid)
    assert report.total_demand == expected_valid
    assert report.raw_valid_count == expected_valid
    assert report.difference == 0
    assert report.reconciled is True
