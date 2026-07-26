"""Property-based test for error-by-period partition (Property 10).

Design reference:
- Correctness Properties -> Property 10 (Error-by-period partition)
- Requirements 6.5 (report whether forecast error varies across distinct time
  periods rather than reporting only an aggregate error).

This module implements the *single* Hypothesis property test the design assigns
to Property 10, at 100+ iterations, per the Testing Strategy. It exercises
``src.evaluation.error_by_period``: given positionally-aligned ``actual``,
``forecast`` and ``buckets`` sequences of equal length, the returned table must
have one row per distinct bucket label (in first-appearance order), every
per-bucket ``mae``/``rmse``/``mape`` must be non-negative, and the ``n_periods``
counts must sum to the total number of holdout periods - the machine-checkable
statement that the buckets *partition* the holdout (every period lands in
exactly one bucket).

The generator constrains inputs to the valid space:

* a common length ``n >= 1`` (the function raises on empty input);
* finite float ``actual`` / ``forecast`` values (no NaN/inf) so the metric
  arithmetic and the non-negativity comparison are well defined;
* ``buckets`` labels drawn from a small pool of hashable labels (mixing strings
  and integers) so multiple periods share buckets and the partition/first-
  appearance behavior is exercised across realistic bucketings.
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from src.evaluation import error_by_period

# Finite floats only: NaN/inf would make the metric arithmetic and the
# non-negativity assertion ill-defined.
_FINITE = dict(allow_nan=False, allow_infinity=False)

# A small pool of hashable, mixed-type bucket labels so many periods collapse
# into the same bucket and the partition invariant is meaningfully exercised.
_BUCKET_LABELS = ["weekday", "weekend", "holiday", 0, 1, 2]


@st.composite
def aligned_actual_forecast_buckets(draw: st.DrawFn):
    """Generate ``(actual, forecast, buckets)`` of a common length ``n >= 1``.

    ``actual`` and ``forecast`` are finite floats; ``buckets`` are labels drawn
    from :data:`_BUCKET_LABELS`, positionally aligned with the demand values so
    each period carries exactly one bucket label.
    """
    n = draw(st.integers(min_value=1, max_value=60))
    value_strategy = st.floats(min_value=-1_000_000.0, max_value=1_000_000.0, **_FINITE)

    actual = draw(st.lists(value_strategy, min_size=n, max_size=n))
    forecast = draw(st.lists(value_strategy, min_size=n, max_size=n))
    buckets = draw(
        st.lists(st.sampled_from(_BUCKET_LABELS), min_size=n, max_size=n)
    )
    return actual, forecast, buckets


# Feature: ride-hailing-demand-forecasting, Property 10: Error-by-period
# partition - each per-bucket error is non-negative and the buckets partition
# the holdout periods (every period in exactly one bucket)
#
# Validates: Requirements 6.5
@settings(max_examples=200)
@given(data=aligned_actual_forecast_buckets())
def test_error_by_period_partition(data) -> None:
    actual, forecast, buckets = data

    table = error_by_period(actual, forecast, buckets)

    # --- One row per distinct bucket label, in first-appearance order ------ #
    expected_labels: list = []
    for label in buckets:
        if label not in expected_labels:
            expected_labels.append(label)
    assert list(table["bucket"]) == expected_labels
    assert len(table) == len(expected_labels)

    # --- Each per-bucket error metric is non-negative ---------------------- #
    assert (table["mae"] >= 0).all()
    assert (table["rmse"] >= 0).all()
    assert (table["mape"] >= 0).all()

    # --- Partition invariant: n_periods sums to the total holdout length --- #
    # Every period belongs to exactly one bucket, so the per-bucket counts must
    # account for all n periods with none dropped or double-counted.
    assert (table["n_periods"] > 0).all()
    assert int(table["n_periods"].sum()) == len(actual)
