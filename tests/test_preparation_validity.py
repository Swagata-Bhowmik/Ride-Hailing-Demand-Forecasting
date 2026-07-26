"""Property-based test for invalid-record handling (Property 5).

Design reference:
- Correctness Properties -> Property 5 (Invalid-record handling)
- Requirements 4.4 (records failing a Requirement 1 validity check get a
  documented handling rule applied to them).

This module implements the *single* Hypothesis property test the design assigns
to Property 5, at 100 iterations, per the Testing Strategy. It exercises
``src.preparation.apply_validity_rules`` under its default ``"drop"`` rule.

Construction of each example:

* A **clean base** of fully in-domain records is generated: every pickup falls
  inside a fixed, known stated month (April 2026) and every non-negative measure
  column holds a value ``>= 0``. These records must never be flagged.
* A **known number** of invalid records is then injected. To keep the handled
  count deterministic, each injected record violates *exactly one* check - either
  its pickup is outside the stated month, or exactly one measure column is
  negative (never both, and only one column at a time). Consequently every
  injected record is a single distinct violation, so ``total_invalid_handled``
  must equal the number injected.

``stated_month`` and ``non_negative_columns`` are passed explicitly to
``apply_validity_rules`` (and to the re-validation call) so the checks are fully
deterministic and independent of month inference.

The property has three parts:

1. **No invalid record remains** - re-running ``flag_domain_violations`` on the
   cleaned DataFrame with the same ``stated_month``/``non_negative_columns``
   returns no violations at all.
2. **Handled count is exact** - ``HandlingLog.total_invalid_handled`` equals the
   number of injected invalid records.
3. **Row accounting** - ``output_row_count == input_row_count -
   total_invalid_handled`` (dropping removes exactly the handled records).
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st

from src.preparation import apply_validity_rules
from src.validation import flag_domain_violations

# --- Fixed, known validity configuration used across every example ---------- #

#: The file's stated month. Clean pickups fall inside it; out-of-month invalid
#: records fall outside it. Passed explicitly so no inference is involved.
STATED_MONTH = (2026, 4)

#: The measure columns checked for negative values. Passed explicitly so the
#: negative-value violations are deterministic and schema-independent.
NON_NEGATIVE_COLUMNS = ["trip_miles", "base_passenger_fare", "driver_pay"]

PICKUP_COLUMN = "pickup_datetime"

# Column order of the generated frame: pickup timestamp + the measure columns.
_COLUMNS = [PICKUP_COLUMN, *NON_NEGATIVE_COLUMNS]

# --- Value strategies ------------------------------------------------------- #

# A pickup timestamp inside the stated month (April 2026).
_in_month_dt = st.datetimes(
    min_value=datetime(2026, 4, 1, 0, 0, 0),
    max_value=datetime(2026, 4, 30, 23, 59, 59),
)

# A pickup timestamp outside the stated month (any other month in 2025-2027).
_out_of_month_dt = st.datetimes(
    min_value=datetime(2025, 1, 1, 0, 0, 0),
    max_value=datetime(2027, 12, 31, 23, 59, 59),
).filter(lambda d: (d.year, d.month) != STATED_MONTH)

# A non-negative measure value (>= 0), the valid domain for the measure columns.
_nonneg = st.floats(min_value=0.0, max_value=1_000_000.0, allow_nan=False, allow_infinity=False)

# A strictly negative measure value - the out-of-domain case for a measure column.
_neg = st.floats(min_value=-1_000_000.0, max_value=-0.001, allow_nan=False, allow_infinity=False)


@st.composite
def datasets_with_injected_invalids(draw: st.DrawFn):
    """Generate ``(df, n_invalid)`` with a clean base plus ``n_invalid`` injected rows.

    Every clean record is fully in-domain; every injected record violates exactly
    one check (out-of-month pickup, or a single negative measure column). The
    second element is the *known* number of injected invalid records, which the
    property asserts equals ``HandlingLog.total_invalid_handled``.
    """
    n_clean = draw(st.integers(min_value=0, max_value=20))
    n_invalid = draw(st.integers(min_value=0, max_value=20))

    rows: list[dict] = []

    # Clean, in-domain base: in-month pickup, all measures non-negative.
    for _ in range(n_clean):
        rows.append(
            {
                PICKUP_COLUMN: draw(_in_month_dt),
                "trip_miles": draw(_nonneg),
                "base_passenger_fare": draw(_nonneg),
                "driver_pay": draw(_nonneg),
            }
        )

    # Injected invalid records: each violates EXACTLY ONE check.
    violation_kinds = ["out_of_month", *(f"negative_{c}" for c in NON_NEGATIVE_COLUMNS)]
    for _ in range(n_invalid):
        kind = draw(st.sampled_from(violation_kinds))
        # Start from a fully valid record, then break exactly one thing.
        row = {
            PICKUP_COLUMN: draw(_in_month_dt),
            "trip_miles": draw(_nonneg),
            "base_passenger_fare": draw(_nonneg),
            "driver_pay": draw(_nonneg),
        }
        if kind == "out_of_month":
            row[PICKUP_COLUMN] = draw(_out_of_month_dt)
        else:
            col = kind[len("negative_") :]
            row[col] = draw(_neg)
        rows.append(row)

    df = pd.DataFrame(rows, columns=_COLUMNS)
    return df, n_invalid


# Feature: ride-hailing-demand-forecasting, Property 5: Invalid-record handling -
# after handling no injected invalid record remains and the HandlingLog count
# equals the number injected
#
# Validates: Requirements 4.4
@settings(max_examples=200)
@given(data=datasets_with_injected_invalids())
def test_invalid_record_handling(data) -> None:
    df, n_invalid = data
    input_row_count = len(df)

    cleaned_df, log = apply_validity_rules(
        df,
        rule="drop",
        ts_col=PICKUP_COLUMN,
        stated_month=STATED_MONTH,
        non_negative_columns=NON_NEGATIVE_COLUMNS,
    )

    # --- Part 1: no injected invalid record remains ------------------------ #
    remaining_violations = flag_domain_violations(
        cleaned_df,
        ts_col=PICKUP_COLUMN,
        stated_month=STATED_MONTH,
        non_negative_columns=NON_NEGATIVE_COLUMNS,
    )
    assert remaining_violations == []

    # --- Part 2: handled count equals the number injected ------------------ #
    assert log.total_invalid_handled == n_invalid

    # --- Part 3: row accounting -------------------------------------------- #
    assert log.input_row_count == input_row_count
    assert log.output_row_count == input_row_count - log.total_invalid_handled
    assert len(cleaned_df) == log.output_row_count
