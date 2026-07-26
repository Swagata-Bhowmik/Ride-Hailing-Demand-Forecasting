"""Property-based test for duplicate and domain-violation flagging (Property 2).

This module implements design **Property 2: Duplicate and domain-violation
flagging** for the ride-hailing-demand-forecasting feature. It uses the
Hypothesis library to build DataFrames from a *de-duplicated, in-domain* base of
clean records and then injects a known number of exact-duplicate rows and a
known number of out-of-domain records (pickups outside the stated month, or a
negative value in a non-negative column). It then asserts that:

- ``count_duplicates`` reports exactly the number of injected duplicate rows, and
- ``flag_domain_violations`` reports a total flagged count equal to the number of
  injected out-of-domain records, with every flagged example being a genuinely
  violating record.

The stated month is passed explicitly to ``flag_domain_violations`` so the
month check is deterministic and does not depend on modal inference (important
when many out-of-month records are injected).

Design references:
- Correctness Properties -> Property 2: Duplicate and domain-violation flagging
- Testing Strategy -> Property-based tests (Hypothesis, >= 100 iterations)
- Validates: Requirements 1.5, 1.6
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from src.validation import count_duplicates, flag_domain_violations

# --------------------------------------------------------------------------- #
# Fixed scenario parameters
# --------------------------------------------------------------------------- #

#: The file's stated month, passed explicitly to ``flag_domain_violations`` so the
#: out-of-month check is deterministic regardless of how many out-of-month
#: records are injected.
STATED_YEAR = 2026
STATED_MONTH = 4
STATED_PERIOD = pd.Period(year=STATED_YEAR, month=STATED_MONTH, freq="M")

#: Non-negative measure columns checked for negative values (all real FHVHV
#: columns). Passed explicitly so the check is deterministic.
NON_NEG_COLS: tuple[str, ...] = ("base_passenger_fare", "driver_pay", "trip_miles")

# --------------------------------------------------------------------------- #
# Value generators
# --------------------------------------------------------------------------- #

# A clean pickup timestamp inside the stated month (April 2026).
_clean_ts = st.datetimes(
    min_value=datetime(STATED_YEAR, STATED_MONTH, 1),
    max_value=datetime(STATED_YEAR, STATED_MONTH, 30, 23, 59, 59),
)

# A pickup timestamp genuinely outside the stated month, drawn from the span
# strictly before April 2026 or strictly after it.
_outside_ts = st.one_of(
    st.datetimes(
        min_value=datetime(2024, 1, 1),
        max_value=datetime(STATED_YEAR, STATED_MONTH - 1, 31, 23, 59, 59),
    ),
    st.datetimes(
        min_value=datetime(STATED_YEAR, STATED_MONTH + 1, 1),
        max_value=datetime(2027, 12, 31, 23, 59, 59),
    ),
)

# Non-negative and strictly-negative measure values.
_nonneg = st.floats(min_value=0.0, max_value=10_000.0, allow_nan=False, allow_infinity=False)
_neg = st.floats(min_value=-10_000.0, max_value=-0.01, allow_nan=False, allow_infinity=False)


def _clean_row(draw: st.DrawFn, row_id: int) -> dict:
    """A clean, in-domain record: in-month pickup and all measures non-negative."""
    return {
        "row_id": row_id,
        "pickup_datetime": draw(_clean_ts),
        "base_passenger_fare": draw(_nonneg),
        "driver_pay": draw(_nonneg),
        "trip_miles": draw(_nonneg),
    }


@st.composite
def _flagging_scenarios(draw: st.DrawFn) -> tuple[pd.DataFrame, int, int]:
    """Build a DataFrame with known injected duplicate and violation counts.

    The base is de-duplicated by construction (every base row carries a unique
    ``row_id``), so the only duplicates in the final frame are the ones injected,
    and the only domain violations are the out-of-domain records injected — each
    of which violates exactly one rule, so the total flagged count equals the
    number injected.

    Returns:
        ``(df, n_dup, n_viol)`` — the assembled frame and the exact number of
        injected duplicate rows and out-of-domain records.
    """
    # --- de-duplicated, in-domain base --------------------------------------
    n_base = draw(st.integers(min_value=1, max_value=8))
    base_rows = [_clean_row(draw, i) for i in range(n_base)]

    # --- inject exact duplicates of clean base rows -------------------------
    # Each duplicate is an exact copy (row_id included) of an existing clean
    # base row, so it is a genuine duplicate and never a domain violation.
    n_dup = draw(st.integers(min_value=0, max_value=5))
    dup_rows = [dict(base_rows[draw(st.integers(0, n_base - 1))]) for _ in range(n_dup)]

    # --- inject out-of-domain records, each violating exactly one rule ------
    # Unique row_ids (>= n_base) keep these from ever being duplicates.
    n_viol = draw(st.integers(min_value=0, max_value=6))
    viol_rows = []
    for j in range(n_viol):
        row = _clean_row(draw, n_base + j)
        if draw(st.sampled_from(["outside_month", "negative"])) == "outside_month":
            row["pickup_datetime"] = draw(_outside_ts)  # only the month is wrong
        else:
            neg_col = draw(st.sampled_from(list(NON_NEG_COLS)))
            row[neg_col] = draw(_neg)  # exactly one measure is negative
        viol_rows.append(row)

    df = pd.DataFrame(base_rows + dup_rows + viol_rows)
    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])
    return df, n_dup, n_viol


# --------------------------------------------------------------------------- #
# Property 2: Duplicate and domain-violation flagging
# --------------------------------------------------------------------------- #

# Feature: ride-hailing-demand-forecasting, Property 2: Duplicate and domain-violation flagging — reported duplicate and flagged-violation counts equal injected counts and each flagged item is genuinely violating
@settings(max_examples=150, suppress_health_check=[HealthCheck.too_slow])
@given(scenario=_flagging_scenarios())
def test_duplicate_and_domain_violation_flagging(
    scenario: tuple[pd.DataFrame, int, int],
) -> None:
    """Reported counts equal injected counts and every flagged item truly violates.

    Validates: Requirements 1.5, 1.6
    """
    df, n_dup, n_viol = scenario

    # --- Requirement 1.5: duplicate count equals injected duplicates --------
    assert count_duplicates(df) == n_dup

    # --- Requirement 1.6: domain violations equal injected out-of-domain ----
    violations = flag_domain_violations(
        df,
        ts_col="pickup_datetime",
        stated_month=(STATED_YEAR, STATED_MONTH),
        non_negative_columns=NON_NEG_COLS,
    )

    # Total flagged records across all violation types equals the injected count.
    assert sum(v.count for v in violations) == n_viol

    # Every reported violation is a known type with a positive count and a
    # genuinely-violating example record.
    for violation in violations:
        assert violation.count >= 1
        assert violation.example is not None

        if violation.violation_type == "pickup_outside_stated_month":
            example_ts = pd.Timestamp(violation.example["pickup_datetime"])
            assert example_ts.to_period("M") != STATED_PERIOD
        elif violation.violation_type.startswith("negative_"):
            col = violation.violation_type[len("negative_"):]
            assert col in NON_NEG_COLS
            assert float(violation.example[col]) < 0
        else:  # pragma: no cover - guards against unexpected violation types
            raise AssertionError(
                f"Unexpected violation type reported: {violation.violation_type!r}"
            )
