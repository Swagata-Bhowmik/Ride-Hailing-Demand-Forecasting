"""Property-based test for upload input validation (Property 13).

Design reference:
- Correctness Properties -> Property 13 (Upload input validation)
- Error Handling -> "Non-conforming dashboard upload: ``validate_upload`` returns
  a descriptive error naming the missing/invalid column instead of crashing".
- Requirements 9.5.

This module implements the *single* Hypothesis property test the design assigns
to Property 13, at 100+ iterations, per the Testing Strategy. It exercises the
pure ``dashboard.upload_validation.validate_upload`` function (no Streamlit).

The property has three parts, matching the task's generation requirements:

(a) **Conforming datasets pass.** A fully conforming long-format ``DemandSeries``
    (``period`` datetime, ``region`` str, ``demand`` int >= 0) validates with
    ``ok is True`` and no offending column.

(b) **Missing required column fails, naming that column.** Dropping any one of
    the ``REQUIRED_COLUMNS`` yields ``ok is False`` and both the ``error`` message
    and ``offending_column`` reference the dropped column.

(c) **Corrupted dtype fails, naming the offending column.** Replacing one
    column's values with values of the wrong type (unparseable dates in
    ``period``, non-text in ``region``, non-integer/negative in ``demand``) yields
    ``ok is False`` and names that column.

The validator checks columns in the canonical order ``period -> region ->
demand`` and stops at the first failure, so a single injected fault always points
back at exactly the column that was corrupted.
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st

from dashboard.upload_validation import REQUIRED_COLUMNS, validate_upload
from src.preparation import DEMAND_COLUMN, PERIOD_COLUMN, REGION_COLUMN

# A small, fixed borough pool for the region column (Geographic_Grain values).
BOROUGHS = ["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island", "EWR"]

# Bounds for generated pickup timestamps (a modest span keeps generation fast).
_MIN_DT = datetime(2025, 5, 1, 0, 0, 0)
_MAX_DT = datetime(2026, 4, 30, 23, 59, 59)


@st.composite
def conforming_frames(draw: st.DrawFn) -> pd.DataFrame:
    """Generate a fully conforming long-format ``DemandSeries`` DataFrame.

    Every row has a real datetime ``period``, a borough-string ``region`` and a
    non-negative integer ``demand``. Row counts range from 1 to 40 (the empty
    frame is exercised separately in a unit-style example below).
    """
    n = draw(st.integers(min_value=1, max_value=40))
    periods = draw(
        st.lists(
            st.datetimes(min_value=_MIN_DT, max_value=_MAX_DT),
            min_size=n,
            max_size=n,
        )
    )
    regions = draw(
        st.lists(st.sampled_from(BOROUGHS), min_size=n, max_size=n)
    )
    demands = draw(
        st.lists(
            st.integers(min_value=0, max_value=10_000),
            min_size=n,
            max_size=n,
        )
    )
    return pd.DataFrame(
        {
            PERIOD_COLUMN: pd.to_datetime(periods),
            REGION_COLUMN: regions,
            DEMAND_COLUMN: demands,
        }
    )


# Feature: ride-hailing-demand-forecasting, Property 13: Upload input validation
# - a fully conforming dataset passes.
#
# Validates: Requirements 9.5
@settings(max_examples=150)
@given(df=conforming_frames())
def test_conforming_dataset_passes(df: pd.DataFrame) -> None:
    result = validate_upload(df)
    assert result.ok is True
    assert result.error is None
    assert result.offending_column is None


# Feature: ride-hailing-demand-forecasting, Property 13: Upload input validation
# - a missing required column yields an error referencing the offending column.
#
# Validates: Requirements 9.5
@settings(max_examples=150)
@given(df=conforming_frames(), drop_index=st.integers(min_value=0, max_value=len(REQUIRED_COLUMNS) - 1))
def test_missing_required_column_names_it(df: pd.DataFrame, drop_index: int) -> None:
    dropped = REQUIRED_COLUMNS[drop_index]
    corrupted = df.drop(columns=[dropped])

    result = validate_upload(corrupted)

    assert result.ok is False
    assert result.offending_column == dropped
    assert result.error is not None and dropped in result.error


# Feature: ride-hailing-demand-forecasting, Property 13: Upload input validation
# - a corrupted dtype in one column yields an error naming that column.
#
# Validates: Requirements 9.5
@settings(max_examples=150)
@given(
    df=conforming_frames(),
    target=st.sampled_from(REQUIRED_COLUMNS),
    demand_fault=st.sampled_from(["non_numeric", "fractional", "negative"]),
)
def test_corrupted_dtype_names_offending_column(
    df: pd.DataFrame, target: str, demand_fault: str
) -> None:
    corrupted = df.copy()

    if target == PERIOD_COLUMN:
        # Unparseable date strings -> a corrupted period dtype.
        corrupted[PERIOD_COLUMN] = corrupted[PERIOD_COLUMN].astype(object)
        corrupted.loc[corrupted.index[0], PERIOD_COLUMN] = "not-a-date"
    elif target == REGION_COLUMN:
        # Non-text (numeric) entry -> a corrupted region dtype.
        corrupted[REGION_COLUMN] = corrupted[REGION_COLUMN].astype(object)
        corrupted.loc[corrupted.index[0], REGION_COLUMN] = 12345
    else:  # DEMAND_COLUMN
        corrupted[DEMAND_COLUMN] = corrupted[DEMAND_COLUMN].astype(object)
        if demand_fault == "non_numeric":
            corrupted.loc[corrupted.index[0], DEMAND_COLUMN] = "many"
        elif demand_fault == "fractional":
            corrupted.loc[corrupted.index[0], DEMAND_COLUMN] = 3.5
        else:  # negative
            corrupted.loc[corrupted.index[0], DEMAND_COLUMN] = -1

    result = validate_upload(corrupted)

    assert result.ok is False
    assert result.offending_column == target
    assert result.error is not None and target in result.error


def test_empty_conforming_frame_passes() -> None:
    """An empty frame that still carries the three required columns conforms."""
    empty = pd.DataFrame(
        {
            PERIOD_COLUMN: pd.Series([], dtype="datetime64[ns]"),
            REGION_COLUMN: pd.Series([], dtype="object"),
            DEMAND_COLUMN: pd.Series([], dtype="int64"),
        }
    )
    result = validate_upload(empty)
    assert result.ok is True
    assert result.offending_column is None
