"""Property-based test for Data_Validator profiling accuracy (Property 1).

This module implements design **Property 1: Profiling accuracy** for the
ride-hailing-demand-forecasting feature. It uses the Hypothesis library to
generate a wide variety of DataFrames — varied columns, dtypes, null values,
and a ``pickup_datetime`` timestamp column with varied timestamps — and asserts
that every statistic the profilers report equals ground truth computed directly
from the generated DataFrame:

- ``profile_schema``  -> row count, column names (in order), and per-column dtype
- ``profile_nulls``   -> per-column null count and percentage
- ``pickup_date_range`` -> min/max pickup timestamp over non-null values

Design references:
- Correctness Properties -> Property 1: Profiling accuracy
- Testing Strategy -> Property-based tests (Hypothesis, >= 100 iterations)
- Validates: Requirements 1.2, 1.3, 1.4
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from src.validation import (
    DEFAULT_PICKUP_COLUMN,
    pickup_date_range,
    profile_nulls,
    profile_schema,
)

# --------------------------------------------------------------------------- #
# Generators
# --------------------------------------------------------------------------- #

# A pool of realistic-but-arbitrary extra column names to draw from. The pickup
# timestamp column is added separately and is intentionally excluded here so the
# generated column set never collides with it.
_EXTRA_COLUMN_POOL: tuple[str, ...] = (
    "trip_miles",
    "base_passenger_fare",
    "PULocationID",
    "hvfhs_license_num",
    "driver_pay",
    "tips",
    "is_shared",
)

# Plausible timestamp span for pickup values (kept away from pandas' extremes to
# avoid overflow while still exercising a wide, varied range).
_timestamps = st.datetimes(
    min_value=datetime(2019, 1, 1),
    max_value=datetime(2027, 12, 31),
)


def _nullable(base: st.SearchStrategy) -> st.SearchStrategy:
    """Wrap a value strategy so it also yields ``None`` (an injected null)."""
    return st.one_of(st.none(), base)


# Value strategies for extra columns, chosen to induce a variety of real pandas
# dtypes (int64, float64, object/string, bool) once assembled into a DataFrame,
# including the null-driven dtype coercions pandas performs.
_COLUMN_VALUE_STRATEGIES: tuple[st.SearchStrategy, ...] = (
    st.integers(min_value=-1000, max_value=1000),                         # -> int64
    _nullable(st.integers(min_value=-1000, max_value=1000)),              # -> float64 (NaN)
    st.floats(allow_nan=True, allow_infinity=False, width=32),            # -> float64
    _nullable(st.floats(allow_nan=False, allow_infinity=False, width=32)),
    _nullable(st.text(max_size=8)),                                       # -> object
    st.booleans(),                                                        # -> bool
)


@st.composite
def _profiling_dataframes(draw: st.DrawFn) -> pd.DataFrame:
    """Generate a DataFrame with a ``pickup_datetime`` column plus varied extras.

    Guarantees at least one row and at least one non-null pickup timestamp so
    that ``pickup_date_range`` always has a defined range to compute (its
    empty/all-null case raises ``ValueError`` and is exercised by unit tests
    elsewhere, not by this property).
    """
    n_rows = draw(st.integers(min_value=1, max_value=12))

    # Pickup timestamps: each may be null, but force at least one non-null value.
    ts_values = draw(
        st.lists(_nullable(_timestamps), min_size=n_rows, max_size=n_rows)
    )
    if all(v is None for v in ts_values):
        ts_values[draw(st.integers(min_value=0, max_value=n_rows - 1))] = draw(_timestamps)

    data: dict[str, object] = {
        DEFAULT_PICKUP_COLUMN: pd.Series(pd.to_datetime(ts_values)),
    }

    # A subset of extra columns, each with an independently chosen value strategy.
    extra_names = draw(
        st.lists(
            st.sampled_from(_EXTRA_COLUMN_POOL),
            unique=True,
            max_size=len(_EXTRA_COLUMN_POOL),
        )
    )
    for name in extra_names:
        value_strategy = draw(st.sampled_from(_COLUMN_VALUE_STRATEGIES))
        column_values = draw(
            st.lists(value_strategy, min_size=n_rows, max_size=n_rows)
        )
        data[name] = column_values

    return pd.DataFrame(data)


# --------------------------------------------------------------------------- #
# Property 1: Profiling accuracy
# --------------------------------------------------------------------------- #

# Feature: ride-hailing-demand-forecasting, Property 1: Profiling accuracy — every reported statistic equals ground truth computed directly from the DataFrame
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
@given(df=_profiling_dataframes())
def test_profiling_accuracy(df: pd.DataFrame) -> None:
    """Every reported profiling statistic equals ground truth from the DataFrame.

    Validates: Requirements 1.2, 1.3, 1.4
    """
    row_count = len(df)

    # --- Requirement 1.2: schema (row count, columns, dtypes) ---------------
    schema = profile_schema(df)
    assert schema.row_count == row_count
    assert schema.column_names == list(df.columns)
    assert schema.dtypes == {str(col): str(dtype) for col, dtype in df.dtypes.items()}

    # --- Requirement 1.3: per-column null count and percentage --------------
    nulls = profile_nulls(df)
    assert set(nulls.keys()) == {str(col) for col in df.columns}
    for col in df.columns:
        expected_count = int(df[col].isna().sum())
        expected_pct = (expected_count / row_count * 100.0) if row_count else 0.0
        stat = nulls[str(col)]
        assert stat.count == expected_count
        assert stat.percentage == pytest.approx(expected_pct)
        assert 0.0 <= stat.percentage <= 100.0

    # --- Requirement 1.4: min/max pickup timestamp over non-null values -----
    non_null_ts = df[DEFAULT_PICKUP_COLUMN].dropna()
    min_ts, max_ts = pickup_date_range(df)
    assert min_ts == non_null_ts.min()
    assert max_ts == non_null_ts.max()
    assert min_ts <= max_ts
