"""Upload input validation for the dashboard's upload-and-analyze mode
(Requirements 9.4, 9.5).

This module is deliberately **pure** - it imports only ``pandas`` (and the
canonical column-name constants from :mod:`src.preparation`) and has **no
Streamlit dependency**. That separation is intentional:

* The Streamlit app (:mod:`dashboard.app`) imports :func:`validate_upload` here to
  power its "Upload & analyze" section, so a user-supplied file is checked before
  any analysis runs.
* The property test for design **Property 13** (task 13.3) can ``import`` this
  module and exercise :func:`validate_upload` directly **without a running
  Streamlit server**, because nothing here touches ``streamlit``.

Expected input format (the long-format ``DemandSeries`` the pipeline emits):

============  =========  ===================================================
column        dtype      meaning
============  =========  ===================================================
``period``    datetime   observation timestamp at the Time_Grain
``region``    str        borough (Geographic_Grain)
``demand``    int >= 0   trip count for that ``(period, region)`` bucket
============  =========  ===================================================

The column names come from :mod:`src.preparation` (``PERIOD_COLUMN``,
``REGION_COLUMN``, ``DEMAND_COLUMN``) so this validator and the preparation
pipeline never disagree on the schema.

Design references:
- Components and Interfaces -> Dashboard (`dashboard/app.py`) upload mode
- Error Handling -> "Non-conforming dashboard upload: ``validate_upload`` returns
  a descriptive error naming the missing/invalid column instead of crashing"
- Correctness Properties -> Property 13 (Upload input validation)
- Requirements 9.4, 9.5
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from src.preparation import DEMAND_COLUMN, PERIOD_COLUMN, REGION_COLUMN

#: The required columns of a conforming upload, in a fixed canonical order so
#: validation is deterministic: the first offending column (in this order) is the
#: one named in the returned error.
REQUIRED_COLUMNS: tuple[str, ...] = (PERIOD_COLUMN, REGION_COLUMN, DEMAND_COLUMN)


@dataclass(frozen=True)
class UploadValidationResult:
    """Outcome of validating an uploaded dataset (Requirement 9.5).

    :func:`validate_upload` returns one of these instead of raising, so the
    dashboard can surface a friendly message rather than crash on bad input.

    Attributes:
        ok: ``True`` when the upload conforms to the expected format.
        error: A descriptive, human-readable error message when ``ok`` is
            ``False``; ``None`` on success. When set, the message always names
            the offending column.
        offending_column: The name of the column that caused the failure
            (``None`` on success). Machine-readable counterpart to ``error`` so
            callers/tests can assert on the exact column.
    """

    ok: bool
    error: Optional[str] = None
    offending_column: Optional[str] = None

    def __bool__(self) -> bool:  # pragma: no cover - trivial convenience
        """Allow ``if validate_upload(df): ...`` truthiness checks."""
        return self.ok


def _passed() -> UploadValidationResult:
    """Return the canonical success result."""
    return UploadValidationResult(ok=True)


def _failed(column: str, reason: str) -> UploadValidationResult:
    """Return a failure result whose message names the offending ``column``."""
    return UploadValidationResult(
        ok=False,
        error=f"Column '{column}' {reason}",
        offending_column=column,
    )


def validate_upload(df: pd.DataFrame) -> UploadValidationResult:
    """Validate a user-uploaded dataset against the expected input format (R9.5).

    Checks, in order, that the uploaded frame is a proper long-format
    ``DemandSeries``:

    1. **Required columns present.** ``period``, ``region`` and ``demand`` must
       all exist. The first missing one (in canonical order) is named.
    2. **``period`` is datetime-like.** Values must already be a datetime dtype or
       be fully parseable to datetimes; a value that cannot be parsed (a corrupted
       dtype) fails and names ``period``.
    3. **``region`` is string-like.** Every non-null value must be a ``str``; a
       numeric/other dtype fails and names ``region``.
    4. **``demand`` is a non-negative integer.** Values must be numeric, whole
       numbers, and ``>= 0``; non-numeric, fractional, or negative values fail and
       name ``demand``.

    Because the checks run in a fixed order and stop at the first failure, when a
    single column is missing or has a corrupted dtype the returned error always
    references *that* column - the behaviour design Property 13 asserts. A fully
    conforming dataset (including an empty frame that still carries the three
    columns) passes.

    This function **never raises** on bad data: it returns an
    :class:`UploadValidationResult` describing the problem so the dashboard can
    display a descriptive message instead of crashing (design Error Handling).

    Args:
        df: The parsed uploaded dataset (e.g. from ``pd.read_csv`` /
            ``pd.read_parquet`` of the user's file).

    Returns:
        An :class:`UploadValidationResult`: ``ok=True`` when the dataset conforms,
        otherwise ``ok=False`` with a descriptive ``error`` naming the
        ``offending_column``.
    """
    # Guard: a non-DataFrame input is itself a conformance failure. Name the first
    # required column so the message still points at the expected schema.
    if not isinstance(df, pd.DataFrame):
        return UploadValidationResult(
            ok=False,
            error=(
                "Uploaded data could not be read as a table with the required "
                f"columns {list(REQUIRED_COLUMNS)}."
            ),
            offending_column=REQUIRED_COLUMNS[0],
        )

    # --- 1. Required columns present -------------------------------------------------
    for column in REQUIRED_COLUMNS:
        if column not in df.columns:
            return _failed(
                column,
                "is required but missing from the uploaded data. Expected long-format "
                f"columns {list(REQUIRED_COLUMNS)} "
                f"(period: datetime, region: text, demand: integer >= 0).",
            )

    # An empty frame that carries the schema is trivially conforming: there are no
    # values to violate the dtype expectations.
    if len(df) == 0:
        return _passed()

    # --- 2. period must be datetime-like ---------------------------------------------
    period = df[PERIOD_COLUMN]
    if not pd.api.types.is_datetime64_any_dtype(period):
        parsed = pd.to_datetime(period, errors="coerce")
        # A value that was not originally null but becomes NaT failed to parse -> the
        # column is not datetime-like (a corrupted dtype).
        newly_invalid = parsed.isna() & ~pd.Series(period).isna().to_numpy()
        if bool(newly_invalid.any()):
            return _failed(
                PERIOD_COLUMN,
                "must contain datetime values but has entries that cannot be parsed "
                "as dates.",
            )

    # --- 3. region must be string-like -----------------------------------------------
    region = df[REGION_COLUMN]
    region_non_null = region.dropna()
    if len(region_non_null) > 0 and not region_non_null.map(lambda v: isinstance(v, str)).all():
        return _failed(
            REGION_COLUMN,
            "must contain text (borough) values but has non-text entries.",
        )

    # --- 4. demand must be a non-negative integer ------------------------------------
    demand = df[DEMAND_COLUMN]
    numeric = pd.to_numeric(demand, errors="coerce")
    # Non-numeric entries (that were not already null) -> corrupted dtype.
    newly_non_numeric = numeric.isna() & ~pd.Series(demand).isna().to_numpy()
    if bool(newly_non_numeric.any()):
        return _failed(
            DEMAND_COLUMN,
            "must contain integer trip counts but has non-numeric entries.",
        )

    non_null_numeric = numeric.dropna()
    if len(non_null_numeric) > 0:
        # Whole-number check: demand is a count, so fractional values are invalid.
        if not (non_null_numeric == non_null_numeric.round()).all():
            return _failed(
                DEMAND_COLUMN,
                "must contain whole-number trip counts but has fractional values.",
            )
        if bool((non_null_numeric < 0).any()):
            return _failed(
                DEMAND_COLUMN,
                "must be non-negative (>= 0) but has negative values.",
            )

    return _passed()
