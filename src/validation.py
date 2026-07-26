"""Data_Validator - profiling and validation of raw and prepared NYC TLC data.

The golden rule of this project is that only real public NYC TLC data is used and
every reported number is defensible. Before any modeling begins, the raw
``fhvhv_2026-04.parquet`` file is loaded and profiled so the user sees real
numbers - row count, columns and dtypes, per-column nulls, the pickup date range,
and duplicate counts (Requirement 1).

This module provides the *core profiling functions* (Requirements 1.1-1.5, 1.8),
domain-violation flagging (Requirement 1.6), full report aggregation
(Requirement 1.8), multi-file orchestration across the Analysis_Window
(Requirement 1.7), and prepared-dataset reconciliation (Requirements 4.5, 13.4)
that confirms aggregated demand totals conserve the count of valid raw records
before modeling proceeds.

The functions here are pure with respect to a loaded DataFrame; ``load_parquet``
is the only I/O boundary, and it fails loudly - naming the path - when the file
is missing or unreadable so validation halts before any modeling (Requirement 1.8).

Design references:
- Components and Interfaces -> Data_Validator (`src/validation.py`)
- Data Models -> SchemaReport, NullStat, ValidationReport
- Error Handling -> Parquet file missing/unreadable raises a clear error naming the path
- Requirements 1.1, 1.2, 1.3, 1.4, 1.5, 1.8
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Optional

import pandas as pd

if TYPE_CHECKING:  # pragma: no cover - imported only for type checking
    from src.config import ScopeConfig
    from src.preparation import DemandSeries

#: Default demand column of a prepared DemandSeries (long format). Matches
#: ``src.preparation.DEMAND_COLUMN``; duplicated here as a plain string so
#: reconciliation needs only the demand column, not a runtime import of the
#: preparation pipeline (avoiding an import cycle).
DEFAULT_DEMAND_COLUMN = "demand"

#: Default FHVHV pickup timestamp column in the real NYC TLC schema. Used for the
#: date-range profiling that establishes the file's stated coverage (R1.4).
DEFAULT_PICKUP_COLUMN = "pickup_datetime"

#: Common pickup-timestamp column-name variants seen across NYC TLC feeds and
#: exports. The real FHVHV schema uses ``pickup_datetime``, but sibling feeds and
#: re-exports use casing/prefix variants (yellow/green taxis use
#: ``tpep_pickup_datetime`` / ``lpep_pickup_datetime``; older FHV uses
#: ``Pickup_DateTime``). ``_resolve_pickup_column`` falls back to these,
#: matched case-insensitively, when the requested column is absent so profiling
#: degrades gracefully instead of failing on a naming mismatch.
PICKUP_COLUMN_VARIANTS: tuple[str, ...] = (
    "pickup_datetime",
    "Pickup_datetime",
    "Pickup_DateTime",
    "pickup_date_time",
    "tpep_pickup_datetime",
    "lpep_pickup_datetime",
    "request_datetime",
)


@dataclass(frozen=True)
class SchemaReport:
    """Schema profile of a loaded DataFrame (Requirement 1.2).

    Captures the three facts the Data_Validator must report when a Parquet file is
    loaded: how many rows it has, which columns are present (in order), and the
    data type of each column.

    Attributes:
        row_count: Number of rows in the DataFrame.
        column_names: Column names in their DataFrame order.
        dtypes: Mapping of column name -> string representation of its dtype.
    """

    row_count: int
    column_names: list[str]
    dtypes: dict[str, str]


@dataclass(frozen=True)
class NullStat:
    """Null-value statistics for a single column (Requirement 1.3).

    Attributes:
        count: Number of null (missing) values in the column.
        percentage: Nulls as a percentage of total rows, in ``[0, 100]``. Defined
            as ``0.0`` for an empty DataFrame (no rows means no missing values).
    """

    count: int
    percentage: float


@dataclass(frozen=True)
class DomainViolation:
    """A single kind of out-of-domain data quality problem (Requirement 1.6).

    When a column contains values outside its valid domain - a pickup timestamp
    outside the file's stated month, or a negative fare/pay/count - the affected
    records are flagged rather than silently dropped. Each ``DomainViolation``
    groups one *kind* of problem and reports how many records are affected plus a
    concrete example so the user can see a real offending record.

    Attributes:
        violation_type: Stable identifier for the kind of violation, e.g.
            ``"pickup_outside_stated_month"`` or ``"negative_base_passenger_fare"``.
        count: Number of records affected by this violation type (>= 1; only
            violation types that actually occur are reported).
        example: One genuinely-violating record as a plain ``dict`` of
            column -> value (timestamps rendered as ISO strings), so it is easy to
            display and serialize. ``None`` only if no example could be captured.
    """

    violation_type: str
    count: int
    example: Optional[dict[str, Any]]


@dataclass(frozen=True)
class ValidationReport:
    """Aggregated profile of a loaded DataFrame (Requirement 1.8).

    Bundles every validation finding the Data_Validator must present with real
    numbers before modeling begins: the schema, per-column nulls, the pickup date
    range, the duplicate count, and any domain violations. One report is produced
    per file, so multi-file orchestration returns one ``ValidationReport`` per
    month of the Analysis_Window (Requirement 1.7).

    Attributes:
        schema: Row count, column names, and dtypes (Requirement 1.2).
        nulls: Per-column null count and percentage (Requirement 1.3).
        date_range: ``(min, max)`` pickup timestamp, or ``None`` when no pickup
            timestamp column is present or it has no usable values (Requirement 1.4).
        duplicate_count: Number of duplicate rows (Requirement 1.5).
        domain_violations: The domain violations found, one entry per violation
            type that actually occurs (Requirement 1.6).
    """

    schema: SchemaReport
    nulls: dict[str, NullStat]
    date_range: Optional[tuple[pd.Timestamp, pd.Timestamp]]
    duplicate_count: int
    domain_violations: list[DomainViolation] = field(default_factory=list)


@dataclass(frozen=True)
class ReconciliationReport:
    """Result of reconciling a prepared DemandSeries against the raw data (R4.5, R13.4).

    Demand is defined as the *count of trips* per ``(period, region)`` bucket, so
    the total demand summed across every bucket of the prepared series must equal
    the number of valid raw records that fed the aggregation - a conservation
    check (design Correctness Property 3). Zero-fill adds ``0``-demand rows, which
    do not change the total, so the equality still holds after zero-filling. This
    report captures that check so the runner can confirm the transformation
    conserved the data before modeling proceeds (Requirement 13.4).

    Attributes:
        total_demand: Sum of the demand column across all buckets of the prepared
            series (a non-negative integer count of trips).
        raw_valid_count: The expected count of valid raw records the prepared
            series was aggregated from.
        difference: ``total_demand - raw_valid_count``. ``0`` when the totals
            reconcile; non-zero (signed) when they do not, so the caller can see
            both the magnitude and direction of any discrepancy.
        reconciled: ``True`` iff ``total_demand == raw_valid_count`` (equivalently,
            ``difference == 0``).
    """

    total_demand: int
    raw_valid_count: int
    difference: int
    reconciled: bool


def load_parquet(path: str) -> pd.DataFrame:
    """Load a Parquet file into a DataFrame, failing loudly if it cannot be read.

    This is the only I/O boundary of the Data_Validator. Per the error-handling
    design, a missing or unreadable file raises a clear error that *names the
    path* so validation halts before any modeling proceeds (Requirement 1.8).
    Loading uses pandas with the ``pyarrow`` engine, matching the official NYC
    TLC Parquet distribution (Requirement 1.1).

    Args:
        path: Filesystem path to the ``.parquet`` file (e.g.
            ``"data/fhvhv_2026-04.parquet"``).

    Returns:
        The loaded ``pandas.DataFrame``.

    Raises:
        FileNotFoundError: If no file exists at ``path``.
        ValueError: If the file exists but cannot be read as Parquet (corrupt,
            wrong format, or otherwise unreadable). The path is named in the
            message either way.
    """
    try:
        return pd.read_parquet(path, engine="pyarrow")
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Could not load Parquet file: no file found at '{path}'. "
            "Validation cannot proceed until the file is present."
        ) from exc
    except Exception as exc:  # noqa: BLE001 - re-raised with the path for context
        raise ValueError(
            f"Could not read Parquet file at '{path}': {exc}. "
            "The file may be corrupt or not a valid Parquet file."
        ) from exc


def profile_schema(df: pd.DataFrame) -> SchemaReport:
    """Report the row count, column names, and dtype of each column (R1.2).

    Args:
        df: The loaded DataFrame to profile.

    Returns:
        A :class:`SchemaReport` whose fields equal the ground-truth shape and
        schema of ``df``.
    """
    return SchemaReport(
        row_count=int(len(df)),
        column_names=list(df.columns),
        dtypes={str(col): str(dtype) for col, dtype in df.dtypes.items()},
    )


def profile_nulls(df: pd.DataFrame) -> dict[str, NullStat]:
    """Report the count and percentage of null values for each column (R1.3).

    The percentage is computed against the total number of rows. For an empty
    DataFrame (zero rows) the percentage is defined as ``0.0`` to avoid division
    by zero - with no rows there are no missing values to report.

    Args:
        df: The loaded DataFrame to profile.

    Returns:
        A mapping of column name -> :class:`NullStat` (null count and percentage
        in ``[0, 100]``) for every column, in column order.
    """
    row_count = len(df)
    stats: dict[str, NullStat] = {}
    for col in df.columns:
        count = int(df[col].isna().sum())
        percentage = (count / row_count * 100.0) if row_count else 0.0
        stats[str(col)] = NullStat(count=count, percentage=percentage)
    return stats


def _resolve_pickup_column(df: pd.DataFrame, ts_col: str) -> Optional[str]:
    """Resolve the pickup timestamp column, tolerating common naming variants.

    Resolution order:

    1. Exact match on ``ts_col`` (fast path; preserves the requested column when
       it is present).
    2. Case-insensitive match on ``ts_col``.
    3. First available member of :data:`PICKUP_COLUMN_VARIANTS`, matched
       case-insensitively, so a differently-cased or sibling-feed timestamp
       column is still found.

    Args:
        df: The DataFrame whose columns are searched.
        ts_col: The requested pickup timestamp column name.

    Returns:
        The actual column name present in ``df`` to use, or ``None`` when no
        exact, case-insensitive, or known-variant match exists.
    """
    if ts_col in df.columns:
        return ts_col

    lower_to_actual = {str(c).lower(): str(c) for c in df.columns}

    requested = lower_to_actual.get(ts_col.lower())
    if requested is not None:
        return requested

    for variant in PICKUP_COLUMN_VARIANTS:
        match = lower_to_actual.get(variant.lower())
        if match is not None:
            return match

    return None


def pickup_date_range(
    df: pd.DataFrame, ts_col: str = DEFAULT_PICKUP_COLUMN
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Report the minimum and maximum pickup timestamp to establish the range (R1.4).

    The FHVHV pickup timestamp column is ``pickup_datetime`` by default; it is
    exposed as a parameter so the same profiler works if a different timestamp
    column is used. When the requested column is absent, common naming variants
    are resolved gracefully (see :func:`_resolve_pickup_column`) rather than
    failing outright. Null timestamps are ignored when computing the extremes.

    Args:
        df: The loaded DataFrame to profile.
        ts_col: Name of the pickup timestamp column. Defaults to
            ``"pickup_datetime"``.

    Returns:
        A ``(min_timestamp, max_timestamp)`` tuple.

    Raises:
        KeyError: If neither ``ts_col`` nor any known variant is a column of
            ``df``.
        ValueError: If the resolved column has no non-null timestamps to compute
            a range from (an empty or all-null column).
    """
    resolved = _resolve_pickup_column(df, ts_col)
    if resolved is None:
        raise KeyError(
            f"Pickup timestamp column '{ts_col}' not found (also tried common "
            f"variants {list(PICKUP_COLUMN_VARIANTS)}). "
            f"Available columns: {list(df.columns)}."
        )

    timestamps = pd.to_datetime(df[resolved], errors="coerce").dropna()
    if timestamps.empty:
        raise ValueError(
            f"Cannot establish a date range: column '{resolved}' has no non-null "
            "timestamp values."
        )
    return timestamps.min(), timestamps.max()


def count_duplicates(df: pd.DataFrame) -> int:
    """Report the count of duplicate records in the loaded data (R1.5).

    A record is counted as a duplicate when an identical row (across all columns)
    has already appeared earlier in the DataFrame. The first occurrence of each
    distinct row is not counted, so the result is the number of *extra* rows that
    could be removed by de-duplication.

    Args:
        df: The loaded DataFrame to profile.

    Returns:
        The number of duplicate rows (non-negative).
    """
    return int(df.duplicated().sum())


#: Columns that are conceptually counts/fares/pay/measures and must never be
#: negative in the real NYC TLC FHVHV schema. Used as the default set of columns
#: checked for negative values by ``flag_domain_violations`` when the caller does
#: not specify its own list. Any of these that are absent from the DataFrame are
#: simply skipped, so the same default works across schema subsets.
DEFAULT_NON_NEGATIVE_COLUMNS: tuple[str, ...] = (
    "trip_miles",
    "trip_time",
    "base_passenger_fare",
    "tolls",
    "bcf",
    "sales_tax",
    "congestion_surcharge",
    "airport_fee",
    "tips",
    "driver_pay",
    "demand",
    "trip_count",
)


def _row_to_example(df: pd.DataFrame, mask: pd.Series) -> Optional[dict[str, Any]]:
    """Return the first row selected by ``mask`` as a JSON-friendly ``dict``.

    Picks the first ``True`` position in ``mask`` (which is aligned to ``df``'s
    index) and renders that row as a plain dict, converting ``pd.Timestamp``
    values to ISO-8601 strings and numpy scalars to native Python types so the
    example is easy to display and serialize. Returns ``None`` if the mask selects
    nothing.
    """
    if not bool(mask.any()):
        return None
    idx = mask[mask].index[0]
    row = df.loc[idx]
    example: dict[str, Any] = {}
    for col, value in row.items():
        if isinstance(value, pd.Timestamp):
            example[str(col)] = value.isoformat()
        elif hasattr(value, "item") and not isinstance(value, (str, bytes)):
            # numpy scalar -> native python (guarded; leaves plain values alone)
            try:
                example[str(col)] = value.item()
            except (ValueError, AttributeError):
                example[str(col)] = value
        else:
            example[str(col)] = value
    return example


def _infer_stated_month(timestamps: pd.Series) -> Optional[pd.Period]:
    """Infer the file's stated month as the most common pickup year-month.

    NYC TLC publishes one month per file, so the overwhelming majority of pickup
    timestamps fall in that month; a small number of stray records from adjacent
    months are exactly the domain violations we want to flag. The stated month is
    therefore inferred as the modal ``Period[M]`` of the non-null pickup
    timestamps. Returns ``None`` when there are no usable timestamps.
    """
    valid = timestamps.dropna()
    if valid.empty:
        return None
    months = valid.dt.to_period("M")
    modes = months.mode()
    if modes.empty:
        return None
    return modes.iloc[0]


def flag_domain_violations(
    df: pd.DataFrame,
    scope: "ScopeConfig | None" = None,
    *,
    ts_col: str = DEFAULT_PICKUP_COLUMN,
    stated_month: Optional[tuple[int, int]] = None,
    non_negative_columns: Optional[Iterable[str]] = None,
) -> list[DomainViolation]:
    """Flag records whose values fall outside their valid domain (Requirement 1.6).

    Two families of violation are detected, each reported as a
    :class:`DomainViolation` with a count and a concrete example record:

    1. **Pickups outside the file's stated month.** NYC TLC ships one month per
       file, so a pickup timestamp in a different month is out of domain. The
       stated month is taken from ``stated_month`` when provided, otherwise
       inferred as the most common pickup year-month in the data
       (see :func:`_infer_stated_month`). Null timestamps are *not* treated as
       domain violations here - missing values are a separate null-profiling
       concern (Requirement 1.3).
    2. **Negative values** in columns that must be non-negative (fares, pay,
       trip counts, distances, times). The columns checked are
       ``non_negative_columns`` when given, otherwise the members of
       :data:`DEFAULT_NON_NEGATIVE_COLUMNS` that are present in ``df``. Each such
       column with at least one negative value yields its own violation type
       (``"negative_<column>"``).

    Only violation types that actually occur are returned, and each reported
    count is the number of genuinely-violating records for that type. The
    ``scope`` argument is accepted to match the Data_Validator interface and is
    reserved for future domain rules; the current checks do not depend on it.

    Args:
        df: The loaded DataFrame to check.
        scope: The project :class:`~src.config.ScopeConfig` (accepted for
            interface compatibility; unused by the current checks).
        ts_col: Name of the pickup timestamp column. Defaults to
            ``"pickup_datetime"``. If absent from ``df``, the month check is
            skipped.
        stated_month: Optional ``(year, month)`` the file is supposed to cover. If
            omitted, the stated month is inferred from the data.
        non_negative_columns: Optional iterable of column names that must be
            non-negative. If omitted, defaults to the standard FHVHV measure
            columns present in ``df``.

    Returns:
        A list of :class:`DomainViolation`, one per violation type that occurs,
        in a stable order (month violation first, then negative-value violations
        in column order).
    """
    violations: list[DomainViolation] = []

    # --- 1. Pickups outside the stated month --------------------------------
    if ts_col in df.columns:
        timestamps = pd.to_datetime(df[ts_col], errors="coerce")
        if stated_month is not None:
            stated_period: Optional[pd.Period] = pd.Period(
                year=stated_month[0], month=stated_month[1], freq="M"
            )
        else:
            stated_period = _infer_stated_month(timestamps)

        if stated_period is not None:
            record_months = timestamps.dt.to_period("M")
            outside_mask = timestamps.notna() & (record_months != stated_period)
            count = int(outside_mask.sum())
            if count:
                violations.append(
                    DomainViolation(
                        violation_type="pickup_outside_stated_month",
                        count=count,
                        example=_row_to_example(df, outside_mask),
                    )
                )

    # --- 2. Negative values in non-negative columns -------------------------
    if non_negative_columns is None:
        columns_to_check = [
            c for c in DEFAULT_NON_NEGATIVE_COLUMNS if c in df.columns
        ]
    else:
        columns_to_check = [c for c in non_negative_columns if c in df.columns]

    for col in columns_to_check:
        numeric = pd.to_numeric(df[col], errors="coerce")
        negative_mask = numeric < 0  # NaN < 0 is False, so nulls are ignored
        count = int(negative_mask.sum())
        if count:
            violations.append(
                DomainViolation(
                    violation_type=f"negative_{col}",
                    count=count,
                    example=_row_to_example(df, negative_mask),
                )
            )

    return violations


def build_validation_report(
    df: pd.DataFrame,
    scope: "ScopeConfig | None" = None,
    *,
    ts_col: str = DEFAULT_PICKUP_COLUMN,
    stated_month: Optional[tuple[int, int]] = None,
    non_negative_columns: Optional[Iterable[str]] = None,
) -> ValidationReport:
    """Aggregate all profiling checks into a single ``ValidationReport`` (R1.8).

    Runs the schema, null, date-range, duplicate, and domain-violation profilers
    against ``df`` and bundles them into one :class:`ValidationReport`. This is
    the object the runner presents to the user with real numbers before any
    modeling begins (Requirement 1.8).

    The date range is computed when a usable pickup timestamp column exists; if
    the column is absent or has no non-null timestamps, ``date_range`` is set to
    ``None`` rather than raising, so a report can still be produced for schema
    subsets and empty frames.

    Args:
        df: The loaded DataFrame to profile.
        scope: The project :class:`~src.config.ScopeConfig` (passed through to
            :func:`flag_domain_violations`).
        ts_col: Name of the pickup timestamp column. Defaults to
            ``"pickup_datetime"``.
        stated_month: Optional ``(year, month)`` for the month check; inferred
            from the data when omitted.
        non_negative_columns: Optional override of the columns checked for
            negative values.

    Returns:
        A fully-populated :class:`ValidationReport`.
    """
    schema = profile_schema(df)
    nulls = profile_nulls(df)

    date_range: Optional[tuple[pd.Timestamp, pd.Timestamp]]
    if ts_col in df.columns:
        try:
            date_range = pickup_date_range(df, ts_col)
        except (ValueError, KeyError):
            date_range = None
    else:
        date_range = None

    duplicate_count = count_duplicates(df)
    domain_violations = flag_domain_violations(
        df,
        scope,
        ts_col=ts_col,
        stated_month=stated_month,
        non_negative_columns=non_negative_columns,
    )

    return ValidationReport(
        schema=schema,
        nulls=nulls,
        date_range=date_range,
        duplicate_count=duplicate_count,
        domain_violations=domain_violations,
    )


def validate_month_frames(
    frames: Mapping[str, pd.DataFrame],
    scope: "ScopeConfig | None" = None,
    *,
    ts_col: str = DEFAULT_PICKUP_COLUMN,
    stated_months: Optional[Mapping[str, tuple[int, int]]] = None,
    non_negative_columns: Optional[Iterable[str]] = None,
) -> "OrderedDict[str, ValidationReport]":
    """Build one ``ValidationReport`` per already-loaded month DataFrame (R1.7).

    This is the in-memory core of multi-file orchestration: when the
    Analysis_Window is expanded beyond the first file, the same profiling
    criteria (schema, nulls, date range, duplicates, domain violations) are
    applied to every additional month's DataFrame, producing one report each.

    Args:
        frames: Mapping of a label (e.g. the file name or ``"2026-04"``) to its
            loaded DataFrame. Order is preserved in the returned mapping.
        scope: The project :class:`~src.config.ScopeConfig`.
        ts_col: Pickup timestamp column name applied to every frame.
        stated_months: Optional mapping of the same labels to a ``(year, month)``
            pair fixing each file's stated month. Labels not present are inferred
            from their data.
        non_negative_columns: Optional override of the negative-value columns.

    Returns:
        An ``OrderedDict`` mapping each label to its :class:`ValidationReport`,
        in the order the frames were supplied.
    """
    reports: "OrderedDict[str, ValidationReport]" = OrderedDict()
    for label, df in frames.items():
        stated_month = stated_months.get(label) if stated_months else None
        reports[label] = build_validation_report(
            df,
            scope,
            ts_col=ts_col,
            stated_month=stated_month,
            non_negative_columns=non_negative_columns,
        )
    return reports


def validate_month_files(
    paths: Iterable[str],
    scope: "ScopeConfig | None" = None,
    *,
    ts_col: str = DEFAULT_PICKUP_COLUMN,
    stated_months: Optional[Mapping[str, tuple[int, int]]] = None,
    non_negative_columns: Optional[Iterable[str]] = None,
) -> "OrderedDict[str, ValidationReport]":
    """Load and validate every additional month Parquet file, one report each (R1.7).

    When additional months are required to satisfy the selected Analysis_Window,
    this applies the full profiling suite (Requirement 1 criteria 2-6) to each
    file and returns one :class:`ValidationReport` per path. Loading goes through
    :func:`load_parquet`, so a missing or unreadable file raises a clear,
    path-naming error and orchestration halts before modeling (Requirement 1.8).

    Args:
        paths: Filesystem paths to the monthly ``.parquet`` files, in window
            order.
        scope: The project :class:`~src.config.ScopeConfig`.
        ts_col: Pickup timestamp column name applied to every file.
        stated_months: Optional mapping of *path* -> ``(year, month)`` fixing each
            file's stated month; inferred from data when a path is absent.
        non_negative_columns: Optional override of the negative-value columns.

    Returns:
        An ``OrderedDict`` mapping each path to its :class:`ValidationReport`, in
        the order the paths were supplied.
    """
    frames: "OrderedDict[str, pd.DataFrame]" = OrderedDict()
    for path in paths:
        frames[path] = load_parquet(path)
    return validate_month_frames(
        frames,
        scope,
        ts_col=ts_col,
        stated_months=stated_months,
        non_negative_columns=non_negative_columns,
    )


def revalidate_prepared(
    series: "DemandSeries",
    raw_valid_count: int,
    *,
    demand_column: str = DEFAULT_DEMAND_COLUMN,
) -> ReconciliationReport:
    """Reconcile a prepared DemandSeries against the raw valid record count (R4.5, R13.4).

    When preparation is complete, the Data_Validator re-validates the prepared
    dataset and confirms that aggregated totals reconcile with the raw record
    counts before the Project proceeds to modeling (Requirements 4.5, 13.4).
    Because demand is the *count of trips* per ``(period, region)`` bucket, the
    total demand summed across every bucket must equal the number of valid raw
    records that were aggregated - a conservation invariant (design Correctness
    Property 3). Zero-filled buckets contribute ``0`` and therefore leave the
    total unchanged, so the equality holds for the zero-filled series too.

    The demand column is summed as an integer trip count. Any missing values in
    the demand column (e.g. from a malformed series) are treated as ``0`` for the
    total so the check degrades to reporting a mismatch rather than raising.

    Args:
        series: The prepared :data:`~src.preparation.DemandSeries` (long format)
            to reconcile. Only its ``demand_column`` is read here.
        raw_valid_count: The expected number of valid raw records the series was
            aggregated from (e.g. rows remaining after validity handling).
        demand_column: Name of the demand column in ``series``. Defaults to
            ``"demand"``.

    Returns:
        A :class:`ReconciliationReport` with the summed ``total_demand``, the
        ``raw_valid_count``, their signed ``difference``, and ``reconciled`` set
        to ``True`` only when the two totals are equal.

    Raises:
        KeyError: If ``demand_column`` is not a column of ``series``.
    """
    if demand_column not in series.columns:
        raise KeyError(
            f"Demand column '{demand_column}' not found in the prepared series. "
            f"Available columns: {list(series.columns)}."
        )

    total_demand = int(
        pd.to_numeric(series[demand_column], errors="coerce").fillna(0).sum()
    )
    raw_valid_count = int(raw_valid_count)
    difference = total_demand - raw_valid_count

    return ReconciliationReport(
        total_demand=total_demand,
        raw_valid_count=raw_valid_count,
        difference=difference,
        reconciled=difference == 0,
    )
