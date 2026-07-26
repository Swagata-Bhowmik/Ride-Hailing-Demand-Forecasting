"""Data_Preparation_Pipeline - pure functions that reshape validated raw trip
records into the forecasting dataset (Requirement 4).

This module is the most property-rich component of the project. It contains only
deterministic functions with no I/O: they receive already-loaded ``pandas``
DataFrames and return new DataFrames, which is what makes the pure-logic
correctness properties (see design "Correctness Properties") machine-checkable.

Task 4.1 implements the first two steps of the pipeline:

* :func:`map_zones_to_regions` - join ``PULocationID`` to its borough using a
  ``taxi_zone_lookup`` table, materializing the Geographic_Grain as a ``region``
  column.
* :func:`aggregate_demand` - count trips per ``(period, region)`` bucket, where
  ``period`` is the pickup timestamp truncated to the Time_Grain (daily -> the
  calendar day) and ``region`` is the borough. This produces the long-format
  :data:`DemandSeries` at the defined Time_Grain and Geographic_Grain
  (Requirement 4.1).

Later tasks extend the same ``DemandSeries`` in place:
zero-fill (4.2), documented invalid-record handling (4.3), lag features (4.4),
the ``prepare`` orchestrator (4.5), and reconciliation (4.6).

Design references:
- Components and Interfaces -> Data_Preparation_Pipeline (`src/preparation.py`)
- Data Models -> DemandSeries (long format: period, region, demand >= 0)
- Correctness Properties -> Property 3 (aggregation correctness and reconciliation)
- Requirements 4.1
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterable, Optional

import pandas as pd

from src.validation import (
    DEFAULT_NON_NEGATIVE_COLUMNS,
    DEFAULT_PICKUP_COLUMN,
    _infer_stated_month,
)

if TYPE_CHECKING:  # pragma: no cover - imported only for type checking
    from src.config import ScopeConfig

# --- Canonical column names --------------------------------------------------

#: Raw FHVHV pickup-location column, joined to the zone lookup to get a borough.
PICKUP_LOCATION_COLUMN = "PULocationID"

#: Raw FHVHV pickup timestamp column, truncated to the Time_Grain to form ``period``.
PICKUP_DATETIME_COLUMN = "pickup_datetime"

#: Geographic_Grain column materialized by :func:`map_zones_to_regions` and used
#: as the spatial key throughout the pipeline (Requirement 4, Geographic_Grain).
REGION_COLUMN = "region"

#: Temporal key of the DemandSeries: the pickup timestamp truncated to Time_Grain.
PERIOD_COLUMN = "period"

#: The forecasting target: trip count per ``(period, region)`` bucket (>= 0).
DEMAND_COLUMN = "demand"

#: The three long-format columns every DemandSeries carries after aggregation.
#: Later tasks append lag/calendar feature columns alongside these.
DEMAND_SERIES_COLUMNS: tuple[str, ...] = (PERIOD_COLUMN, REGION_COLUMN, DEMAND_COLUMN)

# --- Zone-lookup column names (real NYC TLC taxi_zone_lookup schema) ---------

#: Zone-lookup key column matching the raw ``PULocationID`` values.
ZONE_LOOKUP_ID_COLUMN = "LocationID"

#: Zone-lookup borough column supplying the Geographic_Grain value.
ZONE_LOOKUP_BOROUGH_COLUMN = "Borough"


#: A DemandSeries is a long-format ``pandas.DataFrame`` with (at minimum) the
#: columns :data:`DEMAND_SERIES_COLUMNS`:
#:
#: ============  =========  ========================================================
#: column        dtype      meaning
#: ============  =========  ========================================================
#: ``period``    timestamp  pickup timestamp truncated to the Time_Grain (daily ->
#:                          midnight of the calendar day)
#: ``region``    str        borough (Geographic_Grain)
#: ``demand``    int >= 0   count of trips in that ``(period, region)`` bucket
#: ============  =========  ========================================================
#:
#: Kept deliberately as a plain DataFrame (not a wrapper class) so downstream
#: tasks - zero-fill, lag features, calendar features - can add columns without
#: fighting a rigid schema, consistent with what tasks 4.2/4.4 extend.
DemandSeries = pd.DataFrame

#: Time_Grain -> a pandas floor/truncation frequency alias. Only the daily grain
#: is supported for now (the project default); other grains raise a clear error
#: until a later task wires them in.
_TIME_GRAIN_FREQ: dict[str, str] = {
    "daily": "D",
}


def map_zones_to_regions(
    df: pd.DataFrame,
    zone_lookup: pd.DataFrame,
    *,
    pickup_location_column: str = PICKUP_LOCATION_COLUMN,
    lookup_id_column: str = ZONE_LOOKUP_ID_COLUMN,
    lookup_borough_column: str = ZONE_LOOKUP_BOROUGH_COLUMN,
    region_column: str = REGION_COLUMN,
) -> pd.DataFrame:
    """Add a ``region`` (borough) column by joining pickup locations to the lookup.

    The Geographic_Grain of the project is the borough. Raw FHVHV records carry
    only a numeric ``PULocationID``; the official NYC TLC ``taxi_zone_lookup``
    table maps each ``LocationID`` to its ``Borough``. This function performs that
    left join and materializes the result as a new ``region`` column, leaving the
    original rows and their order untouched (a left join preserves ``df``'s row
    count and order).

    The join is left-preserving on purpose: a ``PULocationID`` with no matching
    lookup row keeps its record but gets ``NaN`` for ``region``, so nothing is
    silently dropped here - documented handling of such records is a later task's
    responsibility (invalid-record handling, task 4.3).

    Args:
        df: Raw trip records containing ``pickup_location_column``.
        zone_lookup: The ``taxi_zone_lookup`` table containing
            ``lookup_id_column`` and ``lookup_borough_column``.
        pickup_location_column: Name of the pickup-location column in ``df``.
            Defaults to ``"PULocationID"``.
        lookup_id_column: Name of the id column in ``zone_lookup``. Defaults to
            ``"LocationID"``.
        lookup_borough_column: Name of the borough column in ``zone_lookup``.
            Defaults to ``"Borough"``.
        region_column: Name of the region column to create. Defaults to
            ``"region"``.

    Returns:
        A new DataFrame equal to ``df`` with an added ``region_column`` holding
        the borough for each row (``NaN`` where the location id is unmatched).

    Raises:
        KeyError: If ``pickup_location_column`` is absent from ``df`` or the
            lookup id/borough columns are absent from ``zone_lookup``.
    """
    if pickup_location_column not in df.columns:
        raise KeyError(
            f"Pickup-location column '{pickup_location_column}' not found in df. "
            f"Available columns: {list(df.columns)}."
        )
    for col, where in (
        (lookup_id_column, "zone_lookup"),
        (lookup_borough_column, "zone_lookup"),
    ):
        if col not in zone_lookup.columns:
            raise KeyError(
                f"Column '{col}' not found in {where}. "
                f"Available columns: {list(zone_lookup.columns)}."
            )

    # Reduce the lookup to the two relevant columns and drop duplicate ids so the
    # left join stays one-to-one and never fans out df's rows.
    lookup = (
        zone_lookup[[lookup_id_column, lookup_borough_column]]
        .drop_duplicates(subset=[lookup_id_column])
        .rename(columns={lookup_borough_column: region_column})
    )

    merged = df.merge(
        lookup,
        how="left",
        left_on=pickup_location_column,
        right_on=lookup_id_column,
    )

    # Drop the redundant lookup-id column introduced by the merge when it differs
    # from the pickup-location column, keeping df's original schema plus region.
    if lookup_id_column != pickup_location_column and lookup_id_column in merged.columns:
        merged = merged.drop(columns=[lookup_id_column])

    # Preserve df's original row index so the result lines up with the input.
    merged.index = df.index
    return merged


def _truncate_to_time_grain(timestamps: pd.Series, time_grain: str) -> pd.Series:
    """Truncate pickup timestamps to the Time_Grain, forming the ``period`` key.

    For the daily grain this floors each timestamp to midnight of its calendar day
    (so all trips on the same day share one ``period``). The result is returned as
    a datetime Series, matching the DemandSeries contract that ``period`` is a
    timestamp at the Time_Grain.

    Args:
        timestamps: The pickup timestamp Series.
        time_grain: The Time_Grain value (e.g. ``"daily"``).

    Returns:
        A datetime Series of truncated ``period`` values.

    Raises:
        ValueError: If ``time_grain`` is not a supported grain.
    """
    grain = time_grain.lower()
    if grain not in _TIME_GRAIN_FREQ:
        raise ValueError(
            f"Unsupported time_grain '{time_grain}'. "
            f"Supported grains: {sorted(_TIME_GRAIN_FREQ)}."
        )
    coerced = pd.to_datetime(timestamps, errors="coerce")
    return coerced.dt.floor(_TIME_GRAIN_FREQ[grain])


def aggregate_demand(
    df: pd.DataFrame,
    scope: "ScopeConfig",
    *,
    pickup_datetime_column: str = PICKUP_DATETIME_COLUMN,
    region_column: str = REGION_COLUMN,
) -> DemandSeries:
    """Count trips per ``(period, region)`` to build the long-format DemandSeries.

    Demand is defined as the count of trips in each ``(period, region)`` bucket,
    where ``period`` is the pickup timestamp truncated to the Time_Grain from
    ``scope`` (daily -> the calendar day) and ``region`` is the borough produced
    by :func:`map_zones_to_regions`. This realizes Requirement 4.1: aggregating
    raw trip records into a demand series at the defined Time_Grain and
    Geographic_Grain.

    Every raw record with both a usable ``period`` and a non-null ``region``
    contributes exactly one to its bucket, so the total demand summed across all
    buckets equals the number of such records (the conservation property the
    reconciliation step relies on - design Property 3). Records with a null pickup
    timestamp or an unmatched region are excluded from aggregation; handling those
    is the job of a later invalid-record task (4.3).

    Args:
        df: Trip records with a pickup timestamp column and a ``region`` column
            (typically the output of :func:`map_zones_to_regions`).
        scope: The :class:`~src.config.ScopeConfig` supplying the Time_Grain.
        pickup_datetime_column: Name of the pickup timestamp column. Defaults to
            ``"pickup_datetime"``.
        region_column: Name of the region column. Defaults to ``"region"``.

    Returns:
        A :data:`DemandSeries` DataFrame with columns ``period``, ``region``,
        ``demand`` - one row per observed ``(period, region)`` bucket, sorted by
        ``period`` then ``region``, with ``demand`` a non-negative integer count.
        Buckets with no trips are *not* added here; explicit zero-fill across the
        window is a later task (4.2).

    Raises:
        KeyError: If the pickup timestamp or region column is absent from ``df``.
        ValueError: If the scope's Time_Grain is unsupported.
    """
    if pickup_datetime_column not in df.columns:
        raise KeyError(
            f"Pickup timestamp column '{pickup_datetime_column}' not found in df. "
            f"Available columns: {list(df.columns)}."
        )
    if region_column not in df.columns:
        raise KeyError(
            f"Region column '{region_column}' not found in df. Run "
            "map_zones_to_regions first. "
            f"Available columns: {list(df.columns)}."
        )

    period = _truncate_to_time_grain(df[pickup_datetime_column], scope.time_grain)
    region = df[region_column]

    work = pd.DataFrame({PERIOD_COLUMN: period, REGION_COLUMN: region})
    # Exclude rows lacking a usable period or region: they cannot be placed in a
    # bucket and must not silently inflate counts (handled explicitly in 4.3).
    work = work.dropna(subset=[PERIOD_COLUMN, REGION_COLUMN])

    if work.empty:
        return pd.DataFrame(
            {
                PERIOD_COLUMN: pd.Series([], dtype="datetime64[ns]"),
                REGION_COLUMN: pd.Series([], dtype="object"),
                DEMAND_COLUMN: pd.Series([], dtype="int64"),
            }
        )

    grouped = (
        work.groupby([PERIOD_COLUMN, REGION_COLUMN], sort=True)
        .size()
        .reset_index(name=DEMAND_COLUMN)
    )
    grouped[DEMAND_COLUMN] = grouped[DEMAND_COLUMN].astype("int64")
    return grouped.reset_index(drop=True)


def fill_missing_periods(
    series: DemandSeries,
    scope: "ScopeConfig",
    regions: "list[str] | None" = None,
    *,
    period_column: str = PERIOD_COLUMN,
    region_column: str = REGION_COLUMN,
    demand_column: str = DEMAND_COLUMN,
) -> DemandSeries:
    """Zero-fill the DemandSeries so every ``(period, region)`` in the window exists.

    :func:`aggregate_demand` only emits rows for ``(period, region)`` buckets that
    actually contained trips, so quiet days or boroughs simply go missing. Every
    candidate model, though, needs a complete, contiguous grid: one demand value
    for each period and region across the whole Analysis_Window. This function
    materializes that grid.

    It builds the full set of periods spanning the Analysis_Window at the
    Time_Grain (daily -> every calendar day from ``scope.window_start`` to
    ``scope.window_end`` inclusive) and the full set of regions, takes their
    cartesian product, left-joins the observed demand onto it, and fills every
    gap with an integer ``0``. The result therefore contains *exactly one* row per
    ``(period, region)`` combination in the window - none omitted - with ``demand``
    equal to the observed count or ``0`` where no input records existed. This is
    the behavior asserted by design Property 4 (Zero-fill completeness,
    Requirement 4.3) and checked by the property test in task 4.8.

    The region set defaults to the distinct regions present in ``series`` but can
    be overridden via ``regions`` so boroughs absent from the data (yet part of
    the Geographic_Grain) are still represented as all-zero rows.

    Args:
        series: A :data:`DemandSeries` (e.g. the output of
            :func:`aggregate_demand`) with ``period``, ``region`` and ``demand``
            columns. Its ``period`` values are truncated to the Time_Grain.
        scope: The :class:`~src.config.ScopeConfig` supplying the
            Analysis_Window (``window_start``/``window_end``) and the Time_Grain.
        regions: Optional explicit list of regions to include. When ``None``
            (default), the distinct regions observed in ``series`` are used. When
            provided, exactly these regions populate the grid so every named
            borough is present even if absent from the data.
        period_column: Name of the period column. Defaults to ``"period"``.
        region_column: Name of the region column. Defaults to ``"region"``.
        demand_column: Name of the demand column. Defaults to ``"demand"``.

    Returns:
        A :data:`DemandSeries` with one row for every ``(period, region)`` in the
        window, sorted by ``period`` then ``region``, ``demand`` as ``int64``
        (``0`` where no input records existed). Original observed demand values
        are preserved.

    Raises:
        KeyError: If ``series`` is missing any of the period/region/demand columns.
        ValueError: If the scope's Time_Grain is unsupported.
    """
    for col in (period_column, region_column, demand_column):
        if col not in series.columns:
            raise KeyError(
                f"Column '{col}' not found in series. "
                f"Available columns: {list(series.columns)}."
            )

    grain = scope.time_grain.lower()
    if grain not in _TIME_GRAIN_FREQ:
        raise ValueError(
            f"Unsupported time_grain '{scope.time_grain}'. "
            f"Supported grains: {sorted(_TIME_GRAIN_FREQ)}."
        )

    # Complete, contiguous set of periods across the window at the Time_Grain.
    # date_range with an inclusive end covers both window_start and window_end.
    all_periods = pd.date_range(
        start=pd.Timestamp(scope.window_start),
        end=pd.Timestamp(scope.window_end),
        freq=_TIME_GRAIN_FREQ[grain],
    )

    # Complete set of regions: explicit override, else distinct observed regions.
    if regions is not None:
        # De-duplicate while preserving the caller's ordering intent; final output
        # is sorted anyway, so ordering here only affects the intermediate grid.
        seen: dict = {}
        for r in regions:
            seen.setdefault(r, None)
        all_regions = list(seen)
    else:
        all_regions = list(pd.unique(series[region_column].dropna()))

    # Cartesian product (period x region) forms the complete grid.
    grid = pd.MultiIndex.from_product(
        [all_periods, all_regions],
        names=[period_column, region_column],
    ).to_frame(index=False)

    if grid.empty:
        return pd.DataFrame(
            {
                period_column: pd.Series([], dtype="datetime64[ns]"),
                region_column: pd.Series([], dtype="object"),
                demand_column: pd.Series([], dtype="int64"),
            }
        )

    # Collapse any duplicate observed buckets so the left-join stays one-to-one.
    observed = (
        series[[period_column, region_column, demand_column]]
        .groupby([period_column, region_column], as_index=False, sort=False)[demand_column]
        .sum()
    )

    filled = grid.merge(observed, how="left", on=[period_column, region_column])
    filled[demand_column] = (
        filled[demand_column].fillna(0).astype("int64")
    )

    filled = filled.sort_values(
        by=[period_column, region_column], kind="stable"
    ).reset_index(drop=True)
    return filled


# --- Documented invalid-record handling (Requirement 4.4) --------------------

#: The default, documented handling rule applied to invalid records. Dropping the
#: offending rows is a defensible, transparent choice: the demand series is a
#: *count* of trips per bucket, so a record that cannot be trusted (a pickup
#: outside the file's stated month, or a negative fare/pay/count that cannot be a
#: real measurement) would otherwise silently pollute those counts. Every drop is
#: recorded in the :class:`HandlingLog` with a per-type breakdown, so nothing is
#: removed silently - consistent with the error-handling design ("invalid records
#: get a documented handling rule and are logged, not silently dropped").
HANDLING_RULE_DROP = "drop"

#: Human-readable description of the default drop rule, stored on the HandlingLog
#: so the applied policy travels with the counts it produced.
_DROP_RULE_DESCRIPTION = (
    "Records failing the Requirement 1 domain checks (pickup timestamp outside "
    "the file's stated month, or a negative value in a non-negative measure "
    "column) are dropped from the dataset. Each removed record is counted by "
    "violation type in this log so the handling is documented, not silent."
)


@dataclass(frozen=True)
class HandlingLog:
    """Audit record of how invalid records were handled (Requirement 4.4).

    :func:`apply_validity_rules` applies a documented handling rule to records
    that fail the Requirement 1 domain checks and returns one of these logs
    alongside the cleaned data. The log makes the handling transparent: it names
    the rule, describes it in plain language, and reports both the total number of
    invalid records handled and a per-violation-type breakdown.

    ``total_invalid_handled`` counts *distinct* records that were handled, so a
    record that violates more than one check (for example a negative fare on a
    pickup outside the stated month) is counted once in the total even though it
    contributes to two entries in ``counts_by_type``. This is the count design
    Property 5 asserts against: for a dataset with ``n`` injected invalid records,
    after handling no invalid record remains and ``total_invalid_handled == n``.

    Attributes:
        rule: Stable identifier of the applied handling rule (e.g.
            :data:`HANDLING_RULE_DROP`).
        rule_description: Plain-language description of what the rule did.
        total_invalid_handled: Number of distinct records handled (rows that
            failed at least one check).
        counts_by_type: Mapping of violation type -> number of records flagged for
            that type. Types mirror :func:`~src.validation.flag_domain_violations`
            (``"pickup_outside_stated_month"``, ``"negative_<column>"``, and,
            when enabled, ``"duplicate"`` / ``"null_<column>"``). Overlapping
            records may appear under more than one type, so these can sum to more
            than ``total_invalid_handled``.
        input_row_count: Number of rows in the input DataFrame.
        output_row_count: Number of rows remaining after handling.
    """

    rule: str
    rule_description: str
    total_invalid_handled: int
    counts_by_type: dict[str, int] = field(default_factory=dict)
    input_row_count: int = 0
    output_row_count: int = 0


def _domain_violation_masks(
    df: pd.DataFrame,
    *,
    ts_col: str,
    stated_month: Optional[tuple[int, int]],
    non_negative_columns: Optional[Iterable[str]],
    check_duplicates: bool,
    null_columns: Optional[Iterable[str]],
) -> "dict[str, pd.Series]":
    """Compute one boolean mask per violation type, aligned to ``df``'s index.

    The masks are computed with exactly the same logic as
    :func:`~src.validation.flag_domain_violations` so that handling and flagging
    stay consistent: whatever that function would report as a violation is what
    this function selects for handling. Two optional extra checks (duplicates and
    nulls) are supported for callers who want to handle them here too; both are
    off by default so the core behavior matches ``flag_domain_violations``.

    Returns:
        A mapping of ``violation_type -> boolean Series``. Only types that have at
        least one offending record are included.
    """
    masks: "dict[str, pd.Series]" = {}

    # --- 1. Pickups outside the stated month (mirrors flag_domain_violations) -
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
            if bool(outside_mask.any()):
                masks["pickup_outside_stated_month"] = outside_mask

    # --- 2. Negative values in non-negative columns --------------------------
    if non_negative_columns is None:
        columns_to_check = [c for c in DEFAULT_NON_NEGATIVE_COLUMNS if c in df.columns]
    else:
        columns_to_check = [c for c in non_negative_columns if c in df.columns]

    for col in columns_to_check:
        numeric = pd.to_numeric(df[col], errors="coerce")
        negative_mask = numeric < 0  # NaN < 0 is False, so nulls are ignored
        if bool(negative_mask.any()):
            masks[f"negative_{col}"] = negative_mask

    # --- 3. (optional) Duplicate rows ----------------------------------------
    if check_duplicates:
        duplicate_mask = df.duplicated()
        if bool(duplicate_mask.any()):
            masks["duplicate"] = duplicate_mask

    # --- 4. (optional) Nulls in required columns -----------------------------
    if null_columns is not None:
        for col in null_columns:
            if col not in df.columns:
                continue
            null_mask = df[col].isna()
            if bool(null_mask.any()):
                masks[f"null_{col}"] = null_mask

    return masks


def apply_validity_rules(
    df: pd.DataFrame,
    scope: "ScopeConfig | None" = None,
    *,
    rule: str = HANDLING_RULE_DROP,
    ts_col: str = DEFAULT_PICKUP_COLUMN,
    stated_month: Optional[tuple[int, int]] = None,
    non_negative_columns: Optional[Iterable[str]] = None,
    check_duplicates: bool = False,
    null_columns: Optional[Iterable[str]] = None,
) -> "tuple[pd.DataFrame, HandlingLog]":
    """Apply a documented handling rule to invalid records (Requirement 4.4).

    Records that fail the Requirement 1 domain checks - a pickup timestamp outside
    the file's stated month, or a negative value in a column that must be
    non-negative - are identified and handled according to a *documented* rule,
    then the handling is recorded in a :class:`HandlingLog`. Nothing is dropped
    silently: every removed record is counted, broken down by violation type, and
    the applied rule is named and described in the returned log. This satisfies
    the error-handling design ("invalid records get a documented handling rule and
    are logged, not silently dropped").

    The offending records are found by computing the violation masks directly
    (see :func:`_domain_violation_masks`) using the *same* logic as
    :func:`~src.validation.flag_domain_violations`, so handling and flagging never
    disagree. The stated month is taken from ``stated_month`` when given, else
    inferred from the data (the most common pickup year-month) exactly as the
    validator does. Two optional checks - duplicate rows and nulls in required
    columns - can be enabled by the caller; both are off by default so the core
    behavior matches the validator.

    Handling rules:

    * ``"drop"`` (default): remove every record that fails at least one check.
      Because all offending rows are removed, the returned DataFrame contains no
      records violating the checks - the post-condition design Property 5 relies
      on. The row order and index of the surviving records are preserved.

    Args:
        df: The trip records to clean.
        scope: The project :class:`~src.config.ScopeConfig` (accepted for
            interface compatibility; the current checks do not depend on it).
        rule: The handling rule to apply. Only :data:`HANDLING_RULE_DROP` is
            currently supported.
        ts_col: Name of the pickup timestamp column. Defaults to
            ``"pickup_datetime"``. If absent from ``df``, the month check is
            skipped.
        stated_month: Optional ``(year, month)`` the file is supposed to cover.
            Inferred from the data when omitted, consistent with the validator.
        non_negative_columns: Optional override of the columns checked for
            negative values. Defaults to the standard FHVHV measure columns
            present in ``df``.
        check_duplicates: When ``True``, duplicate rows are also treated as
            invalid and handled. Defaults to ``False``.
        null_columns: Optional iterable of columns whose null values are treated
            as invalid and handled. Defaults to ``None`` (no null handling).

    Returns:
        A ``(cleaned_df, HandlingLog)`` tuple. ``cleaned_df`` is a new DataFrame
        with the invalid records handled per ``rule`` (for ``"drop"``, the invalid
        rows removed); ``HandlingLog`` documents the rule and the counts.

    Raises:
        ValueError: If ``rule`` is not a supported handling rule.
    """
    if rule != HANDLING_RULE_DROP:
        raise ValueError(
            f"Unsupported handling rule '{rule}'. "
            f"Supported rules: ['{HANDLING_RULE_DROP}']."
        )

    input_row_count = int(len(df))

    masks = _domain_violation_masks(
        df,
        ts_col=ts_col,
        stated_month=stated_month,
        non_negative_columns=non_negative_columns,
        check_duplicates=check_duplicates,
        null_columns=null_columns,
    )

    counts_by_type = {vtype: int(mask.sum()) for vtype, mask in masks.items()}

    # Union of all violation masks: the set of distinct records to handle. A
    # record failing several checks is counted once here (total) though it shows
    # up under each of its types in counts_by_type.
    invalid_mask = pd.Series(False, index=df.index)
    for mask in masks.values():
        invalid_mask = invalid_mask | mask
    total_invalid_handled = int(invalid_mask.sum())

    # Apply the documented rule. "drop" removes offending rows, preserving the
    # order and index of the survivors so downstream steps line up.
    cleaned_df = df.loc[~invalid_mask].copy()

    log = HandlingLog(
        rule=rule,
        rule_description=_DROP_RULE_DESCRIPTION,
        total_invalid_handled=total_invalid_handled,
        counts_by_type=counts_by_type,
        input_row_count=input_row_count,
        output_row_count=int(len(cleaned_df)),
    )
    return cleaned_df, log


# --- Lag feature generation (Requirement 4.6) --------------------------------

#: Template for the lag column names added by :func:`add_lag_features`. Lag ``k``
#: becomes column ``lag_{k}`` (e.g. ``lag_1``, ``lag_7``, ``lag_14``), matching
#: the DemandSeries schema in the design's Data Models table.
LAG_COLUMN_TEMPLATE = "lag_{k}"


def lag_column_name(k: int) -> str:
    """Return the DemandSeries column name for lag ``k`` (e.g. ``k=7`` -> ``lag_7``)."""
    return LAG_COLUMN_TEMPLATE.format(k=k)


def add_lag_features(
    series: DemandSeries,
    lags: "list[int]",
    *,
    period_column: str = PERIOD_COLUMN,
    region_column: str = REGION_COLUMN,
    demand_column: str = DEMAND_COLUMN,
) -> DemandSeries:
    """Add per-region lag features ``lag_k[t] = demand[t-k]`` (Requirement 4.6).

    Lag features let the ML models (XGBoost, and any model consuming engineered
    predictors) see recent history: the demand ``k`` periods ago. This function
    adds one column per requested lag ``k`` - named ``lag_{k}`` (``lag_1``,
    ``lag_7``, ...) - where, *within each region*, the value at period ``t`` is
    the demand at period ``t - k``. The first ``k`` periods of every region have
    no prior value and are therefore ``NaN``.

    Correctness hinges on two things (design Property 6):

    * **Per-region shift, no leakage.** The shift is computed within each region
      (group by ``region``, then shift by ``k``), so a region's earliest periods
      never borrow demand from another region. Lags never cross the region
      boundary.
    * **Ordered by period.** Within each region the rows are sorted by ``period``
      before shifting, so "``t - k``" means the ``k``-th earlier period rather
      than whatever order the input happened to arrive in. The original row order
      of ``series`` is restored on the way out, so only the new columns are added.

    Because a shift by ``k`` leaves the first ``k`` rows of each region without a
    source value, those entries are ``NaN`` (and the column is therefore a
    nullable float dtype, matching the ``int/NaN`` lag columns in the design's
    DemandSeries schema).

    Args:
        series: A :data:`DemandSeries` with at least ``period``, ``region`` and
            ``demand`` columns (typically the zero-filled output of
            :func:`fill_missing_periods`). Existing columns are preserved.
        lags: The lag offsets to add (e.g. ``scope.lags`` -> ``[1, 7, 14]``). Each
            ``k`` produces a ``lag_{k}`` column. Duplicate lags are added once.
        period_column: Name of the period column. Defaults to ``"period"``.
        region_column: Name of the region column. Defaults to ``"region"``.
        demand_column: Name of the demand column. Defaults to ``"demand"``.

    Returns:
        A new :data:`DemandSeries` equal to ``series`` (same rows, same order,
        same index) with an added ``lag_{k}`` column for each requested lag. The
        first ``k`` periods of each region are ``NaN`` in ``lag_{k}``.

    Raises:
        KeyError: If any of the period/region/demand columns is absent.
        ValueError: If any lag ``k`` is not a positive integer.
    """
    for col in (period_column, region_column, demand_column):
        if col not in series.columns:
            raise KeyError(
                f"Column '{col}' not found in series. "
                f"Available columns: {list(series.columns)}."
            )

    for k in lags:
        if not isinstance(k, (int,)) or isinstance(k, bool) or k <= 0:
            raise ValueError(
                f"Lag values must be positive integers; got {k!r}."
            )

    result = series.copy()

    if result.empty:
        # Still materialize the requested lag columns (empty, float dtype) so the
        # output schema is stable regardless of whether any rows are present.
        for k in dict.fromkeys(lags):
            result[lag_column_name(k)] = pd.Series([], dtype="float64")
        return result

    # Sort by (region, period) so that within each region rows are in temporal
    # order; a per-region shift by k then means "k periods earlier in this region".
    # We keep the original index so we can restore the caller's row order at the end.
    ordered = result.sort_values(
        by=[region_column, period_column], kind="stable"
    )
    grouped_demand = ordered.groupby(region_column, sort=False)[demand_column]

    for k in dict.fromkeys(lags):  # de-duplicate lags, preserve order
        # Per-region shift: earliest k periods of each region become NaN, and no
        # value ever crosses a region boundary.
        shifted = grouped_demand.shift(k)
        # Reindex back to the caller's original row order via the preserved index.
        result[lag_column_name(k)] = shifted.reindex(result.index)

    return result


# --- Prepare orchestrator with before/after examples (Requirement 4.2) -------

#: How many real rows to capture in each before/after sample. Kept small so the
#: examples are readable in a notebook/dashboard while still being genuine slices
#: of the actual data (Requirement 4.2 asks for a *real* example, not a summary).
_BEFORE_AFTER_SAMPLE_ROWS = 5


@dataclass(frozen=True)
class BeforeAfter:
    """A real before-and-after example of a single preparation transformation (R4.2).

    Requirement 4.2 requires that when a data transformation is applied, the
    pipeline presents a *real* before-and-after example of the affected data.
    :func:`prepare` records one of these for every stage it runs, capturing
    genuine slices of the actual DataFrames flowing through the pipeline (not
    fabricated or summarized data) so the transformation can be shown and audited
    in the notebook and dashboard.

    Attributes:
        name: Short, stable identifier of the transformation stage (e.g.
            ``"apply_validity_rules"``).
        description: Plain-language description of what the stage did to the data.
        before_sample: A small real sample (up to :data:`_BEFORE_AFTER_SAMPLE_ROWS`
            head rows) of the data *before* the stage ran, as a DataFrame copy.
        after_sample: The corresponding small real sample of the data *after* the
            stage ran, as a DataFrame copy.
        rows_before: Total number of rows in the input to the stage.
        rows_after: Total number of rows in the output of the stage.
    """

    name: str
    description: str
    before_sample: pd.DataFrame
    after_sample: pd.DataFrame
    rows_before: int
    rows_after: int

    def before_records(self) -> list[dict]:
        """Return the before-sample as a list of plain dict records (display-friendly)."""
        return self.before_sample.to_dict(orient="records")

    def after_records(self) -> list[dict]:
        """Return the after-sample as a list of plain dict records (display-friendly)."""
        return self.after_sample.to_dict(orient="records")


def _sample_rows(df: pd.DataFrame, n: int = _BEFORE_AFTER_SAMPLE_ROWS) -> pd.DataFrame:
    """Return a small real head-sample copy of ``df`` for a before/after example.

    The sample is an independent copy of the first ``n`` rows so later, in-place
    mutations of the pipeline DataFrames can never retroactively change a recorded
    example. When ``df`` is empty the copy is simply empty, which faithfully
    represents "no affected rows at this stage".
    """
    return df.head(n).copy()


def prepare(
    df: pd.DataFrame,
    zone_lookup: pd.DataFrame,
    scope: "ScopeConfig",
    *,
    regions: "list[str] | None" = None,
) -> "tuple[DemandSeries, HandlingLog, list[BeforeAfter]]":
    """Run the full preparation pipeline and record real before/after examples.

    This orchestrates the pure-logic preparation stages in order and returns the
    finished forecasting dataset together with the invalid-record
    :class:`HandlingLog` and a list of :class:`BeforeAfter` examples - one per
    transformation stage - so Requirement 4.2's "present a real before-and-after
    example of the affected data" holds for every step.

    Pipeline order (design: Data_Preparation_Pipeline):

    1. :func:`apply_validity_rules` - documented handling of invalid records
       (Requirement 4.4). Produces the :class:`HandlingLog` returned to the caller.
    2. :func:`map_zones_to_regions` - join ``PULocationID`` to its borough,
       materializing the Geographic_Grain ``region`` column.
    3. :func:`aggregate_demand` - count trips per ``(period, region)`` at the
       Time_Grain/Geographic_Grain (Requirement 4.1).
    4. :func:`fill_missing_periods` - zero-fill so every ``(period, region)`` in
       the Analysis_Window is present, missing buckets set to 0 (Requirement 4.3).
    5. :func:`add_lag_features` - add ``lag_{k}`` columns for each ``scope.lags``
       value (Requirement 4.6).

    For each stage a :class:`BeforeAfter` is recorded from the *actual* input and
    output DataFrames (small head-samples plus true row counts), so the returned
    examples are genuine slices of the real data rather than fabricated ones.

    Args:
        df: The validated raw trip records (with ``PULocationID`` and a pickup
            timestamp column).
        zone_lookup: The ``taxi_zone_lookup`` table mapping ``LocationID`` to
            ``Borough``.
        scope: The :class:`~src.config.ScopeConfig` supplying the Time_Grain,
            Analysis_Window, and ``lags``.
        regions: Optional explicit list of regions to zero-fill across (passed to
            :func:`fill_missing_periods`). When ``None`` (default), the distinct
            regions observed after mapping/aggregation are used.

    Returns:
        A ``(series, handling_log, before_after)`` tuple where ``series`` is the
        final :data:`DemandSeries` (zero-filled, with lag features), ``handling_log``
        is the :class:`HandlingLog` from the validity stage, and ``before_after``
        is the list of :class:`BeforeAfter` examples in pipeline order.
    """
    examples: list[BeforeAfter] = []

    # --- Stage 1: documented invalid-record handling (R4.4) ------------------
    before = _sample_rows(df)
    cleaned_df, handling_log = apply_validity_rules(df, scope)
    examples.append(
        BeforeAfter(
            name="apply_validity_rules",
            description=(
                "Applied the documented invalid-record handling rule "
                f"('{handling_log.rule}'): {handling_log.total_invalid_handled} "
                "invalid record(s) handled. "
                f"Breakdown by violation type: {handling_log.counts_by_type}."
            ),
            before_sample=before,
            after_sample=_sample_rows(cleaned_df),
            rows_before=int(len(df)),
            rows_after=int(len(cleaned_df)),
        )
    )

    # --- Stage 2: zone -> borough mapping (Geographic_Grain) -----------------
    before = _sample_rows(cleaned_df)
    mapped_df = map_zones_to_regions(cleaned_df, zone_lookup)
    examples.append(
        BeforeAfter(
            name="map_zones_to_regions",
            description=(
                "Joined PULocationID to its borough via the taxi_zone_lookup, "
                f"adding the '{REGION_COLUMN}' (Geographic_Grain) column."
            ),
            before_sample=before,
            after_sample=_sample_rows(mapped_df),
            rows_before=int(len(cleaned_df)),
            rows_after=int(len(mapped_df)),
        )
    )

    # --- Stage 3: aggregate to demand per (period, region) (R4.1) ------------
    before = _sample_rows(mapped_df)
    aggregated = aggregate_demand(mapped_df, scope)
    examples.append(
        BeforeAfter(
            name="aggregate_demand",
            description=(
                "Counted trips per (period, region) at the "
                f"'{scope.time_grain}' Time_Grain to build the long-format "
                "DemandSeries (Requirement 4.1)."
            ),
            before_sample=before,
            after_sample=_sample_rows(aggregated),
            rows_before=int(len(mapped_df)),
            rows_after=int(len(aggregated)),
        )
    )

    # --- Stage 4: zero-fill missing periods (R4.3) ---------------------------
    before = _sample_rows(aggregated)
    filled = fill_missing_periods(aggregated, scope, regions=regions)
    examples.append(
        BeforeAfter(
            name="fill_missing_periods",
            description=(
                "Zero-filled the series so every (period, region) in the "
                "Analysis_Window is present, with demand 0 where no trips "
                "occurred (Requirement 4.3)."
            ),
            before_sample=before,
            after_sample=_sample_rows(filled),
            rows_before=int(len(aggregated)),
            rows_after=int(len(filled)),
        )
    )

    # --- Stage 5: lag features (R4.6) ----------------------------------------
    before = _sample_rows(filled)
    final_series = add_lag_features(filled, list(scope.lags))
    lag_cols = [lag_column_name(k) for k in dict.fromkeys(scope.lags)]
    examples.append(
        BeforeAfter(
            name="add_lag_features",
            description=(
                f"Added per-region lag feature column(s) {lag_cols} where "
                "lag_k[t] = demand[t-k], NaN for the first k periods of each "
                "region (Requirement 4.6)."
            ),
            before_sample=before,
            after_sample=_sample_rows(final_series),
            rows_before=int(len(filled)),
            rows_after=int(len(final_series)),
        )
    )

    return final_series, handling_log, examples
