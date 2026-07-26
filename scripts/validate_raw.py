"""Raw-data validation runner - present real numbers before any modeling begins.

This is the user-facing entry point for Requirement 1: it loads the real NYC TLC
``fhvhv_2026-04.parquet`` file, builds a :class:`~src.validation.ValidationReport`,
and prints every finding with *real numbers* so the user can review the raw data
before modeling proceeds (Requirements 1.1-1.6, 1.8, 13.1).

The golden rule of the project is that only real public NYC TLC data is used and
every reported number is defensible. Accordingly, Kiro writes and verifies this
runner, but the USER executes it in their own terminal against the real (~1 GB)
Parquet file and reviews the reported numbers:

    python scripts/validate_raw.py
    python scripts/validate_raw.py data/fhvhv_2026-04.parquet

The reporting logic lives in :func:`format_validation_report`, which returns a
plain string. The notebook (task 12.1) imports that helper so the notebook and
the CLI present identical numbers - see the ``NOTEBOOK CELL`` snippet at the
bottom of this file.

Design references:
- Components and Interfaces -> Data_Validator (`src/validation.py`)
- Requirements 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.8, 13.1
"""

from __future__ import annotations

import argparse
import sys
from typing import Optional

# Make ``import src...`` work whether the script is run from the repo root
# (``python scripts/validate_raw.py``) or from inside ``scripts/``.
try:
    from src.config import ScopeConfig, default_scope
    from src.validation import (
        ValidationReport,
        build_validation_report,
        load_parquet,
    )
except ModuleNotFoundError:  # pragma: no cover - path bootstrap for direct runs
    import os

    _REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _REPO_ROOT not in sys.path:
        sys.path.insert(0, _REPO_ROOT)
    from src.config import ScopeConfig, default_scope
    from src.validation import (
        ValidationReport,
        build_validation_report,
        load_parquet,
    )

#: Default path to the already-downloaded FHVHV file, relative to the repo root.
DEFAULT_PARQUET_PATH = "data/fhvhv_2026-04.parquet"

_RULE = "=" * 78
_SUBRULE = "-" * 78


def _fmt_int(value: int) -> str:
    """Format an integer with thousands separators (e.g. ``19938536`` -> ``19,938,536``)."""
    return f"{value:,}"


def format_validation_report(
    report: ValidationReport, *, path: str, scope: Optional[ScopeConfig] = None
) -> str:
    """Render a :class:`ValidationReport` as a human-readable, well-formatted string.

    This is the single source of the presentation so the CLI runner and the
    notebook (task 12.1) show identical numbers. Every value printed here comes
    straight from the report built off the real DataFrame - nothing is fabricated
    (Requirements 1.2-1.6, 1.8, 13.1).

    Args:
        report: The fully-populated validation report to present.
        path: The Parquet path the report was built from (shown in the header).
        scope: Optional scope config, printed for context when supplied.

    Returns:
        A multi-line string ready to ``print``.
    """
    schema = report.schema
    row_count = schema.row_count
    lines: list[str] = []

    # --- Header --------------------------------------------------------------
    lines.append(_RULE)
    lines.append("RAW DATA VALIDATION REPORT")
    lines.append(f"File: {path}")
    if scope is not None:
        lines.append(
            f"Scope: time_grain={scope.time_grain}, "
            f"geographic_grain={scope.geographic_grain}, "
            f"window={scope.window_start} -> {scope.window_end}"
        )
    lines.append(_RULE)

    # --- 1. Schema: rows, columns, dtypes (R1.2) -----------------------------
    lines.append("")
    lines.append(f"[1] SCHEMA  -  {_fmt_int(row_count)} rows, "
                 f"{len(schema.column_names)} columns")
    lines.append(_SUBRULE)
    name_width = max((len(c) for c in schema.column_names), default=4)
    name_width = max(name_width, len("COLUMN"))
    lines.append(f"  {'COLUMN':<{name_width}}   DTYPE")
    for col in schema.column_names:
        lines.append(f"  {col:<{name_width}}   {schema.dtypes.get(col, '?')}")

    # --- 2. Null values per column (R1.3) ------------------------------------
    lines.append("")
    lines.append("[2] NULL VALUES per column")
    lines.append(_SUBRULE)
    lines.append(f"  {'COLUMN':<{name_width}}   {'NULLS':>14}   {'PERCENT':>9}")
    for col in schema.column_names:
        stat = report.nulls.get(col)
        if stat is None:
            continue
        lines.append(
            f"  {col:<{name_width}}   {_fmt_int(stat.count):>14}   "
            f"{stat.percentage:>8.4f}%"
        )

    # --- 3. Pickup date range (R1.4) -----------------------------------------
    lines.append("")
    lines.append("[3] PICKUP DATE RANGE")
    lines.append(_SUBRULE)
    if report.date_range is None:
        lines.append("  No usable pickup timestamp column found - date range unavailable.")
    else:
        lo, hi = report.date_range
        lines.append(f"  Earliest pickup: {lo}")
        lines.append(f"  Latest pickup:   {hi}")
        span_days = (hi - lo).days
        lines.append(f"  Span:            {_fmt_int(span_days)} days")

    # --- 4. Duplicate records (R1.5) -----------------------------------------
    lines.append("")
    lines.append("[4] DUPLICATE RECORDS")
    lines.append(_SUBRULE)
    dup = report.duplicate_count
    dup_pct = (dup / row_count * 100.0) if row_count else 0.0
    lines.append(f"  Duplicate rows: {_fmt_int(dup)} ({dup_pct:.4f}% of all rows)")

    # --- 5. Domain violations (R1.6) -----------------------------------------
    lines.append("")
    lines.append("[5] DOMAIN VIOLATIONS (out-of-domain records)")
    lines.append(_SUBRULE)
    if not report.domain_violations:
        lines.append("  None detected. All checked columns are within their valid domain.")
    else:
        for v in report.domain_violations:
            v_pct = (v.count / row_count * 100.0) if row_count else 0.0
            lines.append(
                f"  * {v.violation_type}: {_fmt_int(v.count)} records "
                f"({v_pct:.4f}%)"
            )
            if v.example is not None:
                lines.append(f"      example: {v.example}")
            else:
                lines.append("      example: <none captured>")

    lines.append("")
    lines.append(_RULE)
    lines.append("END OF REPORT - review these real numbers before modeling proceeds.")
    lines.append(_RULE)
    return "\n".join(lines)


def run(path: str = DEFAULT_PARQUET_PATH) -> ValidationReport:
    """Load ``path``, build the validation report, print it, and return it.

    This is the reusable core: the notebook can call ``run()`` directly, and the
    CLI ``main`` wraps it with argument parsing and error handling. Loading uses
    :func:`~src.validation.load_parquet`, which fails loudly and names the path
    when the file is missing or unreadable (Requirement 1.8).

    Args:
        path: Path to the FHVHV Parquet file.

    Returns:
        The :class:`ValidationReport` that was printed.
    """
    scope = default_scope()
    df = load_parquet(path)
    report = build_validation_report(df, scope)
    print(format_validation_report(report, path=path, scope=scope))
    return report


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entry point. Returns a process exit code (0 on success).

    Args:
        argv: Optional argument vector (defaults to ``sys.argv[1:]``).

    Returns:
        ``0`` when the report is produced, ``1`` when the file cannot be loaded.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Validate a raw NYC TLC FHVHV Parquet file and print all findings "
            "with real numbers (Requirement 1)."
        )
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=DEFAULT_PARQUET_PATH,
        help=f"Path to the Parquet file (default: {DEFAULT_PARQUET_PATH}).",
    )
    args = parser.parse_args(argv)

    try:
        run(args.path)
    except (FileNotFoundError, ValueError) as exc:
        # load_parquet raises these with the path already named (R1.8).
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# ---------------------------------------------------------------------------
# NOTEBOOK CELL (task 12.1 will place this in the notebook; kept minimal here).
# The notebook imports the helper so it shows the *same* real numbers as the CLI:
#
#     from scripts.validate_raw import run
#     report = run("data/fhvhv_2026-04.parquet")   # prints the full report
#
# ...or, if a DataFrame is already loaded in the notebook:
#
#     from src.config import default_scope
#     from src.validation import build_validation_report
#     from scripts.validate_raw import format_validation_report
#     report = build_validation_report(df, default_scope())
#     print(format_validation_report(report, path="data/fhvhv_2026-04.parquet"))
# ---------------------------------------------------------------------------
