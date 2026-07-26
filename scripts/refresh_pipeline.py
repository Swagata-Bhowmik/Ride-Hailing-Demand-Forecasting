"""Scheduled auto-refresh entry point - re-run the prepare -> forecast pipeline.

This runnable script is what the free-tier GitHub Actions workflow
(``.github/workflows/refresh.yml``) invokes on its cron schedule (Requirement
10.1, 10.3). It re-runs the core prepare -> forecast pipeline and records a clear
SUCCESS / FAILURE line in its logs so that both outcomes are preserved in the
Actions run logs (Requirements 10.1, 10.4).

Graceful degradation for CI (the "golden rule" stays intact)
------------------------------------------------------------
The real analysis runs on ~1 GB NYC TLC FHVHV Parquet files that are **not**
committed to the repository (``data/`` is git-ignored). A scheduled CI runner
therefore has no real data to read. Rather than fabricating results - which would
violate the project's golden rule - this script:

* **Uses real data when it is present.** If a FHVHV Parquet file is found under
  ``data/`` (or an explicit ``--data`` path is given), the pipeline loads and
  prepares that real data.
* **Degrades gracefully when it is absent.** When no real data is available (the
  normal case in CI), it builds a tiny, clearly-labelled *synthetic* trip-record
  fixture and runs the exact same pure-logic pipeline functions
  (:func:`~src.preparation.prepare` -> :func:`~src.evaluation.split_holdout` ->
  :func:`~src.models.base.train_all`). This exercises the real code paths end to
  end and proves the pipeline still runs, without inventing any real-world
  numbers. The log makes the synthetic mode explicit.

Only the lightweight Holt-Winters baseline is fitted here so the refresh stays
fast and dependency-light in a free-tier runner; the full candidate-model
comparison is run by the user against real data (see the notebook).

Exit code: ``0`` on success, ``1`` on failure. The non-zero exit is what makes a
failed run show up as a failed GitHub Actions job while the FAILURE log line
(Requirement 10.4) preserves the reason.

Usage::

    python scripts/refresh_pipeline.py                 # auto-detect data/, else synthetic
    python scripts/refresh_pipeline.py --data data/fhvhv_2026-04.parquet
    python scripts/refresh_pipeline.py --synthetic     # force synthetic (CI default)

Design references:
- Components and Interfaces -> Automation (`.github/workflows/refresh.yml`)
- Error Handling -> "Automated run fails" row
- Requirements 10.1, 10.2, 10.3, 10.4
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
import sys
from datetime import timedelta
from typing import Optional

# Make ``import src...`` work whether run from the repo root or from scripts/.
try:
    from src.config import ScopeConfig, default_scope
    from src.models.base import ExclusionRecord, TrainedModel, train_all
    from src.models.holt_winters import HoltWinters
    from src.preparation import (
        PICKUP_DATETIME_COLUMN,
        PICKUP_LOCATION_COLUMN,
        ZONE_LOOKUP_BOROUGH_COLUMN,
        ZONE_LOOKUP_ID_COLUMN,
        prepare,
    )
    from src.evaluation import split_holdout
    from src.validation import load_parquet
except ModuleNotFoundError:  # pragma: no cover - path bootstrap for direct runs
    _REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _REPO_ROOT not in sys.path:
        sys.path.insert(0, _REPO_ROOT)
    from src.config import ScopeConfig, default_scope
    from src.models.base import ExclusionRecord, TrainedModel, train_all
    from src.models.holt_winters import HoltWinters
    from src.preparation import (
        PICKUP_DATETIME_COLUMN,
        PICKUP_LOCATION_COLUMN,
        ZONE_LOOKUP_BOROUGH_COLUMN,
        ZONE_LOOKUP_ID_COLUMN,
        prepare,
    )
    from src.evaluation import split_holdout
    from src.validation import load_parquet

import pandas as pd

logger = logging.getLogger("refresh_pipeline")

#: Where committed-absent real Parquet data would live if present.
DEFAULT_DATA_DIR = "data"


def _configure_logging() -> None:
    """Send timestamped INFO logs to stdout so GitHub Actions captures them."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        stream=sys.stdout,
    )


def find_real_data(data_dir: str = DEFAULT_DATA_DIR) -> Optional[str]:
    """Return the path to a real FHVHV Parquet file under ``data_dir``, or None.

    ``data/`` is git-ignored, so in CI this normally returns ``None`` and the
    pipeline falls back to synthetic mode. Locally, when the user has downloaded
    the real ~1 GB files, the most recently named FHVHV file is used.
    """
    matches = sorted(glob.glob(os.path.join(data_dir, "fhvhv_*.parquet")))
    return matches[-1] if matches else None


def _synthetic_scope() -> ScopeConfig:
    """A scope with a compact holdout so the synthetic run stays quick in CI."""
    return default_scope().record_scope_change(
        "holdout_periods",
        7,
        rationale=(
            "Synthetic CI refresh: shrink the holdout to 7 days so the pipeline "
            "exercises split -> fit -> forecast quickly on the tiny fixture."
        ),
    )


def build_synthetic_inputs(
    scope: ScopeConfig,
) -> "tuple[pd.DataFrame, pd.DataFrame]":
    """Build a tiny, clearly-labelled synthetic ``(trips, zone_lookup)`` fixture.

    The fixture contains deterministic trip records spanning the start of the
    Analysis_Window for two boroughs, plus a matching two-row zone lookup. It is
    intentionally small and synthetic - it invents *no* real-world demand numbers,
    it only exercises the real pipeline code paths so CI can prove the refresh
    still runs (Requirement 10.1) when the git-ignored real data is absent.

    Args:
        scope: The scope whose ``window_start`` anchors the synthetic dates.

    Returns:
        A ``(trips, zone_lookup)`` tuple ready for :func:`~src.preparation.prepare`.
    """
    # ~8 weeks of daily records so Holt-Winters has a usable series after
    # zero-fill, kept anchored inside the Analysis_Window.
    num_days = 56
    location_to_borough = {100: "Manhattan", 200: "Brooklyn"}

    rows: list[dict] = []
    start = pd.Timestamp(scope.window_start)
    for day in range(num_days):
        ts = start + timedelta(days=day)
        for location_id, _ in location_to_borough.items():
            # Deterministic, obviously-synthetic trip counts (weekly ripple).
            trips = 5 + (day % 7) + (0 if location_id == 100 else 3)
            for _ in range(trips):
                rows.append(
                    {
                        PICKUP_DATETIME_COLUMN: ts + timedelta(hours=8),
                        PICKUP_LOCATION_COLUMN: location_id,
                    }
                )

    trips_df = pd.DataFrame(rows)
    zone_lookup = pd.DataFrame(
        {
            ZONE_LOOKUP_ID_COLUMN: list(location_to_borough.keys()),
            ZONE_LOOKUP_BOROUGH_COLUMN: list(location_to_borough.values()),
        }
    )
    return trips_df, zone_lookup


def load_real_inputs(path: str) -> "tuple[pd.DataFrame, pd.DataFrame]":
    """Load a real FHVHV Parquet file and a best-effort zone lookup.

    A committed ``data/taxi_zone_lookup.csv`` is used when present; otherwise a
    minimal identity-style borough lookup is derived from the observed pickup
    location ids so the pipeline can still run. Real data is preferred whenever it
    exists (Requirement 10.1) - this keeps the refresh honest.
    """
    trips_df = load_parquet(path)

    lookup_csv = os.path.join(DEFAULT_DATA_DIR, "taxi_zone_lookup.csv")
    if os.path.exists(lookup_csv):
        zone_lookup = pd.read_csv(lookup_csv)
    else:
        logger.warning(
            "No taxi_zone_lookup.csv found; deriving a minimal lookup from "
            "observed PULocationID values so the refresh can proceed."
        )
        ids = sorted(pd.unique(trips_df[PICKUP_LOCATION_COLUMN].dropna()))
        zone_lookup = pd.DataFrame(
            {
                ZONE_LOOKUP_ID_COLUMN: ids,
                ZONE_LOOKUP_BOROUGH_COLUMN: [f"Region-{i}" for i in ids],
            }
        )
    return trips_df, zone_lookup


def run_pipeline(*, data_path: Optional[str], force_synthetic: bool) -> None:
    """Run prepare -> split -> forecast, logging each stage. Raises on failure.

    Args:
        data_path: Explicit Parquet path, or ``None`` to auto-detect under ``data/``.
        force_synthetic: When ``True``, skip real data and use the synthetic fixture.

    Raises:
        Exception: Propagates any pipeline error so the caller can log FAILURE and
            exit non-zero (Requirement 10.4).
    """
    resolved = None if force_synthetic else (data_path or find_real_data())

    if resolved:
        logger.info("Real data found: %s - running pipeline on real data.", resolved)
        scope = default_scope()
        trips_df, zone_lookup = load_real_inputs(resolved)
        mode = "real"
    else:
        logger.info(
            "No real data available (data/ is git-ignored) - running in SYNTHETIC "
            "mode to verify the pipeline end to end without fabricating results."
        )
        scope = _synthetic_scope()
        trips_df, zone_lookup = build_synthetic_inputs(scope)
        mode = "synthetic"

    logger.info("Loaded %s trip records (mode=%s).", f"{len(trips_df):,}", mode)

    # --- Prepare -------------------------------------------------------------
    series, handling_log, before_after = prepare(trips_df, zone_lookup, scope)
    logger.info(
        "Prepared demand series: %s rows, %s region(s); %s invalid record(s) handled.",
        f"{len(series):,}",
        series["region"].nunique(),
        handling_log.total_invalid_handled,
    )
    logger.info("Recorded %d before/after transformation example(s).", len(before_after))

    # --- Split ---------------------------------------------------------------
    train, holdout = split_holdout(series, scope.holdout_periods)
    logger.info(
        "Holdout split: %s train rows, %s holdout rows (holdout_periods=%d).",
        f"{len(train):,}",
        f"{len(holdout):,}",
        scope.holdout_periods,
    )

    # --- Forecast (lightweight baseline keeps the refresh fast in CI) --------
    horizon = scope.holdout_periods
    results = train_all([HoltWinters()], train, scope, horizon)
    trained = [r for r in results if isinstance(r, TrainedModel)]
    excluded = [r for r in results if isinstance(r, ExclusionRecord)]

    for record in excluded:
        logger.warning("Model excluded: %s - %s", record.model_name, record.reason)

    if not trained:
        raise RuntimeError(
            "Forecast step produced no trained models; "
            f"all {len(results)} candidate(s) were excluded."
        )

    for model in trained:
        logger.info(
            "Forecast produced by %s over horizon=%d.",
            model.model_name,
            horizon,
        )


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entry point. Logs SUCCESS/FAILURE and returns a process exit code.

    Returns:
        ``0`` when the refresh completes, ``1`` when it fails (Requirement 10.4).
    """
    _configure_logging()

    parser = argparse.ArgumentParser(
        description=(
            "Scheduled auto-refresh: re-run the prepare -> forecast pipeline and "
            "record success/failure in the run logs (Requirement 10)."
        )
    )
    parser.add_argument(
        "--data",
        default=None,
        help="Explicit path to a FHVHV Parquet file (default: auto-detect under data/).",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Force synthetic mode (skip real data). This is the normal CI path.",
    )
    args = parser.parse_args(argv)

    logger.info("=== Auto-refresh pipeline starting ===")
    try:
        run_pipeline(data_path=args.data, force_synthetic=args.synthetic)
    except Exception as exc:  # noqa: BLE001 - top-level guard so FAILURE is always logged
        logger.exception("Refresh FAILED: %s", exc)
        logger.error("STATUS=FAILURE")
        logger.info("=== Auto-refresh pipeline finished (FAILURE) ===")
        return 1

    logger.info("STATUS=SUCCESS")
    logger.info("=== Auto-refresh pipeline finished (SUCCESS) ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
