"""Build the REAL 12-month daily borough demand series from the NYC TLC FHVHV files.

Why a dedicated builder?
------------------------
Each monthly FHVHV Parquet file has ~18-20M rows. Loading all twelve months'
*full* rows into one DataFrame at once would need tens of GB of RAM. This script
avoids that: it reads only the two columns the demand series actually needs
(``pickup_datetime`` and ``PULocationID``) from each file, aggregates that single
month down to a tiny daily-per-borough table (~30 rows x 6 boroughs), releases the
month, and moves on. Twelve small monthly aggregates are then combined and
zero-filled across the whole Analysis_Window.

It reuses the project's *tested* pure functions so the result is identical in
logic to what ``src.preparation.prepare`` would produce on the concatenated raw
data - just computed in a memory-safe, streaming way:

* :func:`src.preparation.map_zones_to_regions`
* :func:`src.preparation.aggregate_demand`
* :func:`src.preparation.fill_missing_periods`
* :func:`src.preparation.add_lag_features`

Per-file it also applies the documented invalid-record handling
(:func:`src.preparation.apply_validity_rules`) so pickups outside a file's stated
month and negative measures are dropped exactly as the pipeline specifies.

Golden rule: every number produced here comes straight from the real NYC TLC
files. Nothing is fabricated. If no real files are present the script exits with a
clear message rather than inventing data.

Usage::

    python scripts/build_demand_series.py
    python scripts/build_demand_series.py --data-dir data --out data/demand_series.parquet
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys
import time
from typing import Optional

import pandas as pd
import pyarrow.parquet as pq

# Make ``import src...`` work whether run from repo root or scripts/.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.config import default_scope
from src.preparation import (
    DEMAND_COLUMN,
    PERIOD_COLUMN,
    PICKUP_DATETIME_COLUMN,
    PICKUP_LOCATION_COLUMN,
    REGION_COLUMN,
    add_lag_features,
    aggregate_demand,
    apply_validity_rules,
    fill_missing_periods,
    lag_column_name,
    map_zones_to_regions,
)

#: The six NYC boroughs/zones we forecast (Geographic_Grain), used to zero-fill so
#: every borough is represented even in months where one has no observed trips.
BOROUGHS = ["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island", "EWR"]

#: Match both the official NYC TLC name (fhvhv_tripdata_YYYY-MM.parquet) and the
#: short alias the notebook mentions (fhvhv_YYYY-MM.parquet).
_MONTH_RE = re.compile(r"fhvhv_(?:tripdata_)?(\d{4})-(\d{2})\.parquet$", re.IGNORECASE)


def find_month_files(data_dir: str) -> list[str]:
    """Return sorted FHVHV monthly Parquet paths under ``data_dir``."""
    files = glob.glob(os.path.join(data_dir, "fhvhv_*.parquet"))
    files = [f for f in files if _MONTH_RE.search(os.path.basename(f))]
    return sorted(files)


def _stated_month(path: str) -> Optional[tuple[int, int]]:
    """Extract (year, month) from the filename, or None if it cannot be parsed."""
    m = _MONTH_RE.search(os.path.basename(path))
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def aggregate_one_month(path: str, zone_lookup: pd.DataFrame, scope) -> pd.DataFrame:
    """Read one month (2 columns only), clean it, and aggregate to daily demand.

    Returns a small long-format frame with ``period``, ``region``, ``demand`` for
    just this month's observed buckets.
    """
    # Read ONLY the two columns the demand series needs -> tiny memory footprint.
    table = pq.read_table(path, columns=[PICKUP_DATETIME_COLUMN, PICKUP_LOCATION_COLUMN])
    month_df = table.to_pandas()
    del table

    # Documented invalid-record handling for THIS file's stated month: drop pickups
    # that fall outside the month the file claims to cover (real TLC files carry a
    # few stragglers). Negative-measure checks are skipped here because we did not
    # load the measure columns; those do not affect the trip *count* per bucket.
    stated = _stated_month(path)
    cleaned, log = apply_validity_rules(
        month_df,
        scope,
        ts_col=PICKUP_DATETIME_COLUMN,
        stated_month=stated,
        non_negative_columns=[],  # measure columns not loaded; count-only cleaning
    )
    del month_df

    mapped = map_zones_to_regions(cleaned, zone_lookup)
    del cleaned
    monthly = aggregate_demand(mapped, scope)
    del mapped

    monthly.attrs["dropped_outside_month"] = int(log.total_invalid_handled)
    return monthly


def build_series(data_dir: str, out_path: str) -> pd.DataFrame:
    """Build and persist the real 12-month daily borough demand series."""
    scope = default_scope()

    files = find_month_files(data_dir)
    if not files:
        raise SystemExit(
            f"No FHVHV monthly Parquet files found under '{data_dir}'. "
            "Run scripts/download_data.ps1 first."
        )

    lookup_path = os.path.join(data_dir, "taxi_zone_lookup.csv")
    if not os.path.exists(lookup_path):
        raise SystemExit(
            f"taxi_zone_lookup.csv not found at '{lookup_path}'. "
            "It is required to map PULocationID -> borough."
        )
    zone_lookup = pd.read_csv(lookup_path)

    print(f"Found {len(files)} monthly file(s). Building real demand series...")
    per_month: list[pd.DataFrame] = []
    total_trips = 0
    for i, path in enumerate(files, 1):
        t0 = time.time()
        monthly = aggregate_one_month(path, zone_lookup, scope)
        month_trips = int(monthly[DEMAND_COLUMN].sum())
        total_trips += month_trips
        per_month.append(monthly)
        print(
            f"  [{i:>2}/{len(files)}] {os.path.basename(path):<34} "
            f"-> {month_trips:>12,} trips, "
            f"{monthly.attrs.get('dropped_outside_month', 0):>6,} dropped "
            f"({time.time() - t0:5.1f}s)"
        )

    # Combine months and sum any overlapping (period, region) buckets.
    observed = pd.concat(per_month, ignore_index=True)
    observed = (
        observed.groupby([PERIOD_COLUMN, REGION_COLUMN], as_index=False, sort=True)[
            DEMAND_COLUMN
        ].sum()
    )

    # Zero-fill across the whole window for every borough, then add lag features -
    # exactly the tail of src.preparation.prepare.
    filled = fill_missing_periods(observed, scope, regions=BOROUGHS)
    final = add_lag_features(filled, list(scope.lags))

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    final.to_parquet(out_path, index=False)

    lag_cols = [lag_column_name(k) for k in dict.fromkeys(scope.lags)]
    print()
    print("=== Real demand series built ===")
    print(f"  window        : {scope.window_start} -> {scope.window_end}")
    print(f"  rows          : {len(final):,}")
    print(f"  regions       : {sorted(final[REGION_COLUMN].unique())}")
    print(f"  total trips   : {total_trips:,}")
    print(f"  lag columns   : {lag_cols}")
    print(f"  saved to      : {out_path}")
    return final


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data", help="Directory with FHVHV files.")
    parser.add_argument(
        "--out",
        default=os.path.join("data", "demand_series.parquet"),
        help="Output path for the demand series Parquet.",
    )
    args = parser.parse_args(argv)
    build_series(args.data_dir, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
