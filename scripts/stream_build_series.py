"""Disk-safe STREAMING builder for the real 12-month FHVHV demand series.

The C: drive here cannot hold all twelve ~480 MB monthly files at once. This
script therefore processes months one at a time and never keeps more than a
single raw file on disk:

    for each month:
        if a tiny cached partial aggregate already exists -> skip
        else:
            ensure the raw .parquet is present (download it if missing)
            aggregate that ONE month down to a small daily-per-borough table
            save the partial aggregate (~a few KB)
            DELETE the ~480 MB raw file to free space

    finally: combine all twelve tiny partials -> zero-fill + lag features
             -> data/demand_series.parquet

It is fully resumable: re-running picks up wherever it left off because finished
months already have a partial and are skipped, and their raw files are gone.

Golden rule: every number is computed straight from the real NYC TLC files.
Nothing is fabricated. Missing months would be zero-filled, so the script refuses
to write the final series unless ALL twelve months have real partials.
"""

from __future__ import annotations

import os
import sys
import time
import urllib.request

import pandas as pd

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.config import default_scope
from src.preparation import (
    DEMAND_COLUMN,
    PERIOD_COLUMN,
    REGION_COLUMN,
    add_lag_features,
    fill_missing_periods,
    lag_column_name,
)
from scripts.build_demand_series import BOROUGHS, aggregate_one_month

DATA_DIR = os.path.join(_REPO_ROOT, "data")
PARTIALS_DIR = os.path.join(DATA_DIR, "_partials")
OUT_PATH = os.path.join(DATA_DIR, "demand_series.parquet")
URL_BASE = "https://d37ci6vzurychx.cloudfront.net/trip-data"

MONTHS = [
    "2025-05", "2025-06", "2025-07", "2025-08", "2025-09", "2025-10",
    "2025-11", "2025-12", "2026-01", "2026-02", "2026-03", "2026-04",
]


def raw_path(month: str) -> str:
    return os.path.join(DATA_DIR, f"fhvhv_tripdata_{month}.parquet")


def partial_path(month: str) -> str:
    return os.path.join(PARTIALS_DIR, f"{month}.parquet")


def download_month(month: str, dest: str) -> None:
    url = f"{URL_BASE}/fhvhv_tripdata_{month}.parquet"
    tmp = dest + ".part"
    print(f"      downloading {url} ...", flush=True)
    t0 = time.time()
    urllib.request.urlretrieve(url, tmp)
    os.replace(tmp, dest)
    print(
        f"      downloaded {os.path.getsize(dest)/1024/1024:.1f} MB "
        f"({time.time()-t0:.0f}s)",
        flush=True,
    )


def main() -> int:
    scope = default_scope()
    os.makedirs(PARTIALS_DIR, exist_ok=True)

    lookup_path = os.path.join(DATA_DIR, "taxi_zone_lookup.csv")
    if not os.path.exists(lookup_path):
        raise SystemExit(f"Missing {lookup_path}")
    zone_lookup = pd.read_csv(lookup_path)

    for i, month in enumerate(MONTHS, 1):
        ppath = partial_path(month)
        if os.path.exists(ppath):
            print(f"[{i:>2}/12] {month}: partial exists -> skip", flush=True)
            continue

        print(f"[{i:>2}/12] {month}: building partial", flush=True)
        rpath = raw_path(month)
        if not os.path.exists(rpath):
            download_month(month, rpath)

        monthly = aggregate_one_month(rpath, zone_lookup, scope)
        monthly.to_parquet(ppath, index=False)
        trips = int(monthly[DEMAND_COLUMN].sum())
        print(f"      -> {trips:,} trips aggregated, partial saved", flush=True)

        # Free the ~480 MB raw file immediately.
        os.remove(rpath)
        print(f"      -> deleted raw file, freed disk", flush=True)

    # --- Combine all twelve partials -----------------------------------------
    missing = [m for m in MONTHS if not os.path.exists(partial_path(m))]
    if missing:
        raise SystemExit(f"Refusing to write series; missing months: {missing}")

    frames = [pd.read_parquet(partial_path(m)) for m in MONTHS]
    observed = pd.concat(frames, ignore_index=True)
    observed = observed.groupby(
        [PERIOD_COLUMN, REGION_COLUMN], as_index=False, sort=True
    )[DEMAND_COLUMN].sum()

    filled = fill_missing_periods(observed, scope, regions=BOROUGHS)
    final = add_lag_features(filled, list(scope.lags))
    final.to_parquet(OUT_PATH, index=False)

    total = int(observed[DEMAND_COLUMN].sum())
    lag_cols = [lag_column_name(k) for k in dict.fromkeys(scope.lags)]
    print()
    print("=== REAL demand series built (streaming) ===", flush=True)
    print(f"  window      : {scope.window_start} -> {scope.window_end}")
    print(f"  rows        : {len(final):,}")
    print(f"  regions     : {sorted(final[REGION_COLUMN].unique())}")
    print(f"  total trips : {total:,}")
    print(f"  lag columns : {lag_cols}")
    print(f"  saved to    : {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
