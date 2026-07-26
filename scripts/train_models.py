"""Train the REAL candidate model set on the prepared demand series and persist
an honest results artifact the dashboards can load instead of illustrative demo
numbers.

Why this script exists
----------------------
Both dashboards (``dashboard/app.py`` and the standalone ``dashboard.html``) fall
back to a *clearly-labelled illustrative* model scoreboard until real model fits
exist. This script produces those real fits: it loads the prepared
``data/demand_series.parquet`` (247M+ real NYC TLC trips over the 12-month
window), trains every candidate model on the real training portion, scores them
on the reserved holdout, and writes a small JSON artifact
(``dashboard/model_results.json``) holding the actual holdout demand plus each
model's forecast. The dashboards read that artifact and recompute the metrics
through the *real* :func:`src.evaluation.build_model_results`, so nothing is
hand-typed and the numbers trace straight back to the raw data.

Evaluation level (documented, defensible choice)
------------------------------------------------
The candidate models emit forecasts at different shapes: the univariate/ML
families return a long per-``(period, region)`` grid, VAR returns a wide
period x region matrix, and Prophet (total) returns one system-wide series. To
score every model on the *same* fair target they are all reduced to
**total daily demand** - the system-wide trip count per day over the 30-day
holdout. This is the natural "how much demand tomorrow" business question, it is
identical for every model, and it matches the shape the dashboards already render
(a single actual line plus one forecast line per model).

Golden rule (honesty)
---------------------
Every number here is computed from the real prepared series. Models that cannot
train in this environment (for example LSTM/GRU when ``tensorflow`` is not
installed) are recorded with an explicit exclusion reason and still appear in the
scoreboard - never silently dropped.

Long-running note
-----------------
Fitting SARIMA/SARIMAX (one seasonal model per borough), VAR, Prophet and XGBoost
on the real series is a multi-minute job. Run it in your own terminal::

    python scripts/train_models.py

It prints per-model progress and timing, then writes the artifact.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

# Make ``import src...`` work whether run from the repo root or scripts/.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.config import default_scope
from src.evaluation import (
    build_model_results,
    comparison_table,
    error_metrics,
    select_carry_forward,
    split_holdout,
)
from src.models.base import ExclusionRecord, TrainedModel, train_all
from src.preparation import DEMAND_COLUMN, PERIOD_COLUMN, REGION_COLUMN

#: Where the small, committed real-results artifact is written (NOT under the
#: git-ignored data/ directory - this summary is a deliverable the deployed
#: dashboards load).
DEFAULT_OUT = os.path.join(_REPO_ROOT, "dashboard", "model_results.json")
DEFAULT_SERIES = os.path.join(_REPO_ROOT, "data", "demand_series.parquet")


def build_candidates() -> "list[tuple[str, Callable[[], Any]]]":
    """Return ``(label, factory)`` pairs for the full candidate model set.

    Factories (not instances) so ``train_all`` builds a fresh, unfitted model per
    run. Heavy/optional-dependency models are included too; if their dependency is
    missing the factory or fit raises and ``train_all`` records an exclusion.
    """

    def _holt_winters():
        from src.models.holt_winters import HoltWinters

        return HoltWinters()

    def _sarima():
        from src.models.sarima import Sarima

        return Sarima()

    def _sarimax():
        from src.models.sarima import Sarimax

        return Sarimax()

    def _var():
        from src.models.var import VarVarmax

        return VarVarmax()

    def _prophet():
        from src.models.prophet_model import ProphetModel

        return ProphetModel()  # total demand across regions

    def _xgboost():
        from src.models.xgboost_lags import XGBoostLags

        return XGBoostLags()

    def _lstm():
        from src.models.lstm_gru import LSTMModel

        return LSTMModel(epochs=50)

    def _gru():
        from src.models.lstm_gru import GRUModel

        return GRUModel(epochs=50)

    return [
        ("Holt-Winters", _holt_winters),
        ("SARIMA", _sarima),
        ("SARIMAX", _sarimax),
        ("VAR", _var),
        ("Prophet", _prophet),
        ("XGBoost", _xgboost),
        ("LSTM", _lstm),
        ("GRU", _gru),
    ]


def forecast_to_total_per_period(
    forecast: Any, holdout_period_index: pd.DatetimeIndex
) -> np.ndarray:
    """Reduce any model's holdout forecast to a total-per-period array.

    Handles the three forecast shapes the candidate models emit and returns one
    value per holdout period (system-wide daily total), aligned to
    ``holdout_period_index`` (ascending). Missing periods are filled with 0.

    * Long per-``(period, region)`` forecast (MultiIndex): sum the values within
      each period.
    * Wide VAR forecast (``DataFrame`` rows=periods, cols=regions): sum across
      regions per row.
    * Total series (1-D values indexed by period): used directly.
    """
    values = forecast.values
    index = forecast.index

    if isinstance(values, pd.DataFrame):
        # VAR wide: rows are periods, columns are regions -> row-sum is the total.
        per_period = values.sum(axis=1)
        per_period.index = pd.to_datetime(values.index)
    elif isinstance(index, pd.MultiIndex) and PERIOD_COLUMN in (index.names or []):
        periods = pd.to_datetime(index.get_level_values(PERIOD_COLUMN))
        tmp = pd.DataFrame({"period": periods, "v": np.asarray(values, dtype=float).ravel()})
        per_period = tmp.groupby("period")["v"].sum()
    else:
        arr = np.asarray(values, dtype=float).ravel()
        if index is not None and len(index) == len(arr):
            per_period = pd.Series(arr, index=pd.to_datetime(list(index)))
        else:
            # No usable index: assume already aligned to the holdout periods.
            per_period = pd.Series(arr, index=holdout_period_index[: len(arr)])

    aligned = per_period.reindex(holdout_period_index).fillna(0.0)
    return aligned.to_numpy(dtype=float)


def train_and_evaluate(series_path: str, out_path: str) -> dict:
    """Train the candidate set on the real series and write the results artifact."""
    scope = default_scope()

    if not os.path.exists(series_path):
        raise SystemExit(
            f"Prepared demand series not found at '{series_path}'. "
            "Build it first with scripts/stream_build_series.py."
        )

    series = pd.read_parquet(series_path)
    series[PERIOD_COLUMN] = pd.to_datetime(series[PERIOD_COLUMN])
    missing = {PERIOD_COLUMN, REGION_COLUMN, DEMAND_COLUMN} - set(series.columns)
    if missing:
        raise SystemExit(f"Series is missing required columns: {sorted(missing)}.")

    n_regions = series[REGION_COLUMN].nunique()
    train, holdout = split_holdout(series, scope.holdout_periods)
    horizon = holdout[PERIOD_COLUMN].nunique()

    # The holdout period axis (ascending) and the system-wide actual per period.
    holdout_period_index = pd.DatetimeIndex(
        np.sort(holdout[PERIOD_COLUMN].unique())
    )
    actual_total = (
        holdout.groupby(PERIOD_COLUMN)[DEMAND_COLUMN].sum().reindex(holdout_period_index)
    ).to_numpy(dtype=float)

    print(f"Series          : {len(series):,} rows, {n_regions} regions")
    print(f"Train periods   : {train[PERIOD_COLUMN].nunique()}")
    print(f"Holdout periods : {horizon} (system-wide total-demand evaluation)")
    print(f"Actual holdout total demand: {int(actual_total.sum()):,} trips\n")

    # Train each candidate in isolation, timing and reporting outcomes.
    candidates = build_candidates()
    train_results = []
    for label, factory in candidates:
        t0 = time.time()
        result = train_all([factory], train, scope, horizon)[0]
        dt = time.time() - t0
        if isinstance(result, TrainedModel):
            print(f"  [OK]  {label:<13} trained in {dt:6.1f}s")
        else:
            print(f"  [SKIP]{label:<13} excluded: {result.reason.splitlines()[0][:80]}")
        train_results.append((label, result))

    # Reduce every trained forecast to total-per-period; keep exclusions as-is.
    models_payload: list[dict] = []
    reduced_results = []
    for label, result in train_results:
        if isinstance(result, TrainedModel):
            total = forecast_to_total_per_period(result.forecast, holdout_period_index)
            models_payload.append({"name": label, "values": [float(v) for v in total]})
            # Rebuild a lightweight TrainedModel carrying the reduced total forecast
            # so build_model_results scores it against actual_total.
            from src.models.base import Forecast

            reduced_results.append(
                TrainedModel(
                    model_name=label,
                    forecaster=object(),
                    forecast=Forecast(
                        model_name=label, values=total, index=holdout_period_index
                    ),
                )
            )
        else:
            models_payload.append(
                {"name": label, "values": None, "excluded_reason": result.reason}
            )
            reduced_results.append(result)

    # Score with the real evaluation code and select the carry-forward set.
    results = build_model_results(reduced_results, actual_total)
    table = comparison_table(results)
    try:
        carry_forward, justification = select_carry_forward(
            table, return_justification=True
        )
    except Exception as exc:  # noqa: BLE001 - selection needs enough scored models
        carry_forward, justification = [], f"carry-forward selection unavailable: {exc}"

    artifact = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "evaluation_level": "total_daily_demand",
        "source": "data/demand_series.parquet (real NYC TLC FHVHV)",
        "window": f"{scope.window_start} -> {scope.window_end}",
        "holdout_periods": int(horizon),
        "n_regions": int(n_regions),
        "index": [d.strftime("%Y-%m-%d") for d in holdout_period_index],
        "actual": [float(v) for v in actual_total],
        "models": models_payload,
        "carry_forward": list(carry_forward),
        "carry_forward_justification": justification,
    }

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(artifact, fh, indent=2)

    # Print the honest scoreboard.
    print("\n=== Real model scoreboard (total daily demand, holdout) ===")
    show = table.sort_values("mae", na_position="last").reset_index(drop=True)
    with pd.option_context("display.max_columns", None, "display.width", 120):
        print(show.to_string(index=False))
    print(f"\nCarry forward: {carry_forward}")
    print(f"Saved artifact -> {out_path}")
    return artifact


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--series", default=DEFAULT_SERIES, help="Prepared demand series parquet.")
    parser.add_argument("--out", default=DEFAULT_OUT, help="Output JSON artifact path.")
    args = parser.parse_args(argv)
    train_and_evaluate(args.series, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
