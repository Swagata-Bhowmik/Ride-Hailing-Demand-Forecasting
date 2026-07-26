"""Train ONLY the deep-learning models (LSTM, GRU) on the real series and merge
their real results into the existing ``dashboard/model_results.json``.

Why a separate merge script?
----------------------------
``scripts/train_models.py`` trains the whole candidate set at once. But TensorFlow
has no wheels for Python 3.14 (this project's default interpreter), so LSTM/GRU are
recorded there as *excluded*. To add them without losing the already-real results
for the other six models - and without needing Prophet/statsmodels/XGBoost
reinstalled in a second environment - this script:

1. trains just LSTM and GRU on the real prepared series (in a Python 3.11 venv
   where ``tensorflow-cpu`` is installed);
2. reduces their holdout forecasts to the same system-wide total-daily-demand
   level the artifact uses;
3. replaces the two LSTM/GRU entries in the existing artifact with the real
   values;
4. recomputes the carry-forward selection over the *full* updated scoreboard via
   the real ``src.evaluation`` code;
5. writes the artifact back.

Run it with the TensorFlow-enabled interpreter, e.g.::

    .\\.venv-tf\\Scripts\\python.exe scripts/add_deep_learning.py

Golden rule: every number comes from the real series; nothing is fabricated.
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.config import default_scope
from src.evaluation import (
    build_model_results,
    comparison_table,
    select_carry_forward,
    split_holdout,
)
from src.models.base import TrainedModel, train_all
from src.preparation import DEMAND_COLUMN, PERIOD_COLUMN, REGION_COLUMN
from scripts.train_models import DEFAULT_OUT, DEFAULT_SERIES, forecast_to_total_per_period

#: The deep-learning models added by this script (name -> factory).
DEEP_MODELS = ("LSTM", "GRU")


def main() -> int:
    scope = default_scope()

    if not os.path.exists(DEFAULT_SERIES):
        raise SystemExit(f"Prepared series not found: {DEFAULT_SERIES}")
    if not os.path.exists(DEFAULT_OUT):
        raise SystemExit(
            f"Existing results artifact not found: {DEFAULT_OUT}. "
            "Run scripts/train_models.py first."
        )

    series = pd.read_parquet(DEFAULT_SERIES)
    series[PERIOD_COLUMN] = pd.to_datetime(series[PERIOD_COLUMN])
    train, holdout = split_holdout(series, scope.holdout_periods)
    horizon = holdout[PERIOD_COLUMN].nunique()

    holdout_period_index = pd.DatetimeIndex(np.sort(holdout[PERIOD_COLUMN].unique()))
    actual_total = (
        holdout.groupby(PERIOD_COLUMN)[DEMAND_COLUMN].sum().reindex(holdout_period_index)
    ).to_numpy(dtype=float)

    def _lstm():
        from src.models.lstm_gru import LSTMModel

        return LSTMModel(epochs=100)

    def _gru():
        from src.models.lstm_gru import GRUModel

        return GRUModel(epochs=100)

    factories = {"LSTM": _lstm, "GRU": _gru}

    print(f"Training deep-learning models on the real series ({len(series):,} rows)...")
    trained: dict[str, dict] = {}
    for name in DEEP_MODELS:
        t0 = time.time()
        result = train_all([factories[name]], train, scope, horizon)[0]
        dt = time.time() - t0
        if isinstance(result, TrainedModel):
            total = forecast_to_total_per_period(result.forecast, holdout_period_index)
            trained[name] = {"name": name, "values": [float(v) for v in total]}
            print(f"  [OK]  {name} trained in {dt:5.1f}s")
        else:
            trained[name] = {
                "name": name,
                "values": None,
                "excluded_reason": result.reason,
            }
            print(f"  [SKIP]{name} excluded: {result.reason.splitlines()[0][:80]}")

    # --- Merge into the existing artifact ------------------------------------
    with open(DEFAULT_OUT, "r", encoding="utf-8") as fh:
        artifact = json.load(fh)

    by_name = {m["name"]: m for m in artifact["models"]}
    for name in DEEP_MODELS:
        by_name[name] = trained[name]
    artifact["models"] = list(by_name.values())

    # --- Recompute carry-forward over the FULL updated scoreboard ------------
    from src.models.base import ExclusionRecord, Forecast

    actual = np.asarray(artifact["actual"], dtype=float)
    index = pd.DatetimeIndex(pd.to_datetime(artifact["index"]))
    train_results = []
    for model in artifact["models"]:
        if model.get("values") is None:
            train_results.append(
                ExclusionRecord(
                    model_name=model["name"],
                    reason=model.get("excluded_reason", "excluded"),
                )
            )
        else:
            train_results.append(
                TrainedModel(
                    model_name=model["name"],
                    forecaster=object(),
                    forecast=Forecast(
                        model_name=model["name"],
                        values=np.asarray(model["values"], dtype=float),
                        index=index,
                    ),
                )
            )
    results = build_model_results(train_results, actual)
    table = comparison_table(results)
    carry_forward, justification = select_carry_forward(table, return_justification=True)

    artifact["carry_forward"] = list(carry_forward)
    artifact["carry_forward_justification"] = justification
    artifact["generated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")

    with open(DEFAULT_OUT, "w", encoding="utf-8") as fh:
        json.dump(artifact, fh, indent=2)

    print("\n=== Updated scoreboard (total daily demand, holdout) ===")
    show = table.sort_values("mae", na_position="last").reset_index(drop=True)
    with pd.option_context("display.max_columns", None, "display.width", 120):
        print(show[["model_name", "mae", "rmse", "mape", "excluded"]].to_string(index=False))
    print(f"\nCarry forward: {carry_forward}")
    print(f"Updated artifact -> {DEFAULT_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
