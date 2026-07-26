"""LSTM / GRU deep-learning forecasters (Requirement 5.6).

These are the deep-learning candidates in the Candidate_Model_Set. Both treat
forecasting as a supervised **sequence** problem: a small recurrent network learns
a mapping from a fixed-length window of recent demand ``[demand[t-w], ...,
demand[t-1]]`` to the next value ``demand[t]``. :class:`LSTMModel` uses an LSTM
recurrent layer; :class:`GRUModel` is identical except it swaps in a GRU layer -
the only difference between the two - so they share the common base class
:class:`_RecurrentForecaster`.

Sliding-window supervised sequences
-----------------------------------
The project's :class:`~src.preparation.DemandSeries` is long-format and (at the
Borough Geographic_Grain) carries several parallel regional series. Within each
region the demand is sorted by period and cut into overlapping windows of length
``window``; each window is one training sample whose target is the demand in the
period immediately after it. Windows from every region are stacked into a single
training set and one compact network is trained across all of them, so the model
shares statistical strength across boroughs while staying tiny. Demand is
standardized (zero mean, unit variance over the training data) before training so
the network optimizes stably; predictions are de-standardized and clipped back to
the non-negative demand domain.

Multi-step (recursive) forecasting strategy
--------------------------------------------
A one-step sequence model is rolled forward **recursively** to cover the whole
Holdout_Set: for each region the last ``window`` observed training values seed the
window, the network predicts the next period, that prediction is appended, the
window slides forward, and the process repeats ``horizon`` times. Because
``predict(horizon)`` is called with ``horizon`` equal to the Holdout_Set length,
the forecast covers the same holdout as every other model (Requirement 5.7), and
its ``index`` is the ``(period, region)`` grid of that holdout sorted period-then
-region (matching the preparation pipeline's ordering) so the Evaluation_Framework
can align it to the actuals.

Graceful optional dependency (Requirement 5.8)
----------------------------------------------
``tensorflow`` is a heavyweight optional dependency and its training is expensive
(the full fit is run by the USER; Kiro verifies wiring on a small fixture). It is
imported **lazily inside** :meth:`_RecurrentForecaster.fit`, never at module load,
so importing this module never requires tensorflow and the rest of ``src.models``
keeps working without it. A missing/broken install surfaces as a clear exception
from ``fit`` - which :func:`~src.models.base.train_all` turns into an
:class:`~src.models.base.ExclusionRecord` rather than aborting the whole
comparison.

Design references:
- Components and Interfaces -> Forecasting_System (`src/models/`), LSTMModel/GRUModel
- Data Models -> Forecast (aligned to the Holdout_Set)
- Error Handling -> per-model exclusion recorded, one failure never aborts others
- Requirements 5.6, 5.7, 5.8
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from src.models.base import Forecast
from src.preparation import DEMAND_COLUMN, PERIOD_COLUMN, REGION_COLUMN

if TYPE_CHECKING:  # pragma: no cover - imported only for type checking
    from src.config import ScopeConfig
    from src.preparation import DemandSeries

#: Time_Grain -> pandas date-offset alias used to generate the forecast periods.
#: Mirrors ``src.preparation._TIME_GRAIN_FREQ``; only the daily grain (the project
#: default) is supported for now.
_TIME_GRAIN_FREQ: dict[str, str] = {"daily": "D"}

#: Default sliding-window length (lookback) at the daily Time_Grain. One week of
#: history captures the dominant weekly demand rhythm; kept small so the network
#: and the training set stay tiny.
_DEFAULT_WINDOW = 7


class _RecurrentForecaster:
    """Shared base for the LSTM/GRU sequence forecasters (Requirement 5.6).

    Subclasses set :attr:`_layer_name` to the Keras recurrent layer they use
    (``"LSTM"`` or ``"GRU"``); everything else - windowing, standardization,
    training, and recursive multi-step forecasting - is identical and lives here.
    Conforms structurally to :class:`~src.models.base.Forecaster` (``name``,
    ``fit``, ``predict``).

    Attributes:
        name: Model name used as the key in ``TrainedModel``/``ExclusionRecord``
            and the evaluation comparison table.
    """

    #: Keras recurrent layer class name the subclass builds its network from.
    #: Overridden by :class:`LSTMModel` / :class:`GRUModel`.
    _layer_name: str = "LSTM"

    def __init__(
        self,
        name: str | None = None,
        *,
        window: int = _DEFAULT_WINDOW,
        units: int = 16,
        epochs: int = 200,
        batch_size: int = 32,
        random_state: int = 0,
    ) -> None:
        """Create an (unfitted) recurrent forecaster.

        Constructing this object never imports or requires ``tensorflow`` - the
        dependency is only needed at :meth:`fit`.

        Args:
            name: Human-readable model name. Defaults to the recurrent layer name
                (``"LSTM"`` / ``"GRU"``).
            window: Sliding-window lookback length (number of prior periods fed to
                the network). Defaults to 7 (one weekly cycle at the daily grain).
            units: Number of recurrent units in the (single) hidden layer. Kept
                small so the network is tiny and trains fast on fixtures.
            epochs: Training epochs. The full fit is run by the USER; a smaller
                value is fine for fixture smoke checks.
            batch_size: Mini-batch size for training.
            random_state: Seed applied to Python/NumPy/TensorFlow for reproducible
                fits on fixtures.
        """
        self.name = name if name is not None else self._layer_name
        self._window = int(window)
        self._units = int(units)
        self._epochs = int(epochs)
        self._batch_size = int(batch_size)
        self._random_state = int(random_state)

        # Populated by fit():
        self.model: Any = None
        self._regions: list[str] = []
        self._freq: str | None = None
        self._last_period: pd.Timestamp | None = None
        self._effective_window: int = self._window
        self._mean: float = 0.0
        self._std: float = 1.0
        # region -> ascending array of the region's (standardized) training demand.
        self._region_history: dict[str, np.ndarray] = {}

    # -- sequence construction -------------------------------------------------

    @staticmethod
    def _make_windows(
        values: np.ndarray, window: int
    ) -> "tuple[list[np.ndarray], list[float]]":
        """Cut one region's series into ``(window -> next value)`` samples.

        Args:
            values: The region's demand, ascending by period (already scaled).
            window: The lookback length.

        Returns:
            A ``(X_list, y_list)`` pair; empty when the series is too short to
            yield even one window.
        """
        xs: list[np.ndarray] = []
        ys: list[float] = []
        for start in range(0, len(values) - window):
            xs.append(values[start : start + window])
            ys.append(float(values[start + window]))
        return xs, ys

    def _build_network(self, tf_module: Any) -> Any:
        """Build and compile the tiny recurrent network.

        Uses the subclass's recurrent layer (:attr:`_layer_name`) followed by a
        single Dense output unit. Kept deliberately small (one recurrent layer)
        so it trains quickly and reproducibly on fixtures.

        Args:
            tf_module: The imported ``tensorflow`` module (passed in so the import
                stays lazy and lives only in :meth:`fit`).

        Returns:
            A compiled ``keras`` model expecting input shape
            ``(samples, window, 1)`` and predicting one value.
        """
        keras = tf_module.keras
        recurrent_layer = getattr(keras.layers, self._layer_name)
        model = keras.Sequential(
            [
                keras.layers.Input(shape=(self._effective_window, 1)),
                recurrent_layer(self._units, activation="tanh"),
                keras.layers.Dense(1),
            ]
        )
        model.compile(optimizer="adam", loss="mse")
        return model

    # -- Forecaster interface --------------------------------------------------

    def fit(self, train: "DemandSeries", scope: "ScopeConfig") -> None:
        """Train the recurrent network on windowed sequences from every region.

        ``tensorflow`` is imported here (lazily) and seeded for reproducibility.
        Demand is standardized over the training data, each region is cut into
        sliding windows, all windows are stacked, and one compact network is
        trained across them.

        Args:
            train: The training :class:`~src.preparation.DemandSeries` (long format
                with ``period``/``region``/``demand`` columns, the periods before
                the Holdout_Set).
            scope: The project :class:`~src.config.ScopeConfig` supplying the
                Time_Grain (which fixes the forecast frequency).

        Raises:
            RuntimeError: If ``tensorflow`` is not importable (so ``train_all``
                records an exclusion instead of crashing - Requirement 5.8).
            KeyError: If ``train`` is missing a required column.
            ValueError: If the Time_Grain is unsupported, the series is empty, or
                no region has enough history to form a single training window.
        """
        # Lazy import: keeps module import working without tensorflow, and turns a
        # missing/broken install into a fit-time error that train_all excludes.
        try:  # pragma: no cover - import path depends on the runtime environment
            import tensorflow as tf  # noqa: PLC0415 - deliberately lazy (R5.8)
        except Exception as exc:  # noqa: BLE001 - any import failure must degrade gracefully
            raise RuntimeError(
                "tensorflow is not available, so the "
                f"{self.name} model is excluded: {exc!r}. "
                "Install 'tensorflow' to enable it."
            ) from exc

        for col in (PERIOD_COLUMN, REGION_COLUMN, DEMAND_COLUMN):
            if col not in train.columns:
                raise KeyError(
                    f"Column '{col}' not found in train series. "
                    f"Available columns: {list(train.columns)}."
                )

        grain = scope.time_grain.lower()
        if grain not in _TIME_GRAIN_FREQ:
            raise ValueError(
                f"Unsupported time_grain '{scope.time_grain}'. "
                f"Supported grains: {sorted(_TIME_GRAIN_FREQ)}."
            )
        self._freq = _TIME_GRAIN_FREQ[grain]

        # Seed Python/NumPy/TensorFlow together so fixture fits are reproducible.
        try:  # set_random_seed exists in TF >= 2.7 and seeds all three RNGs.
            tf.keras.utils.set_random_seed(self._random_state)
        except Exception:  # noqa: BLE001 - fall back to seeding individually
            np.random.seed(self._random_state)
            tf.random.set_seed(self._random_state)

        work = train[[PERIOD_COLUMN, REGION_COLUMN, DEMAND_COLUMN]].copy()
        work[PERIOD_COLUMN] = pd.to_datetime(work[PERIOD_COLUMN])
        work = work.sort_values([PERIOD_COLUMN, REGION_COLUMN], kind="stable")

        if work.empty:
            raise ValueError(f"Cannot fit {self.name} on an empty training series.")

        self._regions = sorted(work[REGION_COLUMN].dropna().unique().tolist())
        self._last_period = work[PERIOD_COLUMN].max()

        # Standardize demand over all training data so the network trains stably.
        demand_all = work[DEMAND_COLUMN].to_numpy(dtype="float64")
        self._mean = float(np.mean(demand_all))
        std = float(np.std(demand_all))
        self._std = std if std > 1e-12 else 1.0

        # Per-region ascending (scaled) demand, retained to seed recursive predict.
        self._region_history = {}
        for region in self._regions:
            region_demand = (
                work.loc[work[REGION_COLUMN] == region]
                .sort_values(PERIOD_COLUMN, kind="stable")[DEMAND_COLUMN]
                .to_numpy(dtype="float64")
            )
            self._region_history[region] = (region_demand - self._mean) / self._std

        # Shrink the window if no region is long enough for even one sample, so
        # tiny fixtures still fit (need at least window+1 points somewhere).
        longest = max((len(v) for v in self._region_history.values()), default=0)
        if longest < 2:
            raise ValueError(
                f"Cannot fit {self.name}: no region has enough history "
                "(need at least two periods)."
            )
        self._effective_window = min(self._window, longest - 1)

        # Build the stacked training set from every region's windows.
        x_list: list[np.ndarray] = []
        y_list: list[float] = []
        for scaled in self._region_history.values():
            xs, ys = self._make_windows(scaled, self._effective_window)
            x_list.extend(xs)
            y_list.extend(ys)

        if not x_list:
            raise ValueError(
                f"Cannot fit {self.name}: not enough history to build any "
                f"training window of length {self._effective_window}."
            )

        X = np.asarray(x_list, dtype="float64").reshape(
            -1, self._effective_window, 1
        )
        y = np.asarray(y_list, dtype="float64")

        self.model = self._build_network(tf)
        self.model.fit(
            X,
            y,
            epochs=self._epochs,
            batch_size=self._batch_size,
            verbose=0,
            shuffle=False,
        )

    def predict(self, horizon: int) -> Forecast:
        """Recursively forecast ``horizon`` periods over the Holdout_Set.

        For each region the last ``window`` (scaled) training values seed the
        window; the network predicts the next period, the prediction is appended,
        the window slides, and this repeats ``horizon`` times. Predictions are
        de-standardized and clipped to the non-negative demand domain. The returned
        :class:`Forecast` carries a ``(period, region)`` MultiIndex covering the
        holdout grid, sorted by period then region (matching the preparation
        pipeline's ordering) so it aligns to the actual holdout values.

        Args:
            horizon: Number of periods to forecast (the Holdout_Set length, R5.7).

        Returns:
            A :class:`Forecast` with one value per ``(period, region)`` holdout
            cell and a matching MultiIndex.

        Raises:
            RuntimeError: If called before :meth:`fit`.
            ValueError: If ``horizon`` is not a positive integer.
        """
        if self.model is None or self._last_period is None or self._freq is None:
            raise RuntimeError(f"{self.name}.predict called before fit.")
        if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon <= 0:
            raise ValueError(f"horizon must be a positive integer; got {horizon!r}.")

        window = self._effective_window
        future_periods = pd.date_range(
            start=self._last_period,
            periods=horizon + 1,
            freq=self._freq,
        )[1:]

        frames = []
        for region in self._regions:
            scaled_history = list(self._region_history.get(region, np.array([])))
            # Left-pad short histories with the earliest value so a full window is
            # always available to seed the recursion.
            if len(scaled_history) < window:
                pad_value = scaled_history[0] if scaled_history else 0.0
                scaled_history = [pad_value] * (window - len(scaled_history)) + scaled_history

            preds_scaled: list[float] = []
            for _ in range(horizon):
                window_values = np.asarray(
                    scaled_history[-window:], dtype="float64"
                ).reshape(1, window, 1)
                next_scaled = float(
                    np.asarray(self.model.predict(window_values, verbose=0)).reshape(-1)[0]
                )
                preds_scaled.append(next_scaled)
                scaled_history.append(next_scaled)

            # De-standardize and clip to the non-negative demand domain.
            preds = np.clip(
                np.asarray(preds_scaled, dtype="float64") * self._std + self._mean,
                0.0,
                None,
            )
            frames.append(
                pd.DataFrame(
                    {
                        PERIOD_COLUMN: future_periods,
                        REGION_COLUMN: region,
                        DEMAND_COLUMN: preds,
                    }
                )
            )

        combined = pd.concat(frames, ignore_index=True)
        combined = combined.sort_values(
            [PERIOD_COLUMN, REGION_COLUMN], kind="stable"
        ).reset_index(drop=True)

        index = pd.MultiIndex.from_arrays(
            [combined[PERIOD_COLUMN], combined[REGION_COLUMN]],
            names=[PERIOD_COLUMN, REGION_COLUMN],
        )
        return Forecast(
            model_name=self.name,
            values=combined[DEMAND_COLUMN].to_numpy(dtype="float64"),
            index=index,
        )


class LSTMModel(_RecurrentForecaster):
    """LSTM sequence forecaster (Requirement 5.6).

    Uses a Keras ``LSTM`` recurrent layer. All behavior is inherited from
    :class:`_RecurrentForecaster`; only the recurrent layer type differs from
    :class:`GRUModel`.
    """

    _layer_name = "LSTM"

    def __init__(self, name: str = "LSTM", **kwargs: Any) -> None:
        super().__init__(name=name, **kwargs)


class GRUModel(_RecurrentForecaster):
    """GRU sequence forecaster (Requirement 5.6).

    Identical to :class:`LSTMModel` except it uses a Keras ``GRU`` recurrent layer
    instead of an LSTM one.
    """

    _layer_name = "GRU"

    def __init__(self, name: str = "GRU", **kwargs: Any) -> None:
        super().__init__(name=name, **kwargs)
