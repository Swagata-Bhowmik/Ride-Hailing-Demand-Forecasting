"""Forecasting_System base interface and training orchestration (Requirement 5).

Every candidate model family (Holt-Winters, SARIMA/SARIMAX, VAR/VARMAX, Prophet,
XGBoost, LSTM/GRU) implements the same small :class:`Forecaster` interface so the
Evaluation_Framework can train and score them uniformly. This module defines that
interface plus the lightweight result structures the rest of the pipeline passes
around, and the :func:`train_all` orchestrator.

Two design guarantees are realized here:

* **All models forecast over the same Holdout_Set (R5.7).** ``train_all`` calls
  every candidate's ``predict(horizon)`` with the *same* horizon, and each
  :class:`Forecast` carries the holdout ``index`` it aligns to, so downstream
  evaluation compares like-for-like.
* **One failure never aborts the others (R5.8).** ``train_all`` wraps each
  candidate's ``fit``/``predict`` in a try/except. A candidate that cannot be
  trained on the prepared data (for example VAR on a single region, or
  insufficient history) yields an :class:`ExclusionRecord` capturing the reason,
  and training continues with the next candidate.

This module is the *interface only* (task 7.1). The concrete model families are
implemented in later tasks (7.2-7.7) and integration-tested in 7.8.

Design references:
- Components and Interfaces -> Forecasting_System (`src/models/`)
- Data Models -> Forecast, TrainedModel, ExclusionRecord
- Error Handling -> per-model exclusion recorded, one failure never aborts others
- Requirements 5.7, 5.8
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Iterable, Protocol, Union, runtime_checkable

if TYPE_CHECKING:  # pragma: no cover - imported only for type checking
    from src.config import ScopeConfig
    from src.preparation import DemandSeries


@runtime_checkable
class Forecaster(Protocol):
    """Common interface every candidate model family implements (Requirement 5).

    A ``Forecaster`` is a stateful object: :meth:`fit` trains it on the training
    portion of the :class:`~src.preparation.DemandSeries`, then :meth:`predict`
    produces a :class:`Forecast` over the Holdout_Set. Keeping the surface this
    small lets the Evaluation_Framework treat Holt-Winters, SARIMA, VAR, Prophet,
    XGBoost and the deep-learning models identically.

    The protocol is ``runtime_checkable`` so ``isinstance(obj, Forecaster)`` can be
    used as a light structural sanity check, but callers normally rely on duck
    typing - any object exposing ``name``, ``fit`` and ``predict`` qualifies.

    Attributes:
        name: Human-readable model name (e.g. ``"Holt-Winters"``). Used as the
            key in :class:`TrainedModel` / :class:`ExclusionRecord` and in the
            evaluation comparison table.
    """

    name: str

    def fit(self, train: "DemandSeries", scope: "ScopeConfig") -> None:
        """Train the model on the training portion of the series.

        Args:
            train: The training :class:`~src.preparation.DemandSeries` (the
                periods *before* the Holdout_Set).
            scope: The project :class:`~src.config.ScopeConfig` (Time_Grain,
                Geographic_Grain, seasonality hints, etc.).
        """
        ...

    def predict(self, horizon: int) -> "Forecast":
        """Forecast ``horizon`` periods ahead over the Holdout_Set.

        Args:
            horizon: Number of periods to forecast, equal to the Holdout_Set
                length so every model forecasts over the same holdout (R5.7).

        Returns:
            A :class:`Forecast` whose ``index`` aligns to the holdout periods.
        """
        ...


@dataclass
class Forecast:
    """A model's forecast, aligned to the Holdout_Set it covers (Requirement 5.7).

    Kept deliberately simple: the forecast ``values`` and the ``index`` of holdout
    periods they line up against, tagged with the producing ``model_name``. Every
    model in a ``train_all`` run produces a ``Forecast`` over the *same* horizon,
    so the Evaluation_Framework can place them side by side against the actual
    holdout values.

    Attributes:
        model_name: Name of the model that produced this forecast.
        values: The forecasted demand values, one per holdout period. Any
            array-like (``list``, ``numpy.ndarray``, ``pandas.Series``); left
            untyped-strict so concrete models can return their natural output.
        index: The holdout period index these values align to (typically a
            ``pandas.DatetimeIndex`` or list of periods). ``None`` when the
            producing model does not supply one, but concrete models should set it
            so alignment is explicit.
    """

    model_name: str
    values: Any
    index: Any = None


@dataclass
class TrainedModel:
    """A successfully trained model plus its Holdout_Set forecast (Requirement 5).

    ``train_all`` emits one of these for every candidate that fit and predicted
    without error. It keeps the fitted ``forecaster`` instance (so callers can
    re-forecast or inspect it) alongside the :class:`Forecast` it produced.

    Attributes:
        model_name: Name of the trained model (mirrors ``forecaster.name``).
        forecaster: The fitted :class:`Forecaster` instance.
        forecast: The :class:`Forecast` produced over the Holdout_Set.
    """

    model_name: str
    forecaster: "Forecaster"
    forecast: Forecast


@dataclass
class ExclusionRecord:
    """Why a candidate model was excluded from the comparison (Requirement 5.8).

    When a candidate cannot be trained or cannot forecast on the prepared data,
    ``train_all`` records the reason here instead of raising, so a single failure
    never aborts the remaining candidates. The excluded model still appears in the
    evaluation comparison, flagged with this reason.

    Attributes:
        model_name: Name of the excluded model.
        reason: Human-readable reason for exclusion (typically the string form of
            the exception that was raised during ``fit``/``predict``).
    """

    model_name: str
    reason: str


#: The result of training a single candidate: either a fitted model or a reason
#: it was excluded. ``train_all`` returns a list of these.
TrainResult = Union[TrainedModel, ExclusionRecord]

#: A candidate passed to :func:`train_all`: either a ready :class:`Forecaster`
#: instance or a zero-argument factory that builds one (so a fresh, unfitted model
#: is created per run without sharing state between candidates).
Candidate = Union["Forecaster", Callable[[], "Forecaster"]]


def _resolve_candidate(candidate: Candidate) -> "Forecaster":
    """Return a :class:`Forecaster` from a candidate instance or factory.

    A candidate may be given either as a ready ``Forecaster`` instance or as a
    zero-argument factory that constructs one. Factories are useful when a caller
    wants a fresh, unfitted model per ``train_all`` run. Anything callable that is
    not itself a ``Forecaster`` is treated as a factory and invoked.

    Args:
        candidate: A ``Forecaster`` instance or a zero-arg factory returning one.

    Returns:
        The resolved ``Forecaster`` instance.
    """
    # A Forecaster instance is not (typically) callable; a bare factory is.
    if callable(candidate) and not isinstance(candidate, Forecaster):
        return candidate()  # type: ignore[misc]
    return candidate  # type: ignore[return-value]


def _candidate_name(candidate: Candidate, forecaster: "Forecaster | None") -> str:
    """Best-effort model name for logging, even when construction failed.

    Prefers the resolved forecaster's ``name``; falls back to a ``name`` attribute
    on the raw candidate (e.g. a factory carrying one), else a generic label so an
    :class:`ExclusionRecord` always has a usable ``model_name``.
    """
    if forecaster is not None:
        name = getattr(forecaster, "name", None)
        if name:
            return str(name)
    raw_name = getattr(candidate, "name", None)
    if raw_name:
        return str(raw_name)
    return getattr(candidate, "__name__", "unknown_model")


def train_all(
    candidates: Iterable[Candidate],
    train: "DemandSeries",
    scope: "ScopeConfig",
    horizon: int,
) -> list[TrainResult]:
    """Train every candidate over the same holdout, isolating per-model failures.

    Iterates the ``candidates`` (each a :class:`Forecaster` instance or a
    zero-argument factory returning one). For each candidate it resolves the
    forecaster, then calls ``fit(train, scope)`` followed by ``predict(horizon)``.

    * On success it appends a :class:`TrainedModel` holding the fitted forecaster
      and its :class:`Forecast`.
    * On **any** exception - during construction, ``fit`` or ``predict`` - it
      appends an :class:`ExclusionRecord` capturing the model name and the reason
      (the string form of the exception) and continues with the next candidate.

    Because every candidate is called with the *same* ``horizon``, all produced
    forecasts cover the same Holdout_Set (Requirement 5.7); because failures are
    caught per candidate, one model failing never aborts the others
    (Requirement 5.8).

    Args:
        candidates: The candidate forecasters (instances or factories) to train.
        train: The training :class:`~src.preparation.DemandSeries`.
        scope: The project :class:`~src.config.ScopeConfig`.
        horizon: Forecast horizon (the Holdout_Set length) passed to every model
            so forecasts are comparable.

    Returns:
        A list of :class:`TrainedModel` / :class:`ExclusionRecord` results, in the
        same order as ``candidates`` - one entry per candidate, none dropped.
    """
    results: list[TrainResult] = []

    for candidate in candidates:
        forecaster: "Forecaster | None" = None
        try:
            forecaster = _resolve_candidate(candidate)
            forecaster.fit(train, scope)
            forecast = forecaster.predict(horizon)
            results.append(
                TrainedModel(
                    model_name=getattr(forecaster, "name", _candidate_name(candidate, forecaster)),
                    forecaster=forecaster,
                    forecast=forecast,
                )
            )
        except Exception as exc:  # noqa: BLE001 - deliberate: isolate per-model failure (R5.8)
            results.append(
                ExclusionRecord(
                    model_name=_candidate_name(candidate, forecaster),
                    reason=str(exc),
                )
            )

    return results
