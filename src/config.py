"""Scope configuration - the single source of truth for the project's scope.

Requirement 2 fixes the *shape* of the forecasting scope (a single documented
time grain, a single geographic grain, a 12-24 month analysis window, a
documented candidate-model set) and requires that any post-definition change be
recorded with its rationale. This module makes that structurally enforceable:
every component reads scope from one immutable ``ScopeConfig`` object, so the
"single documented value used consistently" guarantee holds by construction.

Design references:
- Data Models -> ScopeConfig (frozen dataclass, single source of truth)
- Key Design Decisions -> proposed defaults (daily / borough / 2025-05 -> 2026-04)
- Requirements 2.1, 2.2, 2.3, 2.4, 2.5
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, replace
from datetime import date
from typing import Any

# --- Proposed defaults (Key Design Decisions table) --------------------------

#: Time_Grain default (Requirement 2.1). Daily gives a clean, complete series
#: every candidate model can train on.
DEFAULT_TIME_GRAIN = "daily"

#: Geographic_Grain default (Requirement 2.2). Boroughs give a small, stable set
#: of parallel series suited to multivariate (VAR/VARMAX) joint forecasting.
DEFAULT_GEOGRAPHIC_GRAIN = "borough"

#: Analysis_Window default (Requirement 2.3): 12 months ending 2026-04, aligned
#: to the already-downloaded ``fhvhv_2026-04.parquet``.
DEFAULT_WINDOW_START = date(2025, 5, 1)
DEFAULT_WINDOW_END = date(2026, 4, 30)

#: Candidate_Model_Set default (Requirement 2.4): baseline -> classical
#: univariate -> multivariate -> modern -> ML -> deep learning.
DEFAULT_CANDIDATE_MODELS: tuple[str, ...] = (
    "Holt-Winters",
    "SARIMA",
    "SARIMAX",
    "VAR",
    "VARMAX",
    "Prophet",
    "XGBoost",
    "LSTM",
    "GRU",
)

#: Most-recent contiguous holdout reserved for evaluation (30 days at daily grain).
DEFAULT_HOLDOUT_PERIODS = 30

#: Lag features required by the ML models (Requirement 4.6): yesterday, last week,
#: two weeks back.
DEFAULT_LAGS: tuple[int, ...] = (1, 7, 14)

#: Inclusive bounds of the acceptable Analysis_Window length, in months (R2.3).
MIN_WINDOW_MONTHS = 12
MAX_WINDOW_MONTHS = 24


@dataclass(frozen=True)
class ScopeChange:
    """A single recorded post-definition scope change and its rationale (R2.5).

    Instances are immutable and accumulated in ``ScopeConfig.change_log`` so the
    project documentation carries a complete audit trail of every scope value
    that was changed after initial definition.
    """

    field_name: str
    old_value: Any
    new_value: Any
    rationale: str
    changed_at: date = field(default_factory=date.today)


def _window_length_months(start: date, end: date) -> int:
    """Return the inclusive number of calendar months spanned by ``start``..``end``.

    The Analysis_Window is a contiguous span of monthly NYC TLC files, so the
    natural length is the inclusive count of calendar months. For the default
    window 2025-05-01 -> 2026-04-30 this returns 12 (May 2025 through April 2026).
    """
    return (end.year * 12 + end.month) - (start.year * 12 + start.month) + 1


@dataclass(frozen=True)
class ScopeConfig:
    """Immutable single source of truth for the forecasting scope (Requirement 2).

    Every downstream component (EDA, preparation, modeling, evaluation) reads its
    scope values from one ``ScopeConfig`` instance, structurally guaranteeing the
    "single documented value used consistently" requirement (R2.1, R2.2).

    The dataclass is frozen, so a scope value cannot be mutated in place. Changes
    are made through :meth:`record_scope_change`, which returns a *new*
    ``ScopeConfig`` with the change appended to ``change_log`` (R2.5).
    """

    time_grain: str = DEFAULT_TIME_GRAIN
    geographic_grain: str = DEFAULT_GEOGRAPHIC_GRAIN
    window_start: date = DEFAULT_WINDOW_START
    window_end: date = DEFAULT_WINDOW_END
    candidate_models: list[str] = field(
        default_factory=lambda: list(DEFAULT_CANDIDATE_MODELS)
    )
    holdout_periods: int = DEFAULT_HOLDOUT_PERIODS
    lags: list[int] = field(default_factory=lambda: list(DEFAULT_LAGS))
    change_log: list[ScopeChange] = field(default_factory=list)

    def __post_init__(self) -> None:
        # Guarantee the single source of truth is always a valid window: an
        # invalid ScopeConfig can never be constructed (R2.3).
        self.validate_window()

    @property
    def window_months(self) -> int:
        """Inclusive length of the Analysis_Window in calendar months (R2.3)."""
        return _window_length_months(self.window_start, self.window_end)

    def is_window_valid(self) -> bool:
        """Return ``True`` iff the window is a contiguous 12-24 month span (R2.3)."""
        if self.window_end < self.window_start:
            return False
        return MIN_WINDOW_MONTHS <= self.window_months <= MAX_WINDOW_MONTHS

    def validate_window(self) -> None:
        """Validate the Analysis_Window, raising ``ValueError`` if out of range.

        Accepts a contiguous span between 12 and 24 months inclusive; rejects
        reversed windows and spans shorter than 12 or longer than 24 months
        (Requirement 2.3).
        """
        if self.window_end < self.window_start:
            raise ValueError(
                "Analysis_Window is invalid: window_end "
                f"({self.window_end}) is before window_start ({self.window_start})."
            )
        months = self.window_months
        if not (MIN_WINDOW_MONTHS <= months <= MAX_WINDOW_MONTHS):
            raise ValueError(
                f"Analysis_Window must span between {MIN_WINDOW_MONTHS} and "
                f"{MAX_WINDOW_MONTHS} months (Requirement 2.3); got {months} "
                f"months for {self.window_start} -> {self.window_end}."
            )

    def record_scope_change(
        self,
        field_name: str,
        new_value: Any,
        rationale: str,
        changed_at: date | None = None,
    ) -> "ScopeConfig":
        """Return a new ``ScopeConfig`` with ``field_name`` updated and the change logged.

        Because ``ScopeConfig`` is frozen, this does not mutate the current
        instance. It appends a :class:`ScopeChange` (capturing the old value, new
        value, and rationale) to a fresh ``change_log`` and uses
        :func:`dataclasses.replace` to build the updated, still-immutable config.
        The rationale is mandatory (Requirement 2.5), and window fields are
        re-validated on the new instance via ``__post_init__``.

        Args:
            field_name: The scope field being changed (e.g. ``"time_grain"``).
            new_value: The new value for that field.
            rationale: Why the change was made. Required and non-empty (R2.5).
            changed_at: Optional date of the change; defaults to today.

        Raises:
            ValueError: If ``field_name`` is not a changeable scope field, if the
                rationale is empty, or if the resulting window is invalid.
        """
        if not rationale or not rationale.strip():
            raise ValueError(
                "A non-empty rationale is required to record a scope change "
                "(Requirement 2.5)."
            )

        changeable = {f.name for f in fields(self) if f.name != "change_log"}
        if field_name not in changeable:
            raise ValueError(
                f"'{field_name}' is not a changeable scope field. "
                f"Choose one of: {sorted(changeable)}."
            )

        change = ScopeChange(
            field_name=field_name,
            old_value=getattr(self, field_name),
            new_value=new_value,
            rationale=rationale,
            changed_at=changed_at if changed_at is not None else date.today(),
        )
        new_change_log = list(self.change_log) + [change]
        return replace(self, **{field_name: new_value, "change_log": new_change_log})


def default_scope() -> ScopeConfig:
    """Return a fresh ``ScopeConfig`` populated with the proposed project defaults.

    Each call produces an independent instance with its own mutable list fields,
    so callers never share ``candidate_models``/``lags``/``change_log`` state.
    """
    return ScopeConfig()
