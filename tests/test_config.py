"""Unit tests for the ScopeConfig single source of truth (src/config.py).

Covers Requirement 2.3 (the Analysis_Window is a contiguous span between 12 and
24 months) and Requirement 2.5 (post-definition scope changes are recorded with
their rationale). These are example-based unit tests, per the design's Testing
Strategy ("window validator accepts 12-24 months and rejects otherwise (R2.3),
scope-change logging (R2.5)").
"""

from __future__ import annotations

from datetime import date

import pytest

from src.config import (
    MAX_WINDOW_MONTHS,
    MIN_WINDOW_MONTHS,
    ScopeChange,
    ScopeConfig,
    default_scope,
)


def _window_of_months(months: int) -> tuple[date, date]:
    """Return (start, end) for an inclusive `months`-long window starting 2025-01.

    The window is inclusive of both endpoint months, so `window_months` equals
    `months`. For months=1 this is Jan 2025 only; for months=12, Jan..Dec 2025.
    """
    start = date(2025, 1, 1)
    # months inclusive -> advance (months - 1) months from the start month.
    total = (start.year * 12 + (start.month - 1)) + (months - 1)
    end_year, end_month0 = divmod(total, 12)
    return start, date(end_year, end_month0 + 1, 28)


# --------------------------------------------------------------------------- #
# Requirement 2.3 - window validator accepts 12-24 months, rejects otherwise
# --------------------------------------------------------------------------- #


class TestWindowValidatorAccepts:
    """R2.3: a contiguous 12-24 month span is accepted."""

    @pytest.mark.parametrize("months", list(range(MIN_WINDOW_MONTHS, MAX_WINDOW_MONTHS + 1)))
    def test_accepts_spans_in_range(self, months: int) -> None:
        start, end = _window_of_months(months)
        scope = ScopeConfig(window_start=start, window_end=end)
        assert scope.window_months == months
        assert scope.is_window_valid() is True
        # validate_window must not raise for an in-range span.
        scope.validate_window()

    def test_accepts_lower_boundary_12_months(self) -> None:
        start, end = _window_of_months(MIN_WINDOW_MONTHS)
        scope = ScopeConfig(window_start=start, window_end=end)
        assert scope.window_months == 12
        assert scope.is_window_valid() is True

    def test_accepts_upper_boundary_24_months(self) -> None:
        start, end = _window_of_months(MAX_WINDOW_MONTHS)
        scope = ScopeConfig(window_start=start, window_end=end)
        assert scope.window_months == 24
        assert scope.is_window_valid() is True

    def test_default_scope_window_is_valid_12_months(self) -> None:
        scope = default_scope()
        assert scope.window_months == 12
        assert scope.is_window_valid() is True


class TestWindowValidatorRejects:
    """R2.3: spans shorter than 12, longer than 24, or reversed are rejected."""

    @pytest.mark.parametrize("months", [1, 6, 11])
    def test_rejects_spans_shorter_than_12(self, months: int) -> None:
        start, end = _window_of_months(months)
        # __post_init__ calls validate_window, so construction itself must fail.
        with pytest.raises(ValueError):
            ScopeConfig(window_start=start, window_end=end)

    def test_rejects_just_below_lower_boundary_11_months(self) -> None:
        start, end = _window_of_months(MIN_WINDOW_MONTHS - 1)
        with pytest.raises(ValueError):
            ScopeConfig(window_start=start, window_end=end)

    @pytest.mark.parametrize("months", [25, 30, 36])
    def test_rejects_spans_longer_than_24(self, months: int) -> None:
        start, end = _window_of_months(months)
        with pytest.raises(ValueError):
            ScopeConfig(window_start=start, window_end=end)

    def test_rejects_just_above_upper_boundary_25_months(self) -> None:
        start, end = _window_of_months(MAX_WINDOW_MONTHS + 1)
        with pytest.raises(ValueError):
            ScopeConfig(window_start=start, window_end=end)

    def test_rejects_reversed_window(self) -> None:
        # end before start -> invalid regardless of month count.
        with pytest.raises(ValueError):
            ScopeConfig(window_start=date(2026, 4, 30), window_end=date(2025, 5, 1))

    def test_validate_window_error_message_mentions_bounds(self) -> None:
        start, end = _window_of_months(6)
        with pytest.raises(ValueError) as exc_info:
            ScopeConfig(window_start=start, window_end=end)
        message = str(exc_info.value)
        assert str(MIN_WINDOW_MONTHS) in message
        assert str(MAX_WINDOW_MONTHS) in message


# --------------------------------------------------------------------------- #
# Requirement 2.5 - scope-change logging records value and rationale
# --------------------------------------------------------------------------- #


class TestScopeChangeLogging:
    """R2.5: post-definition scope changes are recorded with their rationale."""

    def test_records_old_and_new_value_and_rationale(self) -> None:
        scope = default_scope()
        old_value = scope.time_grain
        updated = scope.record_scope_change(
            "time_grain", "hourly", rationale="Client needs intraday positioning."
        )

        assert len(updated.change_log) == 1
        change = updated.change_log[0]
        assert isinstance(change, ScopeChange)
        assert change.field_name == "time_grain"
        assert change.old_value == old_value
        assert change.new_value == "hourly"
        assert change.rationale == "Client needs intraday positioning."
        # The field itself is updated on the returned config.
        assert updated.time_grain == "hourly"

    def test_returns_new_config_and_leaves_original_unchanged(self) -> None:
        scope = default_scope()
        updated = scope.record_scope_change(
            "geographic_grain", "taxi_zone", rationale="Finer spatial detail requested."
        )

        # Immutability: original config is untouched.
        assert scope.geographic_grain == "borough"
        assert scope.change_log == []
        # A new instance is returned, not the same object.
        assert updated is not scope
        assert updated.geographic_grain == "taxi_zone"

    def test_records_multiple_changes_in_order(self) -> None:
        scope = default_scope()
        step1 = scope.record_scope_change("time_grain", "hourly", rationale="First change.")
        step2 = step1.record_scope_change("holdout_periods", 14, rationale="Shorter holdout.")

        assert len(step2.change_log) == 2
        assert step2.change_log[0].field_name == "time_grain"
        assert step2.change_log[1].field_name == "holdout_periods"
        assert step2.change_log[1].old_value == scope.holdout_periods
        assert step2.change_log[1].new_value == 14
        # The intermediate config keeps only the first change.
        assert len(step1.change_log) == 1

    def test_uses_provided_changed_at_date(self) -> None:
        scope = default_scope()
        when = date(2025, 6, 15)
        updated = scope.record_scope_change(
            "time_grain", "hourly", rationale="Dated change.", changed_at=when
        )
        assert updated.change_log[0].changed_at == when

    @pytest.mark.parametrize("bad_rationale", ["", "   ", "\t\n"])
    def test_requires_non_empty_rationale(self, bad_rationale: str) -> None:
        scope = default_scope()
        with pytest.raises(ValueError):
            scope.record_scope_change("time_grain", "hourly", rationale=bad_rationale)
        # No change recorded and original untouched.
        assert scope.change_log == []
        assert scope.time_grain == "daily"

    def test_rejects_unknown_field_name(self) -> None:
        scope = default_scope()
        with pytest.raises(ValueError):
            scope.record_scope_change(
                "not_a_field", "value", rationale="Attempt to change a non-field."
            )

    def test_rejects_change_that_produces_invalid_window(self) -> None:
        # Changing window_end to before window_start must fail validation on the
        # new instance (via __post_init__), so no invalid config is produced.
        scope = default_scope()
        with pytest.raises(ValueError):
            scope.record_scope_change(
                "window_end", date(2020, 1, 1), rationale="Impossible window."
            )
        # Original remains valid and unchanged.
        assert scope.is_window_valid() is True
        assert scope.change_log == []
