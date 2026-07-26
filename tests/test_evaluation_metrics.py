"""Property-based test for error-metric correctness (Property 8).

Design reference:
- Correctness Properties -> Property 8 (Error-metric correctness)
- Requirements 6.2 (report MAE, RMSE, MAPE forecast-error metrics; all >= 0).

This module implements the *single* Hypothesis property test the design assigns
to Property 8, at 100+ iterations, per the Testing Strategy. It exercises
``src.evaluation.error_metrics``: given positionally aligned ``actual`` and
``forecast`` sequences of equal, non-zero length, the returned
:class:`~src.evaluation.Metrics` must satisfy three universal invariants:

1. **Non-negativity** - ``mae``, ``rmse`` and ``mape`` are all ``>= 0`` (they are
   averages of absolute / squared deviations, so a negative value would signal a
   sign or aggregation bug).
2. **Zero on a perfect forecast** - when ``forecast == actual`` element-for-
   element, every metric is exactly ``0.0`` (no error means no reported error).
3. **RMSE dominates MAE** - ``rmse >= mae`` always, because RMSE is an L2 mean of
   the same absolute deviations MAE averages under L1 (power-mean inequality).

The generator constrains inputs to the metric's real input space: equal-length,
non-empty, finite float pairs. Two shapes are drawn so both branches of the
property are exercised heavily:

* **Independent pairs** - actual and forecast drawn independently, driving the
  general non-negativity / RMSE-dominates-MAE checks with arbitrary error.
* **Perfect-forecast pairs** - ``forecast`` set equal to ``actual``, targeting the
  exact-zero branch. A separate dedicated property nails this case down so it is
  never lost in the random draws.
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from src.evaluation import error_metrics

# Finite, bounded demand-like values. Bounds keep squared terms well within float
# range so RMSE never overflows to ``inf`` (which would be a generator artifact,
# not a property violation). Values span negatives too - error_metrics coerces to
# float and never assumes sign - so the invariants are tested beyond valid demand.
VALUES = st.floats(
    min_value=-1e6,
    max_value=1e6,
    allow_nan=False,
    allow_infinity=False,
)


@st.composite
def aligned_actual_forecast(draw: st.DrawFn):
    """Generate an equal-length ``(actual, forecast)`` pair of finite floats.

    Length is 1-50 (non-empty, since ``error_metrics`` rejects empty input); both
    sequences share that length so they are positionally aligned as the function
    requires.
    """
    n = draw(st.integers(min_value=1, max_value=50))
    actual = draw(st.lists(VALUES, min_size=n, max_size=n))
    forecast = draw(st.lists(VALUES, min_size=n, max_size=n))
    return actual, forecast


# Feature: ride-hailing-demand-forecasting, Property 8: Error-metric correctness -
# MAE, RMSE, MAPE all non-negative; all exactly 0 when forecast equals actual;
# RMSE >= MAE
#
# Validates: Requirements 6.2
@settings(max_examples=200)
@given(data=aligned_actual_forecast())
def test_error_metric_correctness(data) -> None:
    actual, forecast = data

    metrics = error_metrics(actual, forecast)

    # --- Part 1: every metric is non-negative ----------------------------- #
    assert metrics.mae >= 0.0, f"MAE negative: {metrics.mae}"
    assert metrics.rmse >= 0.0, f"RMSE negative: {metrics.rmse}"
    assert metrics.mape >= 0.0, f"MAPE negative: {metrics.mape}"

    # --- Part 3: RMSE >= MAE (power-mean inequality) ---------------------- #
    # Allow a tiny tolerance for floating-point rounding at the boundary where
    # RMSE == MAE (e.g. all errors equal in magnitude).
    assert metrics.rmse >= metrics.mae - 1e-9, (
        f"RMSE ({metrics.rmse}) < MAE ({metrics.mae})"
    )


# Feature: ride-hailing-demand-forecasting, Property 8: Error-metric correctness -
# all metrics exactly 0 when forecast equals actual
#
# Validates: Requirements 6.2
@settings(max_examples=200)
@given(actual=st.lists(VALUES, min_size=1, max_size=50))
def test_error_metrics_zero_on_perfect_forecast(actual) -> None:
    # Part 2: a perfect forecast (forecast == actual) yields exactly zero error
    # for all three metrics.
    metrics = error_metrics(actual, list(actual))

    assert metrics.mae == 0.0, f"MAE not zero on perfect forecast: {metrics.mae}"
    assert metrics.rmse == 0.0, f"RMSE not zero on perfect forecast: {metrics.rmse}"
    assert metrics.mape == 0.0, f"MAPE not zero on perfect forecast: {metrics.mape}"
