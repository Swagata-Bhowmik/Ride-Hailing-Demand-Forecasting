"""Property-based test for comparison table completeness (Property 9).

Design reference:
- Correctness Properties -> Property 9 (Comparison table completeness)
- Requirements 6.3 (the Evaluation_Framework SHALL present a comparison table
  containing the Forecast_Error_Metrics for every trained model, including
  underperforming models).

This module implements the *single* Hypothesis property test the design assigns
to Property 9, at 100+ iterations, per the Testing Strategy. It exercises
``src.evaluation.comparison_table``: given any list of :class:`ModelResult`
objects mixing scored and excluded models, the returned table must contain
exactly one row per input model, in the same order, and every row must carry the
same set of columns (including the metric columns) regardless of whether the
model was scored or excluded.

The generator draws each ``ModelResult`` as either:

* a **scored** result carrying finite non-negative :class:`Metrics` and
  ``excluded_reason=None``; or
* an **excluded** result carrying ``metrics=None`` and a non-empty reason string.

Model names are drawn from text (allowed to repeat, since Property 9 only
requires one row *per input result* preserving order, not global uniqueness).
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from src.evaluation import METRIC_COLUMNS, Metrics, ModelResult, comparison_table

EXPECTED_COLUMNS = ["model_name", *METRIC_COLUMNS, "excluded", "excluded_reason"]

# Finite, non-negative metric components (mae, rmse, mape) as the Metrics dataclass
# carries. Exact numeric values are irrelevant to Property 9 (shape/completeness),
# so any finite float is fine.
_finite_nonneg = st.floats(
    min_value=0.0, max_value=1e9, allow_nan=False, allow_infinity=False
)


@st.composite
def model_results(draw):
    """Draw a :class:`ModelResult` that is either scored or excluded."""
    name = draw(st.text(min_size=1, max_size=12))
    if draw(st.booleans()):
        # Scored model: real metrics, no exclusion reason.
        metrics = Metrics(
            mae=draw(_finite_nonneg),
            rmse=draw(_finite_nonneg),
            mape=draw(_finite_nonneg),
        )
        return ModelResult(model_name=name, metrics=metrics, excluded_reason=None)
    # Excluded model: no metrics, a non-empty reason.
    reason = draw(st.text(min_size=1, max_size=40))
    return ModelResult(model_name=name, metrics=None, excluded_reason=reason)


# Feature: ride-hailing-demand-forecasting, Property 9: Comparison table completeness
@settings(max_examples=200)
@given(results=st.lists(model_results(), max_size=15))
def test_comparison_table_completeness(results):
    """Property 9: exactly one row per model, in order, with identical columns.

    **Validates: Requirements 6.3**
    """
    table = comparison_table(results)

    # Exactly one row per input model, preserving order.
    assert len(table) == len(results)
    assert list(table["model_name"]) == [r.model_name for r in results]

    # Identical column set for every row: a DataFrame has a single shared column
    # schema, so assert it equals the expected set (including all metric columns,
    # present even for excluded models).
    assert list(table.columns) == EXPECTED_COLUMNS
    for col in METRIC_COLUMNS:
        assert col in table.columns

    # The excluded flag reflects each input's exclusion status; excluded rows still
    # carry the metric columns (as NaN), so the schema is uniform across all rows.
    assert list(table["excluded"]) == [r.excluded_reason is not None for r in results]
