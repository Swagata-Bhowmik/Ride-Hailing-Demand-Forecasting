"""Property-based test for carry-forward selection count (Property 11).

Design reference:
- Correctness Properties -> Property 11 (Carry-forward selection count)
- Requirements 6.6 (WHEN the comparison is complete, the Evaluation_Framework
  SHALL identify between three and five models to carry forward for deep
  explanation, with a documented justification based on the reported metrics).

This module implements the *single* Hypothesis property test the design assigns
to Property 11, at 100+ iterations, per the Testing Strategy. It exercises
``src.evaluation.select_carry_forward``: given a comparison table with at least
three distinct models (a mix of scored rows with finite metrics and excluded
rows with ``NaN`` metrics), the returned short-list must contain between three
and five names inclusive, and every returned name must be present in the table's
``model_name`` column.

The generator builds each table from a list of :class:`~src.evaluation.ModelResult`
objects passed through :func:`~src.evaluation.comparison_table`, so the input is
exactly the shape the function is designed to consume:

* distinct ``model_name`` values (so the table has >= 3 distinct models);
* scored models carry finite, non-negative :class:`~src.evaluation.Metrics`;
* excluded models carry an ``excluded_reason`` and thus ``NaN`` metrics.

The mix of scored/excluded rows is drawn freely so the test also covers the
fallback path where fewer than three models are scored and excluded models are
appended to reach the minimum of three.
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from src.evaluation import (
    CARRY_FORWARD_MAX,
    CARRY_FORWARD_MIN,
    Metrics,
    ModelResult,
    comparison_table,
    select_carry_forward,
)

# Finite, non-negative metric values (same units as demand / a percentage for
# MAPE). No NaN/inf so scored rows are unambiguously "scored".
_FINITE = dict(allow_nan=False, allow_infinity=False)
_METRIC = st.floats(min_value=0.0, max_value=1e6, **_FINITE)


@st.composite
def comparison_tables(draw):
    """Draw a comparison table with >= 3 distinct models, mixing scored/excluded.

    Model names are unique (so distinct-model count is well defined). Each model
    is independently either scored (finite Metrics) or excluded (NaN metrics via
    an ``excluded_reason``), so the generator spans the full range from all-scored
    to mostly-excluded tables.
    """
    n_models = draw(st.integers(min_value=3, max_value=8))
    names = draw(
        st.lists(
            st.text(
                alphabet="abcdefghijklmnopqrstuvwxyz0123456789_",
                min_size=1,
                max_size=12,
            ),
            min_size=n_models,
            max_size=n_models,
            unique=True,
        )
    )

    results: list[ModelResult] = []
    for name in names:
        is_scored = draw(st.booleans())
        if is_scored:
            metrics = Metrics(
                mae=draw(_METRIC),
                rmse=draw(_METRIC),
                mape=draw(_METRIC),
            )
            results.append(ModelResult(model_name=name, metrics=metrics))
        else:
            results.append(
                ModelResult(model_name=name, excluded_reason="excluded by generator")
            )

    return comparison_table(results)


# Feature: ride-hailing-demand-forecasting, Property 11: Carry-forward selection count
@settings(max_examples=200)
@given(table=comparison_tables())
def test_carry_forward_selection_count(table):
    """Property 11: 3-5 names, all present in the table.

    **Validates: Requirements 6.6**
    """
    selected = select_carry_forward(table)

    # Between three and five names inclusive.
    assert CARRY_FORWARD_MIN <= len(selected) <= CARRY_FORWARD_MAX

    # Every returned name is present in the table's model_name column.
    present = set(table["model_name"])
    for name in selected:
        assert name in present
