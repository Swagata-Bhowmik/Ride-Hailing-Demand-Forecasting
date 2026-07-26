"""Evaluation_Framework - pure functions that evaluate model forecasts against a
reserved out-of-sample Holdout_Set (Requirement 6).

Like the preparation pipeline, this module is pure logic: it receives already
loaded ``pandas`` DataFrames (the long-format :data:`~src.preparation.DemandSeries`)
and returns new DataFrames, with no I/O. That is what makes its behavior
machine-checkable by the design's Correctness Properties.

Implemented so far:

* :func:`split_holdout` (9.1) - reserve the most-recent contiguous ``n`` periods
  of a demand series as the Holdout_Set and keep the earlier remainder for
  training, disjoint and exactly reconstructing the original on concatenation
  (Requirement 6.1, design Correctness Property 7).
* :func:`error_metrics` (9.2) - MAE, RMSE, MAPE with a documented zero-actual
  convention (Requirement 6.2, Correctness Property 8).
* :class:`ModelResult`, :func:`build_model_results`, :func:`comparison_table`
  and :func:`plot_forecast_vs_actual` (9.3) - a one-row-per-model comparison
  table that includes underperforming and excluded models with the same metric
  columns, plus a forecast-vs-actual overlay plot (Requirements 6.3, 6.4,
  Correctness Property 9).

Later tasks extend this same module: error-by-period reporting (9.4) and
carry-forward model selection (9.5). The module is intentionally kept as a flat
collection of pure functions so those steps can be added without reworking a
rigid schema.

Design references:
- Components and Interfaces -> Evaluation_Framework (`src/evaluation.py`)
- Data Models -> DemandSeries (long format), Metrics, ModelResult
- Correctness Properties -> Property 7 (holdout split), Property 9 (table completeness)
- Requirements 6.1, 6.2, 6.3, 6.4
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional, Sequence, Union

import numpy as np
import pandas as pd

from src.models.base import ExclusionRecord, Forecast, TrainedModel
from src.preparation import PERIOD_COLUMN, DemandSeries

if TYPE_CHECKING:  # pragma: no cover - imported only for type checking
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

__all__ = [
    "split_holdout",
    "Metrics",
    "error_metrics",
    "ModelResult",
    "METRIC_COLUMNS",
    "build_model_results",
    "comparison_table",
    "plot_forecast_vs_actual",
    "error_by_period",
    "CARRY_FORWARD_MIN",
    "CARRY_FORWARD_MAX",
    "select_carry_forward",
]

#: The inclusive bounds Requirement 6.6 places on the carry-forward set: at least
#: three and at most five models are carried forward for deep explanation. Kept as
#: module constants so :func:`select_carry_forward` and its property test (9.10,
#: Correctness Property 11) agree on the exact range.
CARRY_FORWARD_MIN = 3
CARRY_FORWARD_MAX = 5

#: The metric columns every row of the comparison table carries, in order. Kept as
#: a module constant so the table builder and its tests agree on the exact set that
#: Correctness Property 9 requires to be identical for every model row.
METRIC_COLUMNS = ("mae", "rmse", "mape")


def split_holdout(
    series: DemandSeries,
    holdout_periods: int,
    *,
    period_column: str = PERIOD_COLUMN,
) -> "tuple[DemandSeries, DemandSeries]":
    """Split a demand series into ``(train, holdout)`` by reserving recent periods.

    The Holdout_Set is defined (Requirement 6.1) as the *most recent contiguous
    portion* of the demand series, reserved for out-of-sample evaluation and
    excluded from all training. Because the :data:`~src.preparation.DemandSeries`
    is a **multi-region long format** - one row per ``(period, region)`` - "the
    most recent ``n`` periods" is interpreted on the *period* axis, not the row
    axis: the holdout is every row whose ``period`` is among the ``n`` most recent
    distinct period timestamps, and the training set is every row in the earlier
    remaining periods. This keeps all regions' observations for a given period
    together on the same side of the split, so no future period ever leaks into
    training for any region.

    The split is a clean partition of the input rows by period:

    * ``holdout`` = rows whose ``period`` is in the last ``n`` distinct periods.
    * ``train``   = rows whose ``period`` is in the earlier ``d - n`` periods,
      where ``d`` is the number of distinct periods.
    * The two frames are **disjoint** (no shared rows / no shared periods), and
      concatenating them reproduces the original series exactly as a row-set
      (design Correctness Property 7). Each side preserves the input's relative
      row order, so ``pd.concat([train, holdout])`` yields the same rows the
      caller passed in.

    Args:
        series: A :data:`~src.preparation.DemandSeries` (long format) with at
            least a ``period`` column. Its ``period`` values are timestamps at the
            Time_Grain (e.g. one per calendar day at the daily grain).
        holdout_periods: The number ``n`` of most-recent distinct periods to
            reserve for the holdout. Must satisfy ``1 <= n < d`` where ``d`` is the
            number of distinct periods in ``series`` (so that training is
            non-empty).
        period_column: Name of the period column. Defaults to ``"period"``.

    Returns:
        A ``(train, holdout)`` tuple of DemandSeries. ``holdout`` holds the rows in
        the most-recent ``n`` distinct periods; ``train`` holds the earlier
        remainder. Both preserve the original relative row order and index.

    Raises:
        KeyError: If ``period_column`` is absent from ``series``.
        ValueError: If ``holdout_periods`` is not a positive integer, or is not
            strictly less than the number of distinct periods in ``series`` (which
            would leave no data for training).
    """
    if period_column not in series.columns:
        raise KeyError(
            f"Period column '{period_column}' not found in series. "
            f"Available columns: {list(series.columns)}."
        )

    # Reject non-integer / boolean n up front so the range message is meaningful.
    if isinstance(holdout_periods, bool) or not isinstance(holdout_periods, int):
        raise ValueError(
            f"holdout_periods must be an integer; got {holdout_periods!r} "
            f"({type(holdout_periods).__name__})."
        )

    # The most-recent-period convention operates on the set of *distinct* periods.
    distinct_periods = pd.unique(series[period_column])
    num_periods = len(distinct_periods)

    if not (1 <= holdout_periods < num_periods):
        raise ValueError(
            "holdout_periods must satisfy 1 <= holdout_periods < number of "
            f"distinct periods ({num_periods}) so that training is non-empty; "
            f"got holdout_periods={holdout_periods}. With {num_periods} distinct "
            "period(s), no valid holdout that leaves training data exists."
            if num_periods >= 2
            else (
                "holdout_periods requires at least 2 distinct periods to split "
                f"(one for training, one for holdout); series has {num_periods}."
            )
        )

    # Identify the n most-recent distinct periods by timestamp order. Sorting the
    # distinct values makes the split independent of the input row order.
    ordered_periods = pd.Series(distinct_periods).sort_values(kind="stable")
    holdout_period_values = set(ordered_periods.iloc[-holdout_periods:])

    in_holdout = series[period_column].isin(holdout_period_values)

    # Preserve each side's original relative row order and index so concatenation
    # reconstructs the caller's series exactly (Property 7).
    holdout = series.loc[in_holdout].copy()
    train = series.loc[~in_holdout].copy()

    return train, holdout


@dataclass(frozen=True)
class Metrics:
    """Forecast_Error_Metrics for a single model on the Holdout_Set (Requirement 6.2).

    All three fields are non-negative accuracy measures (design Data Models,
    Correctness Property 8):

    * ``mae``  - Mean Absolute Error, in the same units as demand.
    * ``rmse`` - Root Mean Squared Error, in the same units as demand. Because it
      is an L2 average of the same absolute deviations MAE averages under L1,
      ``rmse >= mae`` always holds (Jensen's inequality / power-mean inequality).
    * ``mape`` - Mean Absolute Percentage Error, expressed as a percentage
      (``0.0`` means a perfect forecast). See :func:`error_metrics` for the
      documented zero-actual convention.
    """

    mae: float
    rmse: float
    mape: float


def error_metrics(actual, forecast) -> Metrics:
    """Compute MAE, RMSE and MAPE for aligned actual/forecast values (Requirement 6.2).

    The two inputs are treated as **positionally aligned** 1-D sequences: element
    ``i`` of ``forecast`` is the prediction for element ``i`` of ``actual``. Any
    array-like is accepted - a :class:`numpy.ndarray`, a :class:`pandas.Series`,
    or a plain Python ``list`` - and is coerced to a float ``numpy`` array before
    computation. When a ``pandas.Series`` is passed its index is ignored; only
    positional order matters, so the caller is responsible for aligning the two
    series beforehand (e.g. both sorted by period/region).

    Definitions, where ``e_i = actual_i - forecast_i`` over ``n`` paired values:

    * ``MAE  = mean(|e_i|)``
    * ``RMSE = sqrt(mean(e_i ** 2))``
    * ``MAPE = 100 * mean(|e_i| / |actual_i|)`` over the periods where
      ``actual_i != 0`` (see the zero convention below).

    These satisfy design **Correctness Property 8**: every metric is ``>= 0``;
    when ``forecast == actual`` every metric is exactly ``0.0``; and
    ``RMSE >= MAE`` always (RMSE is an L2 mean of the same ``|e_i|`` that MAE
    averages under L1, so it can never be smaller).

    MAPE zero-actual convention (design Error Handling, "MAPE with zero actuals"):
        MAPE is undefined when an actual value is 0 because it would divide by
        zero. This implementation **excludes periods whose actual value is 0**
        from the MAPE average and computes the mean of the percentage error over
        the remaining nonzero-actual periods only. If **every** actual value is 0
        (no periods remain), MAPE is defined as ``0.0``. This keeps MAPE finite
        and non-negative for every input while leaving MAE and RMSE - which have
        no division - computed over all periods.

    Args:
        actual: Array-like of observed demand values (numpy array, pandas Series,
            or list). Coerced to float.
        forecast: Array-like of predicted demand values, positionally aligned with
            ``actual`` and of the same length. Coerced to float.

    Returns:
        A :class:`Metrics` with ``mae``, ``rmse`` and ``mape`` as floats.

    Raises:
        ValueError: If ``actual`` and ``forecast`` have different lengths, or if
            either is empty (no paired values to score).
    """
    actual_arr = np.asarray(actual, dtype=float).ravel()
    forecast_arr = np.asarray(forecast, dtype=float).ravel()

    if actual_arr.shape != forecast_arr.shape:
        raise ValueError(
            "actual and forecast must have the same length; got "
            f"{actual_arr.shape[0]} and {forecast_arr.shape[0]}."
        )
    if actual_arr.size == 0:
        raise ValueError("actual and forecast must be non-empty to compute metrics.")

    errors = actual_arr - forecast_arr
    abs_errors = np.abs(errors)

    mae = float(np.mean(abs_errors))
    rmse = float(np.sqrt(np.mean(errors**2)))

    # MAPE: exclude zero-actual periods (division would be undefined); if all
    # actuals are zero, no periods remain and MAPE is defined as 0.0.
    nonzero = actual_arr != 0.0
    if np.any(nonzero):
        mape = float(
            100.0
            * np.mean(abs_errors[nonzero] / np.abs(actual_arr[nonzero]))
        )
    else:
        mape = 0.0

    return Metrics(mae=mae, rmse=rmse, mape=mape)


@dataclass
class ModelResult:
    """A single model's evaluation outcome for the comparison table (Requirement 6.3).

    ``train_all`` (see :mod:`src.models.base`) yields either a
    :class:`~src.models.base.TrainedModel` or an
    :class:`~src.models.base.ExclusionRecord` per candidate. A ``ModelResult``
    normalizes both cases into one shape the Evaluation_Framework can compare
    uniformly, so *every* candidate - the winners, the underperformers, and the
    ones that could not be trained at all - occupies exactly one entry.

    * A **trained** model carries computed :class:`Metrics` and its
      :class:`~src.models.base.Forecast`, with ``excluded_reason=None``.
    * An **excluded** model carries ``metrics=None`` and ``forecast=None`` with a
      non-empty ``excluded_reason`` explaining why it was dropped (R5.8).

    Keeping excluded models as first-class rows (rather than silently omitting
    them) is what makes the comparison "honest": the reader sees which models
    failed and why, not just the survivors. :func:`comparison_table` renders every
    ``ModelResult`` with the *same* metric columns regardless of which case it is
    (design Correctness Property 9).

    Attributes:
        model_name: Name of the model (matches ``Forecaster.name``). Should be
            unique within a set of results so the table has one row per model.
        metrics: The :class:`Metrics` computed on the Holdout_Set, or ``None`` when
            the model was excluded / not scored.
        forecast: The model's :class:`~src.models.base.Forecast` over the
            Holdout_Set, or ``None`` when the model was excluded.
        excluded_reason: Human-readable reason the model was excluded, or ``None``
            for a model that was trained and scored normally.
    """

    model_name: str
    metrics: Optional[Metrics] = None
    forecast: Optional[Forecast] = None
    excluded_reason: Optional[str] = None

    @property
    def is_excluded(self) -> bool:
        """Whether this result represents an excluded (non-scored) model."""
        return self.excluded_reason is not None


def _metrics_for_actual(actual, forecast_values) -> Optional[Metrics]:
    """Best-effort :func:`error_metrics`, returning ``None`` on any misalignment.

    Used by :func:`build_model_results` to score a trained model against the
    holdout actuals. A forecast whose length does not match the actuals (or is
    otherwise unusable) yields ``None`` so the model is still represented in the
    table - as an unscored row - rather than aborting the whole comparison.
    """
    try:
        return error_metrics(actual, forecast_values)
    except (ValueError, TypeError):
        return None


def build_model_results(
    train_results: "Sequence[Union[TrainedModel, ExclusionRecord]]",
    actual,
) -> "list[ModelResult]":
    """Turn ``train_all`` output into scored :class:`ModelResult` rows.

    Bridges the Forecasting_System and the Evaluation_Framework: for each result
    from :func:`src.models.base.train_all` it produces one :class:`ModelResult`,
    preserving input order and dropping nothing.

    * A :class:`~src.models.base.TrainedModel` is scored by comparing its
      forecast values against ``actual`` (the Holdout_Set actuals, positionally
      aligned) via :func:`error_metrics`. If the forecast cannot be scored (e.g.
      a length mismatch), the row is kept with ``metrics=None`` and an
      ``excluded_reason`` noting the misalignment, so it is never silently lost.
    * An :class:`~src.models.base.ExclusionRecord` becomes an excluded row with
      its ``reason`` and no metrics.

    Args:
        train_results: The per-candidate results from ``train_all`` (a mix of
            :class:`~src.models.base.TrainedModel` and
            :class:`~src.models.base.ExclusionRecord`).
        actual: The Holdout_Set actual demand values, positionally aligned with
            each model's forecast values (typically the ``demand`` column of the
            holdout series, ordered consistently with how the models forecast).

    Returns:
        A list of :class:`ModelResult`, one per input result, in the same order.
    """
    results: list[ModelResult] = []
    for item in train_results:
        if isinstance(item, ExclusionRecord):
            results.append(
                ModelResult(model_name=item.model_name, excluded_reason=item.reason)
            )
            continue

        # A TrainedModel: score its forecast against the holdout actuals.
        forecast = item.forecast
        metrics = _metrics_for_actual(actual, getattr(forecast, "values", None))
        if metrics is None:
            results.append(
                ModelResult(
                    model_name=item.model_name,
                    forecast=forecast,
                    excluded_reason=(
                        "forecast could not be scored against holdout actuals "
                        "(length mismatch or non-numeric values)"
                    ),
                )
            )
        else:
            results.append(
                ModelResult(
                    model_name=item.model_name,
                    metrics=metrics,
                    forecast=forecast,
                )
            )
    return results


def comparison_table(results: "Sequence[ModelResult]") -> pd.DataFrame:
    """Build the model comparison table - one row per model, losers included (R6.3).

    Requirement 6.3 asks the Evaluation_Framework to "present a comparison table
    containing the Forecast_Error_Metrics for every trained model, **including
    underperforming models**." This implementation goes one step further and also
    includes models that were *excluded* (could not be trained/scored), so the
    table is a complete, honest census of the whole Candidate_Model_Set.

    The returned :class:`pandas.DataFrame` has **exactly one row per input
    ``ModelResult``**, in the order given, and **the same columns for every row**
    (design Correctness Property 9):

    * ``model_name`` - the model's name.
    * ``mae``, ``rmse``, ``mape`` - the :data:`METRIC_COLUMNS`. Populated with the
      model's :class:`Metrics` when it was scored, or ``NaN`` for an excluded /
      unscored model (the column still exists, so the schema is uniform).
    * ``excluded`` - boolean flag, ``True`` for a model that carries an
      ``excluded_reason``.
    * ``excluded_reason`` - the reason string for excluded models, ``None``
      otherwise.

    No sorting is applied: the input order is preserved so the caller controls
    presentation (e.g. sort by ``mae`` for a leaderboard) without this function
    hiding or reordering any model. Because excluded models keep the same metric
    columns (as ``NaN``), a downstream ``sort_values("mae")`` naturally pushes
    them to the end without dropping them.

    Args:
        results: The :class:`ModelResult` objects to tabulate, typically produced
            by :func:`build_model_results`.

    Returns:
        A DataFrame with columns ``[model_name, mae, rmse, mape, excluded,
        excluded_reason]`` and one row per model.
    """
    columns = ["model_name", *METRIC_COLUMNS, "excluded", "excluded_reason"]

    rows: list[dict[str, Any]] = []
    for result in results:
        row: dict[str, Any] = {"model_name": result.model_name}
        metrics = result.metrics
        for col in METRIC_COLUMNS:
            row[col] = float(getattr(metrics, col)) if metrics is not None else np.nan
        row["excluded"] = result.excluded_reason is not None
        row["excluded_reason"] = result.excluded_reason
        rows.append(row)

    # Constructing with an explicit ``columns`` list guarantees the schema (and
    # its order) even when ``results`` is empty, so the table always has the same
    # shape - important for the "same metric columns for every model" property.
    return pd.DataFrame(rows, columns=columns)


def plot_forecast_vs_actual(
    actual,
    results: "Sequence[ModelResult]",
    *,
    index=None,
    ax: "Optional[Axes]" = None,
    title: str = "Forecast vs. actual demand on the Holdout_Set",
) -> "Figure":
    """Plot each model's forecast against the actual Holdout_Set values (R6.4).

    Draws the holdout actuals as a single bold reference line, then overlays one
    line per **scored** model (excluded models have no forecast to plot and are
    skipped, though they remain in :func:`comparison_table`). This is the visual
    companion to the comparison table: the table quantifies accuracy while this
    chart shows *where* each model tracks or misses the real demand across the
    holdout periods.

    Headless-safe: matplotlib is imported lazily and forced onto the
    non-interactive **Agg** backend, and the function never calls ``plt.show()``
    - it returns the :class:`~matplotlib.figure.Figure` for the caller to embed
    (``st.pyplot(fig)`` in the dashboard, inline in the notebook) or save
    (``fig.savefig(...)`` in automation), mirroring the EDA_Module convention.

    Args:
        actual: The Holdout_Set actual demand values (array-like), positionally
            aligned with each model's forecast values.
        results: The :class:`ModelResult` objects whose forecasts to overlay.
            Entries without a usable forecast (excluded/unscored) are skipped.
        index: Optional x-axis values for the holdout periods (e.g. a
            ``DatetimeIndex`` or list of period labels). When ``None``, the
            forecast's own ``index`` is used if present, else a 0-based position
            index is used. Must match the length of ``actual`` when provided.
        ax: Optional existing ``Axes`` to draw on (for composing dashboards). When
            ``None``, a new ``Figure``/``Axes`` pair is created.
        title: Title for the plot.

    Returns:
        The matplotlib ``Figure`` containing the actual-vs-forecast overlay.

    Raises:
        ValueError: If ``index`` is provided but its length does not match
            ``actual``.
    """
    import matplotlib

    # Force a headless-safe backend before pyplot is imported, so this is safe in
    # CI / servers / notebook-execution with no display attached.
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    actual_arr = np.asarray(actual, dtype=float).ravel()

    if index is None:
        x = np.arange(actual_arr.size)
    else:
        x = np.asarray(list(index))
        if x.shape[0] != actual_arr.size:
            raise ValueError(
                "index length must match actual length; got "
                f"{x.shape[0]} and {actual_arr.size}."
            )

    if ax is None:
        figure, axes = plt.subplots(figsize=(12, 6))
    else:
        axes = ax
        figure = axes.figure

    # Actual demand: bold black reference line so forecasts read as overlays.
    axes.plot(
        x,
        actual_arr,
        label="Actual",
        color="black",
        linewidth=2.5,
        zorder=5,
    )

    for result in results:
        forecast = result.forecast
        if forecast is None or getattr(forecast, "values", None) is None:
            # Excluded / unscored model: nothing to draw (still in the table).
            continue
        values = np.asarray(forecast.values, dtype=float).ravel()
        if values.size != actual_arr.size:
            # Misaligned forecast can't be plotted against these actuals; skip it
            # rather than raising, so one bad model never blanks the whole chart.
            continue
        axes.plot(x, values, label=str(result.model_name), linewidth=1.4, alpha=0.9)

    axes.set_title(title)
    axes.set_xlabel("Holdout period")
    axes.set_ylabel("Demand (trip count)")
    if axes.get_legend_handles_labels()[0]:
        axes.legend(loc="upper left", fontsize="small", ncol=2)
    axes.grid(True, alpha=0.3)
    figure.tight_layout()

    return figure


def error_by_period(actual, forecast, buckets) -> pd.DataFrame:
    """Report forecast error per period bucket, not just as one aggregate (R6.5).

    Requirement 6.5 asks the Evaluation_Framework to "report whether forecast
    error varies **across distinct time periods** rather than reporting only an
    aggregate error." A single MAE/RMSE/MAPE over the whole Holdout_Set hides
    *where* a model does well or badly - it might track weekday demand perfectly
    yet blow up on weekends. This function breaks the holdout into caller-defined
    **period buckets** and computes the same :func:`error_metrics` within each
    bucket, so the reader can see the error profile across time rather than a lone
    summary number.

    The three inputs are treated as **positionally aligned** 1-D sequences of the
    same length ``n``: for each holdout period ``i``, ``actual[i]`` is the observed
    demand, ``forecast[i]`` is the prediction, and ``buckets[i]`` is the label of
    the bucket that period belongs to. The ``buckets`` array is therefore a
    *partition* of the holdout periods by construction - every period carries
    exactly one label, so it lands in exactly one bucket and no period is dropped
    or double-counted (design **Correctness Property 10**). Bucket labels are
    arbitrary and hashable: they can be weekday names, month numbers, ``"train"``
    /``"test"`` style tags, integer bucket ids, etc.

    For every distinct bucket the returned table carries a row with:

    * ``bucket``    - the bucket label.
    * ``n_periods`` - how many holdout periods fell in this bucket (a positive
      integer; buckets with no periods do not appear). These counts sum to ``n``,
      which is the machine-checkable statement that the buckets partition the
      holdout.
    * ``mae``, ``rmse``, ``mape`` - the :func:`error_metrics` computed over just
      that bucket's periods. Each is ``>= 0`` (Property 10), and ``rmse >= mae``
      holds within every bucket exactly as it does in aggregate.

    Rows appear in **first-appearance order** of the bucket labels as they occur
    in ``buckets``, so the output order is deterministic and independent of label
    sortability (labels may mix types or be non-orderable). No aggregate row is
    added; the caller already has :func:`error_metrics` for the whole holdout, and
    keeping this table purely per-bucket makes the "error varies across periods"
    comparison direct.

    Args:
        actual: Array-like of observed Holdout_Set demand values (numpy array,
            pandas Series, or list). Coerced to float.
        forecast: Array-like of predicted demand values, positionally aligned with
            ``actual`` and of the same length. Coerced to float.
        buckets: Array-like of bucket labels, positionally aligned with ``actual``
            and of the same length. ``buckets[i]`` names the bucket that holdout
            period ``i`` belongs to; each period thus belongs to exactly one
            bucket (a partition of the holdout periods).

    Returns:
        A :class:`pandas.DataFrame` with columns ``[bucket, n_periods, mae, rmse,
        mape]`` and one row per distinct bucket, in first-appearance order.

    Raises:
        ValueError: If ``actual``, ``forecast`` and ``buckets`` do not all have the
            same length, or if they are empty (no periods to bucket).
    """
    actual_arr = np.asarray(actual, dtype=float).ravel()
    forecast_arr = np.asarray(forecast, dtype=float).ravel()
    # Keep bucket labels as their original (object) values so mixed / non-numeric
    # labels survive; only positional alignment matters.
    bucket_arr = np.asarray(buckets, dtype=object).ravel()

    if not (actual_arr.shape[0] == forecast_arr.shape[0] == bucket_arr.shape[0]):
        raise ValueError(
            "actual, forecast and buckets must all have the same length; got "
            f"{actual_arr.shape[0]}, {forecast_arr.shape[0]} and "
            f"{bucket_arr.shape[0]}."
        )
    if actual_arr.size == 0:
        raise ValueError(
            "actual, forecast and buckets must be non-empty to report "
            "error-by-period."
        )

    columns = ["bucket", "n_periods", *METRIC_COLUMNS]

    # First-appearance order of bucket labels: deterministic and independent of
    # whether the labels are orderable. ``pd.unique`` preserves order of first
    # occurrence and handles arbitrary hashable objects.
    ordered_labels = pd.unique(bucket_arr)

    rows: list[dict[str, Any]] = []
    for label in ordered_labels:
        # Boolean mask of the periods assigned to this bucket. Comparing object
        # arrays elementwise selects exactly the periods whose label matches, so
        # the masks are disjoint and together cover every period (the partition).
        mask = bucket_arr == label
        bucket_metrics = error_metrics(actual_arr[mask], forecast_arr[mask])
        rows.append(
            {
                "bucket": label,
                "n_periods": int(mask.sum()),
                "mae": bucket_metrics.mae,
                "rmse": bucket_metrics.rmse,
                "mape": bucket_metrics.mape,
            }
        )

    return pd.DataFrame(rows, columns=columns)


def select_carry_forward(
    table: pd.DataFrame,
    *,
    model_column: str = "model_name",
    return_justification: bool = False,
) -> Union["list[str]", "tuple[list[str], str]"]:
    """Choose 3-5 models to carry forward for deep explanation (Requirement 6.6).

    Once the comparison is complete (:func:`comparison_table`), Requirement 6.6
    asks the Evaluation_Framework to "identify **between three and five models**
    to carry forward for deep explanation, with a **documented justification**
    based on the reported metrics." This function makes that choice directly from
    the comparison table: it ranks the models by their reported error metrics and
    returns the short-list of names, together (optionally) with a human-readable
    justification string explaining *why* those models were picked.

    Selection policy (based purely on the reported metrics in ``table``):

    * **Scored models are preferred.** A model is "scored" when it has a real
      (non-``NaN``) ``mae`` - i.e. it was trained and evaluated on the
      Holdout_Set. Excluded / unscored models (``NaN`` metrics, see
      :func:`comparison_table`) are used only as fallback fill when there are
      fewer than :data:`CARRY_FORWARD_MIN` scored models, so the set can still
      reach the required minimum of three.
    * **Rank by accuracy.** Scored models are ordered by ``mae`` ascending, with
      ``rmse`` then ``mape`` as deterministic tie-breakers (all "lower is
      better"). Any remaining ties fall back to the model's original row order in
      the table, so the ranking is fully deterministic.
    * **Size the set to [3, 5].** The number carried forward is the count of
      scored models clamped to ``[CARRY_FORWARD_MIN, CARRY_FORWARD_MAX]``: with
      five or more scored models the top five are taken; with three or four, all
      of them; with fewer than three, the best-available excluded models are
      appended (in table order) to reach three. The final count is additionally
      capped by the number of distinct models available, so a table with fewer
      than three rows returns everything it has rather than inventing names.

    This satisfies design **Correctness Property 11**: for any table with at least
    three models the result contains between three and five names, and every
    returned name is present in the table (each name is drawn directly from the
    table's ``model_name`` column, and duplicates are removed while preserving
    order so the count reflects distinct models).

    Args:
        table: A comparison table as produced by :func:`comparison_table`, with a
            ``model_name`` column and the :data:`METRIC_COLUMNS` (``mae``,
            ``rmse``, ``mape``); excluded models carry ``NaN`` metrics.
        model_column: Name of the model-name column. Defaults to ``"model_name"``.
        return_justification: When ``True`` the return value is a
            ``(names, justification)`` tuple; when ``False`` (the default, and the
            shape the design's ``select_carry_forward(table) -> list[str]``
            specifies) only the list of names is returned.

    Returns:
        By default, the list of carried-forward model names (3-5 names for a table
        with at least three models, in ranked order best-first). When
        ``return_justification`` is ``True``, a ``(names, justification)`` tuple
        whose second element is a human-readable string documenting the metric
        basis for the selection.

    Raises:
        KeyError: If ``model_column`` is absent from ``table``.
    """
    if model_column not in table.columns:
        raise KeyError(
            f"Model-name column '{model_column}' not found in table. "
            f"Available columns: {list(table.columns)}."
        )

    # Work on a positional copy so original row order is a stable final tie-break.
    working = table.reset_index(drop=True)
    working = working.assign(_order=np.arange(len(working)))

    # A model is "scored" when it has a real (non-NaN) MAE - i.e. it was trained
    # and evaluated. ``mae`` may be absent entirely for a malformed table; treat a
    # missing column as "nothing scored" rather than raising.
    if "mae" in working.columns:
        scored_mask = working["mae"].notna()
    else:
        scored_mask = pd.Series(False, index=working.index)

    scored = working.loc[scored_mask]
    unscored = working.loc[~scored_mask]

    # Rank scored models by accuracy: MAE, then RMSE, then MAPE (all lower-better),
    # with original order as the final deterministic tie-break. Only sort by the
    # metric columns that actually exist so a partial table still works.
    sort_keys = [c for c in ("mae", "rmse", "mape") if c in scored.columns]
    sort_keys.append("_order")
    ranked_scored = scored.sort_values(sort_keys, kind="stable")

    # Fallback fill (excluded/unscored models) keeps the table's own order.
    ranked_unscored = unscored.sort_values("_order", kind="stable")

    # Ordered candidate names, scored-first, de-duplicated while preserving order
    # so the count reflects distinct models and every name is present in the table.
    ordered_names: list[str] = []
    seen: set = set()
    for name in [*ranked_scored[model_column], *ranked_unscored[model_column]]:
        if name not in seen:
            seen.add(name)
            ordered_names.append(name)

    n_available = len(ordered_names)
    n_scored = int(scored_mask.sum())

    # Size the carry-forward set: clamp the scored count into [MIN, MAX], then cap
    # by however many distinct models actually exist (so a <3-row table returns
    # all it has instead of fabricating names).
    target = min(CARRY_FORWARD_MAX, max(CARRY_FORWARD_MIN, n_scored))
    count = min(target, n_available)

    selected = ordered_names[:count]

    if not return_justification:
        return selected

    justification = _carry_forward_justification(
        selected, ranked_scored, model_column, n_scored, count
    )
    return selected, justification


def _carry_forward_justification(
    selected: "list[str]",
    ranked_scored: pd.DataFrame,
    model_column: str,
    n_scored: int,
    count: int,
) -> str:
    """Build the human-readable justification for a carry-forward selection.

    Documents the metric basis (Requirement 6.6): how many models were scored, the
    ranking rule, the chosen models with their reported MAE (best-first), and a
    note when excluded models had to be included to reach the minimum of three.
    """
    if count == 0:
        return "No models were available in the comparison table to carry forward."

    scored_names = list(ranked_scored[model_column])
    lines: list[str] = [
        f"Carrying forward {count} model(s) for deep explanation "
        f"(Requirement 6.6 target: {CARRY_FORWARD_MIN}-{CARRY_FORWARD_MAX}). "
        f"{n_scored} model(s) were scored on the Holdout_Set; models are ranked "
        "by MAE ascending, with RMSE then MAPE as tie-breakers (lower is better)."
    ]

    mae_lookup = (
        dict(zip(ranked_scored[model_column], ranked_scored["mae"]))
        if "mae" in ranked_scored.columns
        else {}
    )
    for rank, name in enumerate(selected, start=1):
        if name in scored_names:
            mae_val = mae_lookup.get(name)
            mae_txt = f"MAE={mae_val:.4g}" if mae_val is not None else "scored"
            lines.append(f"  {rank}. {name} ({mae_txt})")
        else:
            lines.append(
                f"  {rank}. {name} (included as fallback: not scored, added to "
                f"reach the minimum of {CARRY_FORWARD_MIN})"
            )

    if count < n_scored:
        lines.append(
            f"Selected the top {count} of {n_scored} scored models by reported "
            "error metrics."
        )

    return "\n".join(lines)
