"""EDA_Module - exploratory analysis of the demand series (Requirement 3).

The EDA_Module produces the plots and statistics that let a reader understand the
demand series *before* any modeling: its trend, seasonality, stationarity,
autocorrelation, anomalies, and correlations. Crucially, Requirement 3.7 requires
that **every chart and statistical output be accompanied by a plain-language
interpretation** of what it shows - a chart alone is not enough.

To make that requirement structural rather than a matter of discipline, this
module returns a small :class:`EDAResult` dataclass that pairs the matplotlib
``Figure`` with an ``interpretation`` string. Every EDA function in this module -
this task's :func:`plot_demand_series` and the later stationarity/autocorrelation
(task 6.2) and anomaly/correlation (task 6.3) functions - returns the *same*
shape, so a figure can never travel without its interpretation.

Design references:
- Components and Interfaces -> EDA_Module (`src/eda.py`); signature
  ``plot_demand_series(series, scope)``.
- Requirements 3.1 (time-series plot of demand at the Time_Grain over the
  Analysis_Window) and 3.7 (plain-language interpretation for each output).

Headless / non-interactive note:
    This module never calls ``plt.show()`` and forces the non-interactive **Agg**
    backend at import time, so it is safe to import and run in headless
    environments (CI, GitHub Actions, notebook execution on a server). Callers
    receive the ``Figure``/``Axes`` objects and decide how to render or embed
    them (``st.pyplot(fig)`` in the dashboard, inline display in the notebook,
    ``fig.savefig(...)`` in automation).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import matplotlib

# Force a non-interactive, headless-safe backend before pyplot is imported so the
# module works in CI / servers / notebook-execution with no display attached.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  (import after backend selection)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf  # noqa: E402
from statsmodels.tsa.seasonal import seasonal_decompose  # noqa: E402
from statsmodels.tsa.stattools import acf, adfuller, pacf  # noqa: E402

from src.preparation import (  # noqa: E402
    DEMAND_COLUMN,
    PERIOD_COLUMN,
    REGION_COLUMN,
)

if TYPE_CHECKING:  # pragma: no cover - imported only for type checking
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from src.config import ScopeConfig


@dataclass
class EDAResult:
    """A paired EDA output: a matplotlib figure plus its plain-language reading.

    Requirement 3.7 requires every chart and statistical output to carry a
    plain-language interpretation. Returning this dataclass makes that pairing
    structural: an EDA function hands back the ``figure`` (embeddable in the
    notebook or the Streamlit dashboard) *together with* the ``interpretation``
    describing what the figure shows, so the two never get separated.

    This is the shared shape every EDA function in this module returns, so the
    later stationarity (task 6.2) and anomaly/correlation (task 6.3) functions
    reuse the exact same contract. When a function also computes structured
    statistics (e.g. an ADF statistic and p-value), it attaches them to
    :attr:`stats` so callers can consume the raw numbers as well as the prose.

    Attributes:
        figure: The matplotlib ``Figure`` for the chart. The module never calls
            ``plt.show()``; the caller renders or saves it.
        interpretation: Plain-language description of what the figure shows -
            trend, relative magnitude across regions, span, and any caveats
            (Requirement 3.7).
        axes: The primary ``Axes`` of the figure, provided for convenience so
            callers can further annotate or embed it. Optional.
        stats: Optional mapping of any structured statistics the function
            computed alongside the chart (used by later EDA functions).
    """

    figure: "Figure"
    interpretation: str
    axes: "Optional[Axes]" = None
    stats: Optional[dict] = None


def _format_int(value: float) -> str:
    """Format a numeric value as a thousands-separated integer string."""
    return f"{int(round(value)):,}"


def plot_demand_series(
    series: pd.DataFrame,
    scope: "ScopeConfig",
    *,
    period_column: str = PERIOD_COLUMN,
    region_column: str = REGION_COLUMN,
    demand_column: str = DEMAND_COLUMN,
    show_total: bool = True,
    ax: "Optional[Axes]" = None,
) -> EDAResult:
    """Plot demand over time at the Time_Grain, with a plain-language reading.

    Produces the foundational EDA chart (Requirement 3.1): a time-series plot of
    demand at the defined Time_Grain over the Analysis_Window, with **one line per
    region** so the relative magnitude and shape of each borough's demand are
    directly comparable. When ``show_total`` is set (default) and there is more
    than one region, an aggregate "All regions (total)" line is overlaid so the
    system-wide trend is visible alongside the per-region detail.

    The x-axis is ``period`` (at the Time_Grain), the y-axis is ``demand`` (trip
    count). The returned :class:`EDAResult` carries the ``Figure`` (for embedding
    in the notebook or dashboard) together with an ``interpretation`` string that
    describes, in plain language, what the plot shows - its time span, the busiest
    and quietest regions by total volume, and the overall direction of demand -
    satisfying Requirement 3.7.

    This function is headless-safe: it never calls ``plt.show()`` and relies on
    the module-level Agg backend, so it is importable and runnable in CI and on
    servers with no display.

    Args:
        series: A :data:`~src.preparation.DemandSeries` (long format) with at
            least ``period``, ``region`` and ``demand`` columns. Typically the
            zero-filled output of the preparation pipeline, so every
            ``(period, region)`` in the Analysis_Window is present.
        scope: The :class:`~src.config.ScopeConfig` supplying the Time_Grain and
            Analysis_Window, used to label the chart and frame the interpretation.
        period_column: Name of the period column. Defaults to ``"period"``.
        region_column: Name of the region column. Defaults to ``"region"``.
        demand_column: Name of the demand column. Defaults to ``"demand"``.
        show_total: When ``True`` (default) and more than one region is present,
            overlay an aggregate total-demand line across all regions.
        ax: Optional existing ``Axes`` to draw on (for composing dashboards). When
            ``None``, a new ``Figure``/``Axes`` pair is created.

    Returns:
        An :class:`EDAResult` whose ``figure`` is the time-series plot and whose
        ``interpretation`` explains it in plain language (Requirement 3.7).

    Raises:
        KeyError: If ``series`` is missing any of the period/region/demand columns.
    """
    for col in (period_column, region_column, demand_column):
        if col not in series.columns:
            raise KeyError(
                f"Column '{col}' not found in series. "
                f"Available columns: {list(series.columns)}."
            )

    # Work on a normalized copy: coerce period to datetime and sort so lines are
    # drawn left-to-right in time regardless of the input row order.
    work = series[[period_column, region_column, demand_column]].copy()
    work[period_column] = pd.to_datetime(work[period_column], errors="coerce")
    work = work.dropna(subset=[period_column]).sort_values(period_column, kind="stable")

    # Create or reuse the drawing surface. A wide figure suits a long daily span.
    if ax is None:
        figure, axes = plt.subplots(figsize=(12, 6))
    else:
        axes = ax
        figure = axes.figure

    grain_label = str(scope.time_grain).capitalize()

    # Distinct regions, ordered by total demand (busiest first) so the legend and
    # interpretation agree on which borough dominates.
    region_totals = (
        work.groupby(region_column)[demand_column].sum().sort_values(ascending=False)
    )
    regions = list(region_totals.index)

    for region in regions:
        region_df = work[work[region_column] == region]
        # Collapse any duplicate periods so each region is a single clean line.
        line = region_df.groupby(period_column, as_index=True)[demand_column].sum()
        axes.plot(line.index, line.values, label=str(region), linewidth=1.4)

    total_line = None
    if show_total and len(regions) > 1:
        total_line = work.groupby(period_column)[demand_column].sum()
        axes.plot(
            total_line.index,
            total_line.values,
            label="All regions (total)",
            color="black",
            linewidth=2.2,
            linestyle="--",
        )

    axes.set_title(
        f"{grain_label} ride-hailing demand by {scope.geographic_grain} "
        f"over the Analysis_Window"
    )
    axes.set_xlabel(f"Period ({scope.time_grain})")
    axes.set_ylabel("Demand (trip count)")
    # Only add a legend when at least one line was drawn (avoids a matplotlib
    # "no artists with labels" warning on an empty series).
    if axes.get_legend_handles_labels()[0]:
        axes.legend(loc="upper left", fontsize="small", ncol=2)
    axes.grid(True, alpha=0.3)
    figure.autofmt_xdate()
    figure.tight_layout()

    interpretation = _build_series_interpretation(
        work,
        region_totals=region_totals,
        total_line=total_line,
        scope=scope,
        period_column=period_column,
        demand_column=demand_column,
    )

    return EDAResult(figure=figure, interpretation=interpretation, axes=axes)


def _build_series_interpretation(
    work: pd.DataFrame,
    *,
    region_totals: pd.Series,
    total_line: Optional[pd.Series],
    scope: "ScopeConfig",
    period_column: str,
    demand_column: str,
) -> str:
    """Compose the plain-language reading of the demand time-series plot (R3.7).

    Describes the span covered, the number of regions, which region carries the
    most and least total demand, and the overall direction of demand (comparing
    the first and last thirds of the series). Kept purely descriptive and derived
    from the data so nothing is asserted that the chart does not show.
    """
    if work.empty:
        return (
            "The demand series is empty over the Analysis_Window "
            f"({scope.window_start} to {scope.window_end}); there is nothing to plot. "
            "This usually means no records survived preparation for this window."
        )

    span_start = work[period_column].min()
    span_end = work[period_column].max()
    n_periods = work[period_column].nunique()
    n_regions = int(region_totals.shape[0])

    parts: list[str] = []
    parts.append(
        f"This chart shows {scope.time_grain} demand (trip count) across "
        f"{n_regions} {scope.geographic_grain}(s) over {n_periods} periods, "
        f"spanning {span_start.date()} to {span_end.date()}."
    )

    if n_regions >= 1:
        busiest = region_totals.index[0]
        busiest_total = region_totals.iloc[0]
        quietest = region_totals.index[-1]
        quietest_total = region_totals.iloc[-1]
        if n_regions > 1:
            parts.append(
                f"{busiest} carries the most demand overall "
                f"({_format_int(busiest_total)} trips), while {quietest} carries "
                f"the least ({_format_int(quietest_total)} trips)."
            )
        else:
            parts.append(
                f"{busiest} accounts for {_format_int(busiest_total)} trips over "
                "the window."
            )

    # Overall direction: compare the mean of the first third to the last third of
    # the aggregate (or single-region) demand line.
    reference = (
        total_line
        if total_line is not None
        else work.groupby(period_column)[demand_column].sum()
    )
    reference = reference.sort_index()
    if len(reference) >= 3:
        third = max(1, len(reference) // 3)
        early = reference.iloc[:third].mean()
        late = reference.iloc[-third:].mean()
        if early > 0:
            pct = (late - early) / early * 100.0
        else:
            pct = 0.0
        if abs(pct) < 5:
            direction = "roughly flat"
        elif pct > 0:
            direction = f"trending up (about {pct:.0f}% higher)"
        else:
            direction = f"trending down (about {abs(pct):.0f}% lower)"
        parts.append(
            f"Comparing the start and end of the window, total demand is {direction} "
            "from the first third to the last third of the period."
        )

    peak_period = reference.idxmax()
    peak_value = reference.max()
    parts.append(
        f"Peak total demand of {_format_int(peak_value)} trips occurs around "
        f"{pd.Timestamp(peak_period).date()}. Read alongside seasonality and "
        "anomaly analysis for the drivers behind these movements."
    )

    return " ".join(parts)


# =============================================================================
# Stationarity and autocorrelation analysis (task 6.2; Requirements 3.2-3.4, 3.7)
# =============================================================================
#
# The three functions below wrap statsmodels' time-series diagnostics -
# ``seasonal_decompose`` (R3.2), ``adfuller`` (R3.3) and ``plot_acf``/``plot_pacf``
# (R3.4) - and, per Requirement 3.7, pair every output with a plain-language
# interpretation. They all operate on a *single* demand series: either the demand
# of one region or the aggregate total across all regions. Because the rest of
# the pipeline speaks in the long-format :data:`~src.preparation.DemandSeries`
# (one row per ``(period, region)``), these functions accept *either* a ready
# ``pandas.Series`` indexed by period *or* a DemandSeries DataFrame, which they
# reduce to one period-indexed series by summing demand across regions per period
# (the system-wide total). This keeps callers from having to reshape the series
# themselves before every diagnostic.


def _reduce_to_series(
    series,
    *,
    period_column: str = PERIOD_COLUMN,
    region_column: str = REGION_COLUMN,
    demand_column: str = DEMAND_COLUMN,
) -> pd.Series:
    """Reduce a demand input to a single period-indexed ``pandas.Series``.

    The stationarity/autocorrelation diagnostics all analyse one demand series.
    This helper accepts the two shapes callers naturally have and normalises them:

    * a ``pandas.Series`` - assumed already to be a single demand series (its
      index is treated as the period axis); it is coerced to float, has its
      missing values dropped, and is sorted by index; or
    * a long-format :data:`~src.preparation.DemandSeries` DataFrame - reduced to
      one series by **summing demand across regions within each period**, giving
      the aggregate system-wide demand at the Time_Grain.

    Args:
        series: A ``pandas.Series`` (single series) or a DemandSeries DataFrame.
        period_column: Period column name used when ``series`` is a DataFrame.
        region_column: Region column name (accepted for interface symmetry; the
            reduction simply sums across whatever regions are present).
        demand_column: Demand column name used when ``series`` is a DataFrame.

    Returns:
        A float ``pandas.Series`` sorted by period, with nulls dropped. When the
        input was a DataFrame the index is a ``DatetimeIndex`` of periods and the
        values are total demand per period.

    Raises:
        KeyError: If ``series`` is a DataFrame missing the period/demand columns.
        TypeError: If ``series`` is neither a Series nor a DataFrame.
    """
    if isinstance(series, pd.Series):
        reduced = pd.to_numeric(series, errors="coerce").dropna().sort_index()
        return reduced.astype(float)

    if isinstance(series, pd.DataFrame):
        for col in (period_column, demand_column):
            if col not in series.columns:
                raise KeyError(
                    f"Column '{col}' not found in series. "
                    f"Available columns: {list(series.columns)}."
                )
        work = series[[period_column, demand_column]].copy()
        work[period_column] = pd.to_datetime(work[period_column], errors="coerce")
        work = work.dropna(subset=[period_column])
        reduced = (
            work.groupby(period_column)[demand_column].sum().sort_index().astype(float)
        )
        return reduced

    raise TypeError(
        "series must be a pandas Series (single demand series) or a DemandSeries "
        f"DataFrame; got {type(series).__name__}."
    )


@dataclass
class ADFResult:
    """Result of an Augmented Dickey-Fuller stationarity test (Requirement 3.3).

    The ADF test has a *statistical* output rather than a chart, so - unlike the
    figure-bearing :class:`EDAResult` - this dataclass carries the numbers the
    requirement calls out (the test ``statistic`` and ``p_value``) alongside the
    plain-language ``interpretation`` mandated by Requirement 3.7. The supporting
    ``critical_values``, ``used_lag`` and ``n_obs`` from statsmodels are retained
    so a reader can see the full picture, and :attr:`stats` mirrors the headline
    numbers as a plain mapping for callers that consume results uniformly.

    Attributes:
        statistic: The ADF test statistic (more negative => stronger evidence
            against a unit root).
        p_value: The test's p-value. By convention ``p_value < 0.05`` is read as
            evidence that the series is stationary.
        stationary: Convenience boolean, ``True`` when ``p_value < alpha``.
        alpha: The significance threshold used to decide ``stationary``.
        used_lag: Number of lags statsmodels selected for the test.
        n_obs: Number of observations used after lag selection.
        critical_values: Mapping of confidence level (``"1%"``/``"5%"``/``"10%"``)
            to the corresponding critical value of the statistic.
        interpretation: Plain-language reading of stationarity (Requirement 3.7).
        stats: Flat mapping of the headline numbers for uniform consumption.
    """

    statistic: float
    p_value: float
    stationary: bool
    alpha: float
    used_lag: int
    n_obs: int
    critical_values: dict
    interpretation: str
    stats: Optional[dict] = None


def seasonal_decompose_demand(
    series,
    *,
    period: int = 7,
    model: str = "additive",
    period_column: str = PERIOD_COLUMN,
    region_column: str = REGION_COLUMN,
    demand_column: str = DEMAND_COLUMN,
) -> EDAResult:
    """Decompose a demand series into trend, seasonal and residual (R3.2, R3.7).

    Splits the demand series into three additive (or multiplicative) components -
    a slow-moving ``trend``, a repeating ``seasonal`` cycle, and the ``residual``
    left over - using :func:`statsmodels.tsa.seasonal.seasonal_decompose`. This
    is the seasonal decomposition Requirement 3.2 asks for; the returned
    :class:`EDAResult` carries the four-panel figure (observed + the three
    components) together with a plain-language interpretation (Requirement 3.7)
    describing the trend direction, the size of the weekly swing, and how much
    variation the components explain.

    The input may be a single-region demand series or the aggregate total. When a
    long-format :data:`~src.preparation.DemandSeries` DataFrame is passed it is
    first reduced to the system-wide total by summing demand across regions within
    each period (see :func:`_reduce_to_series`).

    Args:
        series: A period-indexed ``pandas.Series`` or a DemandSeries DataFrame.
        period: The number of observations in one seasonal cycle. For daily data
            with weekly seasonality the sensible default is ``7``; expose it so
            other grains/cycles can be analysed.
        model: ``"additive"`` (default) or ``"multiplicative"``, forwarded to
            statsmodels.
        period_column: Period column name (used when ``series`` is a DataFrame).
        region_column: Region column name (used when ``series`` is a DataFrame).
        demand_column: Demand column name (used when ``series`` is a DataFrame).

    Returns:
        An :class:`EDAResult` whose ``figure`` is the four-panel decomposition
        plot, whose ``interpretation`` explains it in plain language, and whose
        ``stats`` holds the ``observed``, ``trend``, ``seasonal`` and ``residual``
        component Series plus the ``period`` and ``model`` used.

    Raises:
        KeyError: If a DataFrame ``series`` lacks the period/demand columns.
        ValueError: If ``period < 2`` or the series has fewer than ``2 * period``
            observations (statsmodels cannot estimate a full cycle otherwise).
    """
    if period < 2:
        raise ValueError(f"period must be >= 2 to decompose seasonality; got {period}.")

    reduced = _reduce_to_series(
        series,
        period_column=period_column,
        region_column=region_column,
        demand_column=demand_column,
    )
    n = int(reduced.shape[0])
    if n < 2 * period:
        raise ValueError(
            f"Need at least 2 full seasonal cycles ({2 * period} observations for "
            f"period={period}) to decompose; got {n}."
        )

    result = seasonal_decompose(reduced, model=model, period=period)

    figure = result.plot()
    figure.set_size_inches(12, 9)
    figure.suptitle(
        f"Seasonal decomposition of demand ({model}, period={period})", y=1.01
    )
    figure.tight_layout()
    axes = figure.axes[0] if figure.axes else None

    stats = {
        "observed": result.observed,
        "trend": result.trend,
        "seasonal": result.seasonal,
        "residual": result.resid,
        "period": period,
        "model": model,
    }

    interpretation = _build_decomposition_interpretation(result, period=period, model=model)

    return EDAResult(figure=figure, interpretation=interpretation, axes=axes, stats=stats)


def _build_decomposition_interpretation(result, *, period: int, model: str) -> str:
    """Compose the plain-language reading of the seasonal decomposition (R3.7).

    Describes the overall trend direction (first vs last estimated trend value),
    the magnitude of the repeating seasonal swing, and the standard STL "strength"
    measures for trend and seasonality (0 = none, ~1 = dominant), so a reader
    learns what the four panels mean without reading the axes.
    """
    trend = result.trend.dropna()
    seasonal = result.seasonal.dropna()
    resid = result.resid.dropna()

    parts: list[str] = [
        f"This {model} decomposition splits demand into a trend, a repeating "
        f"seasonal cycle of period {period}, and a residual."
    ]

    # Trend direction from the first to the last estimated trend value.
    if len(trend) >= 2:
        start, end = float(trend.iloc[0]), float(trend.iloc[-1])
        if start != 0:
            pct = (end - start) / abs(start) * 100.0
        else:
            pct = 0.0
        if abs(pct) < 5:
            direction = "roughly flat"
        elif pct > 0:
            direction = f"rising (about {pct:.0f}% higher across the window)"
        else:
            direction = f"falling (about {abs(pct):.0f}% lower across the window)"
        parts.append(f"The trend component is {direction}.")

    # Size of one seasonal swing (peak-to-trough of the repeating pattern).
    if len(seasonal) >= 1:
        swing = float(seasonal.max() - seasonal.min())
        parts.append(
            f"The seasonal component repeats every {period} periods with a "
            f"peak-to-trough swing of about {swing:,.0f} trips."
        )

    # STL strength measures on the common (non-NaN) support.
    common = pd.concat(
        {"trend": result.trend, "seasonal": result.seasonal, "resid": result.resid},
        axis=1,
    ).dropna()
    if len(common) >= 2:
        resid_var = float(common["resid"].var())
        deseasonal_var = float((common["resid"] + common["seasonal"]).var())
        detrend_var = float((common["resid"] + common["trend"]).var())
        seasonal_strength = (
            max(0.0, 1.0 - resid_var / deseasonal_var) if deseasonal_var > 0 else 0.0
        )
        trend_strength = (
            max(0.0, 1.0 - resid_var / detrend_var) if detrend_var > 0 else 0.0
        )
        parts.append(
            f"Seasonal strength is {seasonal_strength:.2f} and trend strength is "
            f"{trend_strength:.2f} (0 = negligible, ~1 = dominant); "
            + (
                "seasonality dominates the signal."
                if seasonal_strength >= trend_strength
                else "the trend dominates the signal."
            )
        )

    if len(resid) >= 2 and len(result.observed.dropna()) >= 2:
        resid_share = float(resid.std()) / (float(result.observed.dropna().std()) or 1.0)
        parts.append(
            f"The residual has a spread about {resid_share:.0%} of the observed "
            "series' spread; larger residuals point to noise or anomalies the "
            "trend and seasonal terms do not explain."
        )

    return " ".join(parts)


def adf_test(
    series,
    *,
    alpha: float = 0.05,
    period_column: str = PERIOD_COLUMN,
    region_column: str = REGION_COLUMN,
    demand_column: str = DEMAND_COLUMN,
) -> ADFResult:
    """Assess stationarity with the Augmented Dickey-Fuller test (R3.3, R3.7).

    Runs :func:`statsmodels.tsa.stattools.adfuller` on the demand series and
    reports the test ``statistic`` and ``p_value`` that Requirement 3.3 calls for,
    wrapped in an :class:`ADFResult` with a plain-language interpretation
    (Requirement 3.7). The ADF null hypothesis is that the series has a unit root
    (is non-stationary); by the usual convention a ``p_value`` below ``alpha``
    (default 0.05) rejects that null and the series is read as **stationary**,
    while a larger p-value means it is **non-stationary** and likely needs
    differencing before an ARIMA-family model is fitted.

    The input may be a single-region series or the aggregate total; a long-format
    :data:`~src.preparation.DemandSeries` DataFrame is reduced to the system-wide
    total per period first (see :func:`_reduce_to_series`).

    Args:
        series: A period-indexed ``pandas.Series`` or a DemandSeries DataFrame.
        alpha: Significance threshold for the stationarity decision. Defaults to
            ``0.05``.
        period_column: Period column name (used when ``series`` is a DataFrame).
        region_column: Region column name (used when ``series`` is a DataFrame).
        demand_column: Demand column name (used when ``series`` is a DataFrame).

    Returns:
        An :class:`ADFResult` carrying the ``statistic``, ``p_value``,
        ``stationary`` flag, ``critical_values`` and a plain-language
        ``interpretation``.

    Raises:
        KeyError: If a DataFrame ``series`` lacks the period/demand columns.
        ValueError: If the series has too few observations to run the test.
    """
    reduced = _reduce_to_series(
        series,
        period_column=period_column,
        region_column=region_column,
        demand_column=demand_column,
    )
    if reduced.shape[0] < 4:
        raise ValueError(
            "The ADF test needs at least a handful of observations; "
            f"got {reduced.shape[0]}."
        )

    statistic, p_value, used_lag, n_obs, crit_values, _icbest = adfuller(
        reduced.to_numpy()
    )
    stationary = bool(p_value < alpha)

    interpretation = _build_adf_interpretation(
        statistic=statistic,
        p_value=p_value,
        stationary=stationary,
        alpha=alpha,
        critical_values=crit_values,
    )

    stats = {
        "statistic": float(statistic),
        "p_value": float(p_value),
        "stationary": stationary,
        "used_lag": int(used_lag),
        "n_obs": int(n_obs),
    }

    return ADFResult(
        statistic=float(statistic),
        p_value=float(p_value),
        stationary=stationary,
        alpha=alpha,
        used_lag=int(used_lag),
        n_obs=int(n_obs),
        critical_values={k: float(v) for k, v in crit_values.items()},
        interpretation=interpretation,
        stats=stats,
    )


def _build_adf_interpretation(
    *,
    statistic: float,
    p_value: float,
    stationary: bool,
    alpha: float,
    critical_values: dict,
) -> str:
    """Compose the plain-language reading of the ADF test (Requirement 3.7)."""
    crit = ", ".join(f"{k}: {float(v):.3f}" for k, v in critical_values.items())
    parts = [
        f"Augmented Dickey-Fuller statistic is {statistic:.3f} with a p-value of "
        f"{p_value:.4f} (critical values -> {crit}).",
        "The ADF null hypothesis is that the series has a unit root (is "
        "non-stationary).",
    ]
    if stationary:
        parts.append(
            f"Because the p-value is below {alpha:g}, we reject that null: the "
            "demand series looks stationary, so its mean and variance are stable "
            "over time and it can be modelled without further differencing."
        )
    else:
        parts.append(
            f"Because the p-value is at or above {alpha:g}, we cannot reject that "
            "null: the demand series looks non-stationary (trend and/or changing "
            "variance), so differencing or de-trending is advisable before fitting "
            "an ARIMA-family model."
        )
    return " ".join(parts)


def acf_pacf(
    series,
    *,
    lags: Optional[int] = None,
    period_column: str = PERIOD_COLUMN,
    region_column: str = REGION_COLUMN,
    demand_column: str = DEMAND_COLUMN,
) -> EDAResult:
    """Plot ACF and PACF of the demand series to inform model order (R3.4, R3.7).

    Produces the autocorrelation (ACF) and partial autocorrelation (PACF) plots
    Requirement 3.4 asks for, using :func:`statsmodels.graphics.tsaplots.plot_acf`
    and :func:`~statsmodels.graphics.tsaplots.plot_pacf`. These two correlograms
    are the standard way to read candidate ARIMA orders: a sharp cut-off in the
    PACF after lag *p* suggests an AR(*p*) term, a sharp cut-off in the ACF after
    lag *q* suggests an MA(*q*) term, and a spike at the seasonal lag (7 for daily
    data with weekly seasonality) flags a seasonal component. The returned
    :class:`EDAResult` carries the two-panel figure together with a plain-language
    interpretation (Requirement 3.7) naming the significant lags and the order
    hints they imply.

    The input may be a single-region series or the aggregate total; a long-format
    :data:`~src.preparation.DemandSeries` DataFrame is reduced to the system-wide
    total per period first (see :func:`_reduce_to_series`).

    Args:
        series: A period-indexed ``pandas.Series`` or a DemandSeries DataFrame.
        lags: Number of lags to display. When ``None`` a sensible default is
            chosen from the series length (capped below half the sample size,
            which the PACF estimator requires).
        period_column: Period column name (used when ``series`` is a DataFrame).
        region_column: Region column name (used when ``series`` is a DataFrame).
        demand_column: Demand column name (used when ``series`` is a DataFrame).

    Returns:
        An :class:`EDAResult` whose ``figure`` holds the ACF (top) and PACF
        (bottom) plots, whose ``interpretation`` reads them in plain language, and
        whose ``stats`` holds the ``acf``/``pacf`` value arrays, the ``lags`` used
        and the significance ``confidence`` bound.

    Raises:
        KeyError: If a DataFrame ``series`` lacks the period/demand columns.
        ValueError: If the series is too short to estimate any autocorrelation.
    """
    reduced = _reduce_to_series(
        series,
        period_column=period_column,
        region_column=region_column,
        demand_column=demand_column,
    )
    n = int(reduced.shape[0])
    if n < 4:
        raise ValueError(
            f"Need at least 4 observations to estimate autocorrelation; got {n}."
        )

    # PACF (Yule-Walker) requires lags < n/2; cap accordingly and keep >= 1.
    max_allowed = max(1, n // 2 - 1)
    if lags is None:
        lags = min(40, max_allowed)
    else:
        lags = max(1, min(int(lags), max_allowed))

    x = reduced.to_numpy()

    figure, (ax_acf, ax_pacf) = plt.subplots(2, 1, figsize=(12, 8))
    plot_acf(x, ax=ax_acf, lags=lags)
    ax_acf.set_title("Autocorrelation (ACF)")
    ax_acf.set_xlabel("Lag")
    ax_acf.set_ylabel("Correlation")
    plot_pacf(x, ax=ax_pacf, lags=lags, method="ywm")
    ax_pacf.set_title("Partial autocorrelation (PACF)")
    ax_pacf.set_xlabel("Lag")
    ax_pacf.set_ylabel("Partial correlation")
    figure.tight_layout()

    acf_values = acf(x, nlags=lags, fft=True)
    pacf_values = pacf(x, nlags=lags, method="ywm")
    confidence = 1.96 / np.sqrt(n)

    stats = {
        "acf": acf_values,
        "pacf": pacf_values,
        "lags": lags,
        "confidence": float(confidence),
    }

    interpretation = _build_acf_pacf_interpretation(
        acf_values=acf_values,
        pacf_values=pacf_values,
        confidence=confidence,
        lags=lags,
    )

    return EDAResult(figure=figure, interpretation=interpretation, axes=ax_acf, stats=stats)


def _build_acf_pacf_interpretation(
    *,
    acf_values: "np.ndarray",
    pacf_values: "np.ndarray",
    confidence: float,
    lags: int,
) -> str:
    """Compose the plain-language reading of the ACF/PACF plots (R3.7).

    Names the lags whose correlation exceeds the ~95% significance band and turns
    them into order hints: the first insignificant PACF lag suggests an AR order,
    the first insignificant ACF lag an MA order, and a spike at lag 7 flags weekly
    seasonality for daily data.
    """
    # Significant lags, skipping lag 0 (always 1.0 by construction).
    sig_acf = [i for i in range(1, len(acf_values)) if abs(acf_values[i]) > confidence]
    sig_pacf = [i for i in range(1, len(pacf_values)) if abs(pacf_values[i]) > confidence]

    parts = [
        f"ACF and PACF are shown out to lag {lags}; bars beyond the shaded band "
        f"(about +/-{confidence:.2f}) are statistically significant."
    ]

    if sig_acf:
        shown = ", ".join(str(i) for i in sig_acf[:8])
        parts.append(
            f"Significant ACF lags: {shown}"
            + (" ..." if len(sig_acf) > 8 else "")
            + ". A slow ACF decay indicates non-stationarity or a strong trend, "
            "while a sharp cut-off after lag q suggests an MA(q) term."
        )
    else:
        parts.append(
            "No ACF lag is significant, which is consistent with (near) white "
            "noise - little linear structure left to model."
        )

    if sig_pacf:
        shown = ", ".join(str(i) for i in sig_pacf[:8])
        parts.append(
            f"Significant PACF lags: {shown}"
            + (" ..." if len(sig_pacf) > 8 else "")
            + f". The PACF cutting off after lag {sig_pacf[0]} points to an "
            f"AR({sig_pacf[0]}) starting order."
        )
    else:
        parts.append("No PACF lag is significant.")

    # Weekly-seasonality flag for daily data.
    if len(acf_values) > 7 and abs(acf_values[7]) > confidence:
        parts.append(
            "A clear spike at lag 7 indicates weekly seasonality; consider a "
            "seasonal term (period 7) in the model."
        )

    return " ".join(parts)


# =============================================================================
# Anomaly detection and correlation analysis (task 6.3; Requirements 3.5-3.7)
# =============================================================================
#
# The two functions below round out the EDA module's diagnostics:
#
# * :func:`detect_anomalies` (Requirement 3.5) scans the demand series for
#   spikes and drops - periods where demand departs sharply from its recent
#   local level - and returns a list of :class:`Anomaly` records, each naming the
#   affected period and carrying a plain-language description of the observation
#   (Requirement 3.7). It uses a **robust** rule (rolling median + median
#   absolute deviation, MAD) rather than a plain mean/standard-deviation z-score,
#   because a handful of large spikes would inflate the mean and standard
#   deviation and mask the very outliers we are looking for; the median and MAD
#   are resistant to that.
#
# * :func:`demand_correlations` (Requirement 3.6) reports how demand co-moves
#   with candidate explanatory variables (calendar effects, weather proxies, or
#   any exogenous series a caller supplies), returning a correlation table with
#   values in ``[-1, 1]`` and a plain-language interpretation naming the
#   strongest positive and negative relationships (Requirement 3.7).
#
# Both accept the same demand shapes as the rest of the module (a period-indexed
# ``pandas.Series`` or a long-format DemandSeries DataFrame reduced to the
# system-wide total per period) via :func:`_reduce_to_series`, and both are
# headless-safe.


@dataclass
class Anomaly:
    """A single anomalous period in the demand series (Requirements 3.5, 3.7).

    Requirement 3.5 asks the EDA_Module to identify the affected time period and
    *describe the observation* when a demand spike or drop is detected;
    Requirement 3.7 asks every output to carry a plain-language interpretation.
    This dataclass makes both structural: it pins the ``period`` and its
    ``value``, records how far the point sits from its local level (``score``, a
    robust modified z-score) and the ``direction`` of the departure
    (``"spike"``/``"drop"``), and carries a ready-to-read ``description`` string
    so an anomaly can never travel without its plain-language reading.

    Attributes:
        period: The period (timestamp) at which the anomaly occurs.
        value: The observed demand at that period.
        expected: The robust local level (rolling median) the value is compared
            against; the baseline the point departed from.
        score: The robust modified z-score of the point (based on the rolling
            median and MAD). Larger magnitude means a more extreme departure; the
            sign follows ``direction``.
        direction: ``"spike"`` when demand is anomalously high, ``"drop"`` when
            anomalously low.
        description: Plain-language description of the observation - what happened,
            when, and how far it departed from the local level (Requirement 3.7).
    """

    period: "pd.Timestamp"
    value: float
    expected: float
    score: float
    direction: str
    description: str


def detect_anomalies(
    series,
    *,
    window: int = 7,
    threshold: float = 3.5,
    period_column: str = PERIOD_COLUMN,
    region_column: str = REGION_COLUMN,
    demand_column: str = DEMAND_COLUMN,
) -> "list[Anomaly]":
    """Detect anomalous demand periods (spikes/drops) with a robust rule (R3.5, R3.7).

    Scans the demand series for periods whose demand departs sharply from its
    recent local level and returns one :class:`Anomaly` per flagged period,
    identifying the affected period and describing the observation in plain
    language (Requirements 3.5 and 3.7).

    Method - **robust local outlier detection**. For each period the function
    computes a centered rolling **median** (the resistant local level) and the
    rolling **median absolute deviation (MAD)** around it, then the modified
    z-score ``0.6745 * (value - median) / MAD``. A period is anomalous when the
    magnitude of that score exceeds ``threshold``. Median and MAD are used
    instead of mean and standard deviation precisely because a few large spikes
    would inflate the mean/std and hide the outliers; the robust statistics are
    not dragged around by the very points being detected. Periods scoring above
    ``+threshold`` are labelled ``"spike"``; those below ``-threshold`` are
    labelled ``"drop"``.

    The input may be a single-region series or the aggregate total; a long-format
    :data:`~src.preparation.DemandSeries` DataFrame is reduced to the system-wide
    total per period first (see :func:`_reduce_to_series`).

    Args:
        series: A period-indexed ``pandas.Series`` or a DemandSeries DataFrame.
        window: Size of the centered rolling window used for the local median and
            MAD. Defaults to ``7`` (one week for daily data), so an anomaly is
            judged against the surrounding week rather than the whole series.
        threshold: Modified-z-score magnitude above which a period is flagged.
            Defaults to ``3.5``, the conventional robust-outlier cut-off.
        period_column: Period column name (used when ``series`` is a DataFrame).
        region_column: Region column name (used when ``series`` is a DataFrame).
        demand_column: Demand column name (used when ``series`` is a DataFrame).

    Returns:
        A list of :class:`Anomaly` records, one per flagged period, ordered by
        period. Each carries the affected ``period``, its ``value``, the local
        ``expected`` level, the robust ``score``, the ``direction``
        (``"spike"``/``"drop"``) and a plain-language ``description``. The list is
        empty when no period breaches the threshold.

    Raises:
        KeyError: If a DataFrame ``series`` lacks the period/demand columns.
        ValueError: If ``window < 2`` or ``threshold <= 0``.
    """
    if window < 2:
        raise ValueError(f"window must be >= 2 to estimate a local level; got {window}.")
    if threshold <= 0:
        raise ValueError(f"threshold must be positive; got {threshold}.")

    reduced = _reduce_to_series(
        series,
        period_column=period_column,
        region_column=region_column,
        demand_column=demand_column,
    )
    if reduced.empty:
        return []

    # Robust local level (median) and dispersion (MAD) via a centered rolling
    # window. min_periods=2 lets the edges still get an estimate on short series.
    min_periods = min(2, len(reduced))
    rolling = reduced.rolling(window=window, center=True, min_periods=min_periods)
    local_median = rolling.median()
    abs_dev = (reduced - local_median).abs()
    local_mad = abs_dev.rolling(window=window, center=True, min_periods=min_periods).median()

    # Modified z-score. Where MAD is 0 (a locally flat stretch) fall back to a
    # scaled mean absolute deviation so a lone jump out of a flat run is still
    # catchable; if that is also 0 the point sits on a perfectly flat run and is,
    # by definition, not anomalous (score 0).
    mad_scaled = local_mad * 1.4826  # MAD -> comparable to a standard deviation
    fallback = abs_dev.rolling(window=window, center=True, min_periods=min_periods).mean() * 1.2533
    denom = mad_scaled.where(mad_scaled > 0, fallback)
    with np.errstate(divide="ignore", invalid="ignore"):
        scores = (reduced - local_median) / denom
    scores = scores.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    anomalies: list[Anomaly] = []
    overall_median = float(reduced.median())
    for period, score in scores.items():
        if abs(score) <= threshold:
            continue
        value = float(reduced.loc[period])
        expected = float(local_median.loc[period])
        direction = "spike" if score > 0 else "drop"
        anomalies.append(
            Anomaly(
                period=pd.Timestamp(period),
                value=value,
                expected=expected,
                score=float(score),
                direction=direction,
                description=_describe_anomaly(
                    period=period,
                    value=value,
                    expected=expected,
                    score=float(score),
                    direction=direction,
                    overall_median=overall_median,
                ),
            )
        )

    anomalies.sort(key=lambda a: a.period)
    return anomalies


def _describe_anomaly(
    *,
    period,
    value: float,
    expected: float,
    score: float,
    direction: str,
    overall_median: float,
) -> str:
    """Compose the plain-language description of one anomaly (R3.5, R3.7)."""
    when = pd.Timestamp(period).date()
    if expected > 0:
        pct = (value - expected) / expected * 100.0
        rel = f"{abs(pct):.0f}% {'above' if pct >= 0 else 'below'} the surrounding local level"
    else:
        rel = "well away from a near-zero local level"
    word = "spiked" if direction == "spike" else "dropped"
    return (
        f"Demand {word} on {when}: {_format_int(value)} trips versus a local level "
        f"of about {_format_int(expected)} ({rel}, robust z-score {score:+.1f}). "
        "This period departs sharply from its neighbours and warrants a look for a "
        "driver (event, weather, holiday, or a data issue) before modelling."
    )


def plot_anomalies(
    series,
    anomalies: "Optional[list[Anomaly]]" = None,
    *,
    window: int = 7,
    threshold: float = 3.5,
    period_column: str = PERIOD_COLUMN,
    region_column: str = REGION_COLUMN,
    demand_column: str = DEMAND_COLUMN,
    ax: "Optional[Axes]" = None,
) -> EDAResult:
    """Plot the demand series with detected anomalies highlighted (R3.5, R3.7).

    A convenience companion to :func:`detect_anomalies`: it draws the demand line
    and marks each anomalous period (spikes in red, drops in orange), returning
    the shared :class:`EDAResult` shape so the figure travels with a
    plain-language interpretation summarising how many spikes and drops were found
    and when the most extreme one occurred (Requirement 3.7).

    Args:
        series: A period-indexed ``pandas.Series`` or a DemandSeries DataFrame.
        anomalies: Pre-computed anomalies to plot. When ``None`` they are computed
            from ``series`` with the given ``window``/``threshold``.
        window: Rolling window forwarded to :func:`detect_anomalies` when
            ``anomalies`` is ``None``.
        threshold: Robust-z-score cut-off forwarded to :func:`detect_anomalies`
            when ``anomalies`` is ``None``.
        period_column: Period column name (used when ``series`` is a DataFrame).
        region_column: Region column name (used when ``series`` is a DataFrame).
        demand_column: Demand column name (used when ``series`` is a DataFrame).
        ax: Optional existing ``Axes`` to draw on. When ``None`` a new
            ``Figure``/``Axes`` pair is created.

    Returns:
        An :class:`EDAResult` whose ``figure`` shows the demand line with
        anomalies marked, whose ``interpretation`` summarises them in plain
        language, and whose ``stats`` holds the ``anomalies`` list.
    """
    reduced = _reduce_to_series(
        series,
        period_column=period_column,
        region_column=region_column,
        demand_column=demand_column,
    )
    if anomalies is None:
        anomalies = detect_anomalies(
            reduced, window=window, threshold=threshold
        )

    if ax is None:
        figure, axes = plt.subplots(figsize=(12, 6))
    else:
        axes = ax
        figure = axes.figure

    axes.plot(reduced.index, reduced.values, label="Demand", linewidth=1.4, color="steelblue")
    spikes = [a for a in anomalies if a.direction == "spike"]
    drops = [a for a in anomalies if a.direction == "drop"]
    if spikes:
        axes.scatter(
            [a.period for a in spikes],
            [a.value for a in spikes],
            color="crimson", zorder=5, s=45, label="Spike",
        )
    if drops:
        axes.scatter(
            [a.period for a in drops],
            [a.value for a in drops],
            color="darkorange", zorder=5, s=45, label="Drop",
        )
    axes.set_title("Demand with detected anomalies")
    axes.set_xlabel("Period")
    axes.set_ylabel("Demand (trip count)")
    if axes.get_legend_handles_labels()[0]:
        axes.legend(loc="upper left", fontsize="small")
    axes.grid(True, alpha=0.3)
    figure.autofmt_xdate()
    figure.tight_layout()

    if not anomalies:
        interpretation = (
            "No anomalous periods were flagged at the chosen sensitivity "
            f"(robust z-score threshold {threshold:g}, window {window}); demand "
            "stays within its normal local variation across the window."
        )
    else:
        most_extreme = max(anomalies, key=lambda a: abs(a.score))
        interpretation = (
            f"Detected {len(anomalies)} anomalous period(s): {len(spikes)} spike(s) "
            f"and {len(drops)} drop(s). The most extreme is a {most_extreme.direction} "
            f"on {most_extreme.period.date()} ({_format_int(most_extreme.value)} trips, "
            f"robust z-score {most_extreme.score:+.1f}). Each marked point departs "
            "sharply from its neighbours - investigate for events, weather, holidays "
            "or data issues before modelling."
        )

    return EDAResult(
        figure=figure,
        interpretation=interpretation,
        axes=axes,
        stats={"anomalies": anomalies},
    )


def demand_correlations(
    series,
    exog: "Optional[pd.DataFrame]" = None,
    *,
    method: str = "pearson",
    period_column: str = PERIOD_COLUMN,
    region_column: str = REGION_COLUMN,
    demand_column: str = DEMAND_COLUMN,
) -> pd.DataFrame:
    """Correlate demand with candidate explanatory variables (R3.6, R3.7).

    Reports how demand co-moves with the available candidate explanatory
    variables Requirement 3.6 calls for. The exogenous variables are aligned to
    the demand series by period and a correlation matrix (values in ``[-1, 1]``)
    is computed with :meth:`pandas.DataFrame.corr`. The returned DataFrame *is*
    the correlation table; its plain-language interpretation (Requirement 3.7) -
    naming the variables most strongly and least related to demand - is attached
    to the frame's ``.attrs["interpretation"]`` so the table and its reading
    travel together without changing the documented ``-> pd.DataFrame`` return
    type.

    When ``exog`` is omitted, calendar features derived from the period index
    (day-of-week, weekend flag, month, day-of-month) are used as the candidate
    explanatory variables, so the function is still meaningful with only the
    demand series in hand.

    The demand input may be a single-region series or the aggregate total; a
    long-format :data:`~src.preparation.DemandSeries` DataFrame is reduced to the
    system-wide total per period first (see :func:`_reduce_to_series`).

    Args:
        series: A period-indexed ``pandas.Series`` or a DemandSeries DataFrame.
        exog: Optional candidate explanatory variables as a ``DataFrame`` indexed
            by (or alignable to) the demand periods. When ``None``, calendar
            features are derived from the period index. Non-numeric columns are
            dropped before correlating.
        method: Correlation method passed to :meth:`pandas.DataFrame.corr`
            (``"pearson"`` default, or ``"spearman"``/``"kendall"``).
        period_column: Period column name (used when ``series`` is a DataFrame).
        region_column: Region column name (used when ``series`` is a DataFrame).
        demand_column: Demand column name (used when ``series`` is a DataFrame).

    Returns:
        A ``pandas.DataFrame`` correlation matrix whose first row/column is
        ``"demand"`` and whose remaining entries are the candidate explanatory
        variables, with all values in ``[-1, 1]``. A plain-language
        interpretation (Requirement 3.7) is attached at
        ``result.attrs["interpretation"]``.

    Raises:
        KeyError: If a DataFrame ``series`` lacks the period/demand columns.
        TypeError: If ``exog`` is provided but is not a ``pandas.DataFrame``.
    """
    reduced = _reduce_to_series(
        series,
        period_column=period_column,
        region_column=region_column,
        demand_column=demand_column,
    )
    demand = reduced.rename("demand")

    if exog is None:
        features = _calendar_features(demand.index)
    else:
        if not isinstance(exog, pd.DataFrame):
            raise TypeError(
                "exog must be a pandas DataFrame aligned to the demand periods; "
                f"got {type(exog).__name__}."
            )
        features = exog.copy()
        # Align exogenous rows to the demand periods by index. This handles the
        # common case where exog is indexed by period; a mismatched index simply
        # yields NaNs that corr() ignores pairwise.
        features = features.reindex(demand.index)
        # Keep only numeric columns - correlation is undefined for text/categoricals.
        features = features.select_dtypes(include=[np.number])

    combined = pd.concat([demand, features], axis=1)
    # Move demand first so it heads the resulting matrix.
    cols = ["demand"] + [c for c in combined.columns if c != "demand"]
    combined = combined[cols]

    corr = combined.corr(method=method)
    corr.attrs["interpretation"] = _build_correlation_interpretation(corr, method=method)
    return corr


def _calendar_features(index) -> pd.DataFrame:
    """Derive numeric calendar candidate variables from a period index (R3.6).

    Used when no exogenous variables are supplied: turns the period index into
    day-of-week, weekend flag, month and day-of-month, which are the calendar
    effects most likely to co-move with ride-hailing demand.
    """
    idx = pd.DatetimeIndex(pd.to_datetime(index, errors="coerce"))
    return pd.DataFrame(
        {
            "day_of_week": idx.dayofweek,
            "is_weekend": (idx.dayofweek >= 5).astype(int),
            "month": idx.month,
            "day_of_month": idx.day,
        },
        index=index,
    )


def _build_correlation_interpretation(corr: pd.DataFrame, *, method: str) -> str:
    """Compose the plain-language reading of the correlation table (R3.7).

    Names the candidate explanatory variables most strongly (and least) related
    to demand and reads the sign of each, so a reader learns which drivers move
    with demand without inspecting every cell of the matrix.
    """
    if "demand" not in corr.columns or corr.shape[1] <= 1:
        return (
            "No candidate explanatory variables were available to correlate "
            "against demand, so the table contains only demand's self-correlation."
        )

    # Correlations of every candidate variable with demand, strongest first.
    demand_corr = corr["demand"].drop(labels=["demand"], errors="ignore").dropna()
    if demand_corr.empty:
        return (
            "Candidate explanatory variables were supplied but none had enough "
            "overlapping, varying values to yield a correlation with demand."
        )

    ranked = demand_corr.reindex(demand_corr.abs().sort_values(ascending=False).index)

    def _strength(r: float) -> str:
        a = abs(r)
        if a >= 0.7:
            return "strong"
        if a >= 0.4:
            return "moderate"
        if a >= 0.2:
            return "weak"
        return "negligible"

    parts = [
        f"This table reports {method} correlation between demand and "
        f"{len(ranked)} candidate explanatory variable(s); values run from -1 "
        "(move oppositely) through 0 (unrelated) to +1 (move together)."
    ]

    top = ranked.index[0]
    top_r = float(ranked.iloc[0])
    parts.append(
        f"Demand is most related to '{top}' ({top_r:+.2f}, {_strength(top_r)} "
        f"{'positive' if top_r >= 0 else 'negative'} relationship): "
        + (
            f"higher '{top}' tends to coincide with higher demand."
            if top_r >= 0
            else f"higher '{top}' tends to coincide with lower demand."
        )
    )

    # Mention up to two more notable relationships.
    extras = []
    for name in ranked.index[1:3]:
        r = float(ranked.loc[name])
        extras.append(f"'{name}' ({r:+.2f}, {_strength(r)})")
    if extras:
        parts.append("Other relationships: " + ", ".join(extras) + ".")

    weakest = ranked.index[-1]
    weakest_r = float(ranked.iloc[-1])
    if _strength(weakest_r) == "negligible":
        parts.append(
            f"'{weakest}' ({weakest_r:+.2f}) shows essentially no linear "
            "relationship with demand."
        )

    parts.append(
        "Correlation is not causation; treat these as leads for feature selection "
        "and further investigation, not proof of a driver."
    )
    return " ".join(parts)
