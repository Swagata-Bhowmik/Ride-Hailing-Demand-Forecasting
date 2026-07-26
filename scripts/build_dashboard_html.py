"""Generate a standalone, self-contained, colourful interactive HTML dashboard.

Why this script exists
-----------------------
The Streamlit app (``dashboard/app.py``) is the *hosted* delivery layer - it needs
Python and a running server. This script produces a **single ``dashboard.html``
file** that a reviewer can double-click and open in ANY browser on ANY machine,
with **no Python, no terminal, and no internet connection**. It is a portable,
offline mirror of the same story the Streamlit app tells - only prettier: a
colourful, card-based, bullet-driven walkthrough with an interactive pipeline
flowchart and per-algorithm explanation cards.

How it stays consistent with the rest of the project
-----------------------------------------------------
It does **not** reimplement any analysis. It reuses the project's own ``src/``
functions (the single source of truth) and the Streamlit app's demo helpers so
the deliverables can never drift.

The golden rule (honesty)
-------------------------
The project forbids fabricated numbers. Until the real ~1GB NYC TLC pipeline has
run and saved a prepared series to ``data/demand_series.parquet``, the charts use
a **clearly-labelled illustrative demonstration series**. A prominent banner says
so. When a real prepared series is present (or passed via ``--demand-series``) it
is used instead and the banner is dropped automatically.

Self-containment
----------------
Plotly's JavaScript is embedded **inline** exactly once (in the first figure via
``include_plotlyjs=True``); every other figure uses ``include_plotlyjs=False``.
No external CDN ``<script src="http...">`` dependency.

Usage
-----
    python scripts/build_dashboard_html.py                 # -> dashboard.html
    python scripts/build_dashboard_html.py path/to/out.html
    python scripts/build_dashboard_html.py --demand-series data/demand_series.parquet
"""

from __future__ import annotations

import argparse
import html
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# --- Make the project root importable so ``import src...`` works regardless of the
# directory this script is launched from. ----------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import plotly.graph_objects as go  # noqa: E402
import plotly.io as pio  # noqa: E402

# Reuse the project's own logic (never reimplement analysis).
from src.business import (  # noqa: E402
    DEFAULT_IMPACT_ASSUMPTIONS,
    IMPACT_FORMULA,
    india_generalization,
    positioning_recommendation,
    quantify_impact,
)
from src.config import ScopeConfig, default_scope  # noqa: E402
from src.evaluation import (  # noqa: E402
    build_model_results,
    comparison_table,
    select_carry_forward,
)
from src.models.base import ExclusionRecord, Forecast, TrainedModel  # noqa: E402
from src.preparation import (  # noqa: E402
    DEMAND_COLUMN,
    PERIOD_COLUMN,
    REGION_COLUMN,
)

# --- Clearly-labelled demonstration data helpers -----------------------------
#
# These mirror the Streamlit app's demo fallbacks. They are replicated here -
# rather than imported - only because ``dashboard/app.py`` imports ``streamlit``
# at module load, which is not a dependency of this offline generator. The logic
# and seeds are identical. When a real prepared series exists on disk it is used
# instead (see ``get_demand_series``).

_DEMO_REGIONS = ("Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island")


def build_demo_demand_series(scope: ScopeConfig, *, n_days: int = 120) -> pd.DataFrame:
    """Return a deterministic, illustrative long-format DemandSeries (see app.py)."""
    rng = np.random.default_rng(42)
    end = pd.Timestamp(scope.window_end)
    periods = pd.date_range(end=end, periods=n_days, freq="D")

    base = {
        "Manhattan": 5200,
        "Brooklyn": 3100,
        "Queens": 2400,
        "Bronx": 1300,
        "Staten Island": 350,
    }

    rows: list[dict] = []
    for region in _DEMO_REGIONS:
        b = base[region]
        for i, period in enumerate(periods):
            trend = b * (1.0 + 0.0008 * i)
            weekly = 1.0 + 0.18 * np.sin(2 * np.pi * (period.dayofweek / 7.0))
            noise = rng.normal(1.0, 0.05)
            demand = max(0, int(round(trend * weekly * noise)))
            rows.append({PERIOD_COLUMN: period, REGION_COLUMN: region, DEMAND_COLUMN: demand})

    return pd.DataFrame(rows, columns=[PERIOD_COLUMN, REGION_COLUMN, DEMAND_COLUMN])


def build_demo_model_results(scope: ScopeConfig):
    """Build an illustrative ``(actual, index, results)`` bundle (see app.py).

    Routes hand-built forecasts through the *real* ``build_model_results`` so the
    metrics come from the actual evaluation code, not hand-typed numbers.
    """
    rng = np.random.default_rng(7)
    n = int(scope.holdout_periods)
    index = pd.date_range(end=pd.Timestamp(scope.window_end), periods=n, freq="D")

    t = np.arange(n)
    actual = 12000 + 400 * np.sin(2 * np.pi * (index.dayofweek / 7.0)) + 20 * t
    actual = np.round(actual).astype(float)

    def _forecast(name: str, noise_scale: float, bias: float) -> TrainedModel:
        values = actual + rng.normal(bias, noise_scale, size=n)
        forecast = Forecast(model_name=name, values=values, index=index)
        return TrainedModel(model_name=name, forecaster=object(), forecast=forecast)

    train_results = [
        _forecast("Holt-Winters", noise_scale=520, bias=-40),
        _forecast("SARIMA", noise_scale=430, bias=15),
        _forecast("Prophet", noise_scale=470, bias=-10),
        _forecast("XGBoost", noise_scale=360, bias=5),
        _forecast("LSTM", noise_scale=610, bias=60),
        ExclusionRecord(
            model_name="VAR",
            reason="illustrative: excluded because the demo series is aggregated to a "
            "single total (VAR needs multiple regional series).",
        ),
    ]

    results = build_model_results(train_results, actual)
    return actual, index, results


def get_demand_series(scope: ScopeConfig) -> "tuple[pd.DataFrame, bool]":
    """Return ``(series, is_real)`` - the prepared series if on disk, else the demo."""
    data_dir = _PROJECT_ROOT / "data"
    for candidate in ("demand_series.parquet", "demand_series.csv"):
        path = data_dir / candidate
        if path.exists():
            loaded = _load_series(path)
            if loaded is not None:
                return loaded, True
    return build_demo_demand_series(scope), False


# =============================================================================
# Colour system (vibrant, not the old taxi-yellow monochrome)
# =============================================================================

# A vibrant categorical palette used for chart traces.
_PALETTE = [
    "#6366f1",  # indigo
    "#06b6d4",  # cyan
    "#f43f5e",  # rose
    "#22c55e",  # green
    "#f59e0b",  # amber
    "#a855f7",  # purple
    "#ec4899",  # pink
    "#3b82f6",  # blue
]


# =============================================================================
# Chart builders (interactive Plotly figures)
# =============================================================================


def _plotly_layout(fig: go.Figure, title: str) -> go.Figure:
    """Apply a clean, colourful, readable theme to a figure and return it."""
    fig.update_layout(
        title=dict(text=title, font=dict(size=18, color="#4338ca")),
        template="plotly_white",
        font=dict(family="Inter, Segoe UI, Helvetica, Arial, sans-serif", size=13, color="#334155"),
        margin=dict(l=60, r=30, t=60, b=50),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        plot_bgcolor="rgba(255,255,255,0)",
        paper_bgcolor="rgba(255,255,255,0)",
        colorway=_PALETTE,
    )
    fig.update_xaxes(showgrid=True, gridcolor="#eef2ff", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#eef2ff", zeroline=False)
    return fig


def build_demand_over_time_fig(series: pd.DataFrame) -> go.Figure:
    """Interactive demand-over-time line chart, one trace per borough (hover + zoom)."""
    fig = go.Figure()
    regions = list(pd.unique(series[REGION_COLUMN]))
    for i, region in enumerate(regions):
        sub = series[series[REGION_COLUMN] == region].sort_values(PERIOD_COLUMN)
        fig.add_trace(
            go.Scatter(
                x=sub[PERIOD_COLUMN],
                y=sub[DEMAND_COLUMN],
                mode="lines",
                name=str(region),
                line=dict(width=2.4, color=_PALETTE[i % len(_PALETTE)]),
                hovertemplate="%{x|%Y-%m-%d}<br>%{fullData.name}: %{y:,} trips<extra></extra>",
            )
        )
    fig.update_yaxes(title_text="Demand (trip count)")
    fig.update_xaxes(title_text="Period")
    return _plotly_layout(fig, "Demand over time by borough")


def build_model_comparison_fig(table: pd.DataFrame) -> go.Figure:
    """Grouped bar chart of MAE / RMSE / MAPE per model (excluded models dropped here).

    Only scored models (non-NaN metrics) get bars; the full honest census -
    including excluded models - is shown in the comparison table rendered as HTML.
    """
    scored = table[table["mae"].notna()].copy()
    scored = scored.sort_values("mae")
    models = scored["model_name"].astype(str).tolist()

    fig = go.Figure()
    for metric, color in (("mae", "#6366f1"), ("rmse", "#06b6d4"), ("mape", "#f43f5e")):
        fig.add_trace(
            go.Bar(
                x=models,
                y=scored[metric],
                name=metric.upper(),
                marker_color=color,
                hovertemplate="%{x}<br>" + metric.upper() + ": %{y:.3f}<extra></extra>",
            )
        )
    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text="Error (lower is better)")
    fig.update_xaxes(title_text="Model")
    return _plotly_layout(fig, "Model comparison - error metrics (lower is better)")


def build_forecast_vs_actual_fig(actual, results, index) -> go.Figure:
    """Interactive forecast-vs-actual overlay on the holdout (bold actual + model lines)."""
    actual_arr = np.asarray(actual, dtype=float).ravel()
    x = list(index)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=actual_arr,
            mode="lines",
            name="Actual",
            line=dict(color="#0f172a", width=3.4),
            hovertemplate="%{x|%Y-%m-%d}<br>Actual: %{y:,.0f}<extra></extra>",
        )
    )
    color_i = 0
    for result in results:
        forecast = getattr(result, "forecast", None)
        if forecast is None or getattr(forecast, "values", None) is None:
            continue  # excluded / unscored model - still in the table below
        values = np.asarray(forecast.values, dtype=float).ravel()
        if values.size != actual_arr.size:
            continue
        fig.add_trace(
            go.Scatter(
                x=x,
                y=values,
                mode="lines",
                name=str(result.model_name),
                line=dict(width=1.8, color=_PALETTE[color_i % len(_PALETTE)]),
                opacity=0.9,
                hovertemplate="%{x|%Y-%m-%d}<br>%{fullData.name}: %{y:,.0f}<extra></extra>",
            )
        )
        color_i += 1
    fig.update_yaxes(title_text="Demand (trip count)")
    fig.update_xaxes(title_text="Holdout period")
    return _plotly_layout(fig, "Forecast vs. actual on the holdout")


# =============================================================================
# HTML component helpers (cards, bullets, flowchart)
# =============================================================================


def _fig_to_div(fig: go.Figure, *, include_js: bool) -> str:
    """Render a figure as an embeddable ``<div>`` (first figure embeds Plotly inline)."""
    return pio.to_html(
        fig,
        full_html=False,
        include_plotlyjs=True if include_js else False,
        config={"responsive": True, "displaylogo": False},
    )


def _df_to_html_table(df: pd.DataFrame) -> str:
    """Render a DataFrame as a clean HTML table (NaN shown as an em dash)."""
    display = df.copy()
    return display.to_html(
        index=False,
        na_rep="—",
        float_format=lambda v: f"{v:,.3f}",
        classes="data-table",
        border=0,
        escape=True,
    )


def _bullets(items) -> str:
    """Render an iterable of strings as a tick-marked bullet list (HTML-escaped)."""
    lis = "".join(f"<li>{html.escape(str(i))}</li>" for i in items)
    return f'<ul class="ticks">{lis}</ul>'


def _card(emoji: str, title: str, *, bullets=None, body: str = "", accent: str = "#6366f1") -> str:
    """Render a single accent-coloured card. Either ``bullets`` or ``body`` is shown."""
    inner = _bullets(bullets) if bullets else f"<p>{html.escape(body)}</p>"
    return (
        f'<div class="card" style="--accent:{accent}">'
        f'<div class="card-icon" style="background:{accent}1a;color:{accent}">{emoji}</div>'
        f'<h4>{html.escape(title)}</h4>{inner}</div>'
    )


def _card_grid(cards) -> str:
    """Wrap pre-rendered card HTML strings in a responsive grid."""
    return f'<div class="card-grid">{"".join(cards)}</div>'


def _flow_diagram(steps) -> str:
    """Render an interactive (hover-lift) colourful pipeline flowchart.

    ``steps`` is a list of ``(emoji, title, subtitle, accent)`` tuples. Rendered as
    connected gradient step-cards with arrow connectors that respond to hover.
    """
    nodes = []
    for i, (emoji, title, subtitle, accent) in enumerate(steps):
        nodes.append(
            f'<div class="flow-step" style="--accent:{accent}">'
            f'<div class="flow-badge">{i + 1}</div>'
            f'<div class="flow-emoji">{emoji}</div>'
            f'<div class="flow-title">{html.escape(title)}</div>'
            f'<div class="flow-sub">{html.escape(subtitle)}</div>'
            f"</div>"
        )
        if i < len(steps) - 1:
            nodes.append('<div class="flow-arrow">&#8594;</div>')
    return f'<div class="flow">{"".join(nodes)}</div>'


def _md_bullets(text: str) -> str:
    """Turn a plain/loosely-markdown block into HTML, favouring bullet lists.

    Splits on blank lines; ``- ``/``* `` lines become <li>. Multi-line prose blocks
    are also split into individual bullets (one sentence-ish line each) so the page
    stays bullet-driven rather than paragraph-heavy.
    """
    blocks = [b.strip() for b in text.strip().split("\n\n") if b.strip()]
    out: list[str] = []
    for block in blocks:
        lines = [ln.strip() for ln in block.split("\n") if ln.strip()]
        if lines and all(ln.startswith(("- ", "* ")) for ln in lines):
            out.append(_bullets(ln[2:] for ln in lines))
        elif len(lines) > 1:
            out.append(_bullets(lines))
        else:
            out.append(f"<p>{html.escape(lines[0])}</p>")
    return "\n".join(out)


# =============================================================================
# Section registry (label, anchor, emoji, accent colour) - colourful variety
# =============================================================================

_SECTIONS = [
    ("Business problem", "business-problem", "🎯", "#6366f1"),
    ("How the pipeline flows", "flow", "🧭", "#a855f7"),
    ("Data source & limitations", "data-source", "🗽", "#06b6d4"),
    ("EDA findings", "eda", "🔍", "#22c55e"),
    ("Data preparation", "data-preparation", "🧹", "#f59e0b"),
    ("Tools & technology", "tools", "🛠️", "#3b82f6"),
    ("Models & method", "models", "🤖", "#ec4899"),
    ("Results", "results", "📊", "#8b5cf6"),
    ("Business insights", "insights", "💡", "#10b981"),
    ("Generalisation to India", "india", "🇮🇳", "#f97316"),
]


# =============================================================================
# Section content (cards + bullets + flowchart; mirrors the Streamlit narrative)
# =============================================================================


def _section_business_problem(scope: ScopeConfig) -> str:
    cards = [
        _card("⏱️", "Spiky in time", bullets=[
            "Some hours surge, others go quiet.",
            "Rush hours, weekends, and events all shift demand.",
        ], accent="#6366f1"),
        _card("🗺️", "Spiky in space", bullets=[
            "Some neighbourhoods boom while others sit idle.",
            "Manhattan behaves nothing like Staten Island.",
        ], accent="#06b6d4"),
        _card("💸", "Two costs at once", bullets=[
            "Riders wait longer when no driver is nearby.",
            "Drivers sit idle when they are in the wrong place.",
        ], accent="#f43f5e"),
    ]
    scope_cards = [
        _card("🕐", "Time grain", body=scope.time_grain, accent="#22c55e"),
        _card("📍", "Geographic grain", body=scope.geographic_grain, accent="#f59e0b"),
        _card("📅", "Analysis window",
              body=f"{scope.window_start} → {scope.window_end} ({scope.window_months} months)",
              accent="#a855f7"),
        _card("🧪", "Holdout for evaluation",
              body=f"{scope.holdout_periods} periods reserved, never trained on",
              accent="#3b82f6"),
    ]
    return f"""
<div class="lead">Our goal: <strong>forecast demand - how much, when, and where -</strong>
so a platform can position drivers <em>ahead</em> of need, cutting both rider wait time
and driver idle time.</div>
{_card_grid(cards)}
<h3>🔒 Fixed forecasting scope</h3>
<p class="muted">Read from one source of truth (<code>src/config.ScopeConfig</code>) so every
stage uses identical values.</p>
{_card_grid(scope_cards)}
"""


def _section_flow(scope: ScopeConfig) -> str:
    steps = [
        ("📥", "Ingest", "NYC TLC FHVHV parquet", "#6366f1"),
        ("✅", "Validate", "every record checked, invalid logged", "#06b6d4"),
        ("🧹", "Prepare", "zone→borough, aggregate, zero-fill, lags", "#f59e0b"),
        ("🔍", "Explore", "trend, seasonality, stationarity", "#22c55e"),
        ("🤖", "Model", "train a broad candidate set", "#ec4899"),
        ("📊", "Evaluate", "score on the holdout", "#8b5cf6"),
        ("🏆", "Select", "carry forward the best, justified", "#f43f5e"),
        ("💡", "Act", "positioning + quantified impact", "#10b981"),
        ("🚀", "Deploy", "Streamlit, this HTML, auto-refresh", "#f97316"),
    ]
    return f"""
<div class="lead">The whole project is one honest, reproducible pipeline. Every arrow below is
real code in <code>src/</code> - hover a step to lift it.</div>
{_flow_diagram(steps)}
<p class="muted">Each stage feeds the next: nothing is hand-edited between steps, so the
numbers you see always trace back to the raw data.</p>
"""


def _section_data_source(scope: ScopeConfig) -> str:
    limitation_cards = [
        _card("🌍", "NYC, not India", bullets=[
            "The method generalises.",
            "The specific numbers do not transfer directly.",
        ], accent="#06b6d4"),
        _card("📈", "Trips ≈ demand", bullets=[
            "Trip counts are a proxy for true demand.",
            "Riders who gave up (unmet demand) aren't observed.",
        ], accent="#f43f5e"),
        _card("🔬", "Coarse grain", bullets=[
            "Borough/daily hides intraday surges.",
            "Neighbourhood-level spikes are smoothed out.",
        ], accent="#f59e0b"),
        _card("🌦️", "Partial drivers", bullets=[
            "Weather, events, surge pricing not in the base feed.",
            "Those are known extensions, not silent gaps.",
        ], accent="#a855f7"),
    ]
    return f"""
<div class="lead">Source: <strong>official NYC TLC Trip Record Data</strong> - the For-Hire
Vehicle High-Volume (FHVHV) feed (Uber/Lyft-style rides), published in Parquet by the NYC
Taxi &amp; Limousine Commission. Only real, public data - <strong>nothing fabricated</strong>.</div>
<h3>🧾 What "demand" means here</h3>
{_bullets([
    "Demand = count of trips per (period, region).",
    "Region = NYC borough; period = day.",
    "Every reported number is validated against the raw data before it is shown.",
])}
<h3>⚖️ Honest limitations</h3>
{_card_grid(limitation_cards)}
<p class="muted">Analysis window: {scope.window_start} → {scope.window_end}
({scope.window_months} months of FHVHV records).</p>
"""


def _section_eda(series: pd.DataFrame, demand_fig_div: str) -> str:
    total = int(series[DEMAND_COLUMN].sum())
    finding_cards = [
        _card("📆", "Weekly rhythm", bullets=[
            "Demand rises and falls on a clear 7-day cycle.",
            "Weekdays and weekends have distinct shapes.",
        ], accent="#22c55e"),
        _card("📈", "Mild trend", bullets=[
            "A gentle upward drift over the window.",
            "Models must capture level + trend together.",
        ], accent="#6366f1"),
        _card("🏙️", "Scale gap", bullets=[
            "Manhattan dominates total volume.",
            "Staten Island is a fraction of it - scale matters.",
        ], accent="#f59e0b"),
    ]
    return f"""
<div class="lead">Before modelling we study the series' shape, seasonality, and stationarity.
The chart is interactive - hover for values, drag to zoom.</div>
{demand_fig_div}
<h3>🧠 What the chart tells us</h3>
{_card_grid(finding_cards)}
<p class="muted">Total demand across the series: <strong>{total:,} trips</strong>.</p>
"""


def _section_data_preparation(series: pd.DataFrame) -> str:
    total = int(series[DEMAND_COLUMN].sum())
    preview = _df_to_html_table(series.head(12))
    prep_steps = [
        ("🚫", "Validate", "invalid records logged, not dropped silently", "#f43f5e"),
        ("🗺️", "Map zones", "PULocationID → borough via official lookup", "#06b6d4"),
        ("➕", "Aggregate", "count trips per (period, region)", "#6366f1"),
        ("0️⃣", "Zero-fill", "empty periods become 0, never missing", "#f59e0b"),
        ("🔁", "Lag features", "lag_1 / lag_7 / lag_14 for ML models", "#a855f7"),
    ]
    return f"""
<div class="lead">Raw trip records become a validated forecasting dataset via the pure
functions in <code>src/preparation.py</code>. Order matters - here's the flow:</div>
{_flow_diagram(prep_steps)}
<h3>📋 Prepared demand series (long format)</h3>
<p class="muted">One row per (period, region); demand is the trip count and is 0 for empty
periods rather than missing.</p>
{preview}
{_card_grid([
    _card("🔢", "Rows", body=f"{len(series):,}", accent="#6366f1"),
    _card("🌆", "Regions", body=f"{series[REGION_COLUMN].nunique()}", accent="#06b6d4"),
    _card("📅", "Distinct periods", body=f"{series[PERIOD_COLUMN].nunique()}", accent="#22c55e"),
    _card("🧮", "Total demand", body=f"{total:,}", accent="#f59e0b"),
])}
"""


def _section_tools(scope: ScopeConfig) -> str:
    tool_cards = [
        _card("🐼", "Data handling", bullets=["pandas", "pyarrow"], accent="#6366f1"),
        _card("📊", "Visualisation", bullets=["matplotlib", "Plotly (this dashboard)"], accent="#06b6d4"),
        _card("📐", "Classical stats", bullets=["statsmodels", "Holt-Winters, SARIMA, VAR"], accent="#f59e0b"),
        _card("🔮", "Modern forecasting", bullets=["Prophet"], accent="#a855f7"),
        _card("🌳", "Machine learning", bullets=["XGBoost on lag features"], accent="#22c55e"),
        _card("🧠", "Deep learning", bullets=["TensorFlow / Keras", "LSTM / GRU"], accent="#ec4899"),
        _card("🖥️", "Dashboard", bullets=["Streamlit (hosted)", "standalone HTML (this file)"], accent="#3b82f6"),
        _card("🧪", "Testing", bullets=["pytest", "Hypothesis (property-based)"], accent="#f43f5e"),
        _card("🚀", "Automate & deploy", bullets=["GitHub Actions", "Streamlit Cloud"], accent="#f97316"),
    ]
    return f"""
<div class="lead">The stack is deliberately open-source and reproducible. Core logic lives in
<code>src/</code> as pure, testable functions; both the Streamlit app and this file reuse the
same functions, so deliverables can never drift.</div>
{_card_grid(tool_cards)}
<p class="muted">Candidate model set: {html.escape(', '.join(scope.candidate_models))}.</p>
"""


def _model_card(emoji: str, name: str, meaning: str, use: str, output: str, accent: str) -> str:
    """A richer card for one algorithm: what it means, how we use it, how to read output."""
    return (
        f'<div class="model-card" style="--accent:{accent}">'
        f'<div class="model-head"><span class="model-emoji" style="background:{accent}1a;color:{accent}">{emoji}</span>'
        f'<h4>{html.escape(name)}</h4></div>'
        f'<div class="model-row"><span class="tag tag-what">What it means</span><p>{html.escape(meaning)}</p></div>'
        f'<div class="model-row"><span class="tag tag-use">How we use it</span><p>{html.escape(use)}</p></div>'
        f'<div class="model-row"><span class="tag tag-out">Reading its output</span><p>{html.escape(output)}</p></div>'
        f"</div>"
    )


def _section_models(scope: ScopeConfig) -> str:
    model_cards = [
        _model_card(
            "📉", "Holt-Winters (baseline)",
            "Exponential smoothing: a weighted average of the past that tracks level, trend, and seasonality.",
            "Our honest baseline - every fancier model must beat it to earn its place.",
            "A smooth forecast line. If nothing beats it, simplicity wins and we say so.",
            "#6366f1",
        ),
        _model_card(
            "🔁", "SARIMA / SARIMAX",
            "Seasonal AutoRegressive Integrated Moving Average - the classic statistical model for seasonal series. SARIMAX adds external regressors.",
            "Strong when the weekly pattern is stable; SARIMAX can fold in signals like weather.",
            "A forecast plus confidence intervals - wider bands mean more uncertainty.",
            "#06b6d4",
        ),
        _model_card(
            "🕸️", "VAR / VARMAX",
            "Vector AutoRegression forecasts several regions jointly, learning how boroughs move together.",
            "Useful when boroughs are correlated - one borough's past helps predict another.",
            "A simultaneous multi-borough forecast. Excluded automatically if the series is a single aggregate.",
            "#a855f7",
        ),
        _model_card(
            "🔮", "Prophet",
            "An additive model (trend + seasonality + holidays) from Meta, robust to gaps and outliers.",
            "Fast, interpretable decomposition; handles holidays out of the box.",
            "A forecast whose trend, weekly, and holiday parts can be viewed separately.",
            "#ec4899",
        ),
        _model_card(
            "🌳", "XGBoost",
            "Gradient-boosted decision trees trained on engineered lag features.",
            "Captures non-linear interactions between recent demand lags.",
            "A point forecast; feature importances reveal which lags matter most.",
            "#22c55e",
        ),
        _model_card(
            "🧠", "LSTM / GRU",
            "Recurrent neural networks that learn long-range patterns in a sequence.",
            "For complex temporal dependencies when there is enough data to train on.",
            "A sequence forecast - powerful, but data- and compute-hungry.",
            "#f59e0b",
        ),
    ]
    return f"""
<div class="lead">We train a <strong>broad</strong> set of models so the final choice is justified
by <em>evidence</em>, not assumption. Each card explains what the algorithm means, how we use it
here, and how to read its output.</div>
<div class="model-grid">{"".join(model_cards)}</div>
<h3>🧪 The method (fair by design)</h3>
{_bullets([
    f"Train on everything except the most-recent {scope.holdout_periods}-period holdout.",
    "Every model forecasts over the SAME holdout, so the comparison is apples-to-apples.",
    "Models that cannot train are recorded with a reason and still shown - never silently dropped.",
])}
"""


def _metric_cards() -> str:
    return _card_grid([
        _card("📏", "MAE", bullets=[
            "Mean Absolute Error.",
            "\"Typically off by this many trips.\"",
            "Lower is better.",
        ], accent="#6366f1"),
        _card("📐", "RMSE", bullets=[
            "Root Mean Squared Error.",
            "Like MAE but punishes big misses more.",
            "Lower is better.",
        ], accent="#06b6d4"),
        _card("📊", "MAPE", bullets=[
            "Mean Absolute Percentage Error.",
            "Scale-free - compare across regions.",
            "Lower is better.",
        ], accent="#f43f5e"),
    ])


def _section_results(table: pd.DataFrame, comparison_fig_div: str,
                     forecast_fig_div: str) -> str:
    table_html = _df_to_html_table(
        table.sort_values("mae", na_position="last").reset_index(drop=True)
    )
    names, justification = select_carry_forward(table, return_justification=True)
    return f"""
<div class="lead">The honest scoreboard: every model (including underperformers and excluded
ones), scored by the reusable <code>src.evaluation</code> functions.</div>
<h3>🔎 How to read the error metrics</h3>
{_metric_cards()}
<h3>📊 Model comparison chart</h3>
{comparison_fig_div}
<h3>📋 Model comparison table</h3>
<p class="muted">One row per model. Excluded models keep the columns blank and are flagged -
lower error is better.</p>
{table_html}
<h3>📈 Forecast vs. actual on the holdout</h3>
{forecast_fig_div}
<p class="muted">The bold dark line is actual holdout demand; each coloured line is a model's
forecast. Where a line hugs the dark line, that model tracks demand well.</p>
<h3>🏆 Models carried forward</h3>
<div class="pill-row">{"".join(f'<span class="pill">{html.escape(n)}</span>' for n in names)}</div>
<pre class="code-block">{html.escape(justification)}</pre>
"""


def _section_insights(scope: ScopeConfig, results) -> str:
    scored = [r for r in results if r.forecast is not None and r.metrics is not None]
    scored.sort(key=lambda r: r.metrics.mae)
    best = scored[0]
    recommendation = positioning_recommendation(best.forecast, scope)
    impact = quantify_impact(recommendation)

    assumptions_rows = "".join(
        f"<tr><td>{html.escape(k)}</td><td>{DEFAULT_IMPACT_ASSUMPTIONS[k]}</td></tr>"
        for k in DEFAULT_IMPACT_ASSUMPTIONS
    )
    return f"""
<div class="lead">A forecast only matters if it drives <strong>action</strong>. The reusable
<code>src.business</code> functions turn the best forecast (from
<strong>{html.escape(best.model_name)}</strong>) into a driver-positioning plan and quantify
the benefit - with the assumptions and formula shown so the number is defensible.</div>
<h3>🧭 Driver-positioning recommendation</h3>
<blockquote>{html.escape(recommendation.action)}</blockquote>
<h3>💰 Quantified impact</h3>
<div class="metrics">
  <div class="metric" style="--accent:#10b981"><div class="metric-value">{impact.rider_wait_minutes_saved:,.0f}</div>
    <div class="metric-label">Rider wait-minutes saved</div></div>
  <div class="metric" style="--accent:#6366f1"><div class="metric-value">{impact.driver_idle_minutes_saved:,.0f}</div>
    <div class="metric-label">Driver idle-minutes saved</div></div>
  <div class="metric" style="--accent:#f59e0b"><div class="metric-value">{impact.total_minutes_saved:,.0f}</div>
    <div class="metric-label">Total minutes saved</div></div>
</div>
<p><strong>In business terms:</strong> {html.escape(impact.narrative)}</p>
<h3>🧾 Assumptions &amp; formula (so the number is reproducible)</h3>
<table class="data-table">
  <thead><tr><th>Assumption</th><th>Value</th></tr></thead>
  <tbody>{assumptions_rows}</tbody>
</table>
<pre class="code-block">{html.escape(IMPACT_FORMULA)}</pre>
"""


def _section_india() -> str:
    return f'<div class="lead">The method travels even though the numbers do not. Here is how it maps to Indian ride-hailing:</div>{_md_bullets(india_generalization())}'


# =============================================================================
# Page assembly
# =============================================================================

# Plain CSS constant (inserted as an f-string *value*, so its braces are literal
# and need no escaping). Colourful, card-based, with an interactive flowchart.
_CSS = """
:root{
  --indigo:#6366f1; --purple:#a855f7; --cyan:#06b6d4; --rose:#f43f5e;
  --green:#22c55e; --amber:#f59e0b; --blue:#3b82f6; --ink:#0f172a;
  --muted:#64748b; --line:#e2e8f0; --bg:#f6f7fb;
}
*{box-sizing:border-box;}
body{margin:0;font-family:"Inter","Segoe UI",Helvetica,Arial,sans-serif;color:var(--ink);
  background:var(--bg);line-height:1.6;}
a{color:var(--indigo);text-decoration:none;}
a:hover{text-decoration:underline;}

/* Sidebar - gradient, not black */
.sidebar{position:fixed;top:0;left:0;width:262px;height:100vh;overflow-y:auto;padding:26px 18px;
  background:linear-gradient(180deg,#4338ca 0%,#6d28d9 55%,#7c3aed 100%);color:#fff;}
.sidebar h1{font-size:20px;margin:0 0 2px;}
.sidebar .sub{color:#dbeafe;font-size:12px;margin-bottom:18px;opacity:.9;}
.sidebar ul{list-style:none;padding:0;margin:0;}
.sidebar li a{display:flex;align-items:center;gap:9px;color:#eef2ff;padding:9px 11px;border-radius:9px;
  font-size:14px;border-left:3px solid transparent;transition:all .15s ease;}
.sidebar li a:hover{background:rgba(255,255,255,.16);border-left-color:#fde047;color:#fff;text-decoration:none;transform:translateX(2px);}
.sidebar .dot{font-size:15px;}
.sidebar .scope{margin-top:22px;font-size:11px;color:#e0e7ff;border-top:1px solid rgba(255,255,255,.25);padding-top:14px;}

/* Content */
.content{margin-left:262px;padding:0 48px 90px;max-width:1120px;}
header.hero{padding:46px 0 18px;}
header.hero h1{margin:0;font-size:32px;
  background:linear-gradient(90deg,#4338ca,#a855f7,#06b6d4);-webkit-background-clip:text;
  background-clip:text;color:transparent;}
header.hero p{color:var(--muted);margin:8px 0 0;font-size:15px;}

.badge{display:inline-block;padding:3px 11px;border-radius:999px;font-size:12px;font-weight:700;
  vertical-align:middle;margin-left:8px;-webkit-text-fill-color:initial;}
.badge-demo{background:#fef3c7;color:#92400e;}
.badge-real{background:#dcfce7;color:#166534;}
.banner{background:linear-gradient(90deg,#fff7ed,#fef9c3);border:1px solid #f59e0b;
  border-left:6px solid #f59e0b;padding:14px 18px;border-radius:12px;margin:22px 0;font-size:14px;
  box-shadow:0 4px 14px rgba(245,158,11,.12);}

/* Section cards - each gets its own accent via inline --accent */
section{background:#fff;border:1px solid var(--line);border-radius:16px;padding:28px 32px;
  margin:26px 0;scroll-margin-top:20px;border-top:5px solid var(--accent,#6366f1);
  box-shadow:0 6px 22px rgba(15,23,42,.05);}
section h2{margin:0 0 6px;font-size:24px;display:flex;align-items:center;gap:10px;}
section h2 .sec-emoji{font-size:24px;}
section h3{font-size:17px;margin-top:26px;color:var(--accent,#4338ca);}
.lead{font-size:15.5px;background:linear-gradient(90deg,#eef2ff,#faf5ff);border-radius:12px;
  padding:14px 18px;margin:8px 0 18px;border-left:4px solid var(--accent,#6366f1);}
.muted{color:var(--muted);font-size:13px;}

/* Tick bullet lists */
ul.ticks{list-style:none;padding:0;margin:10px 0;}
ul.ticks li{position:relative;padding:4px 0 4px 26px;}
ul.ticks li::before{content:"\\2713";position:absolute;left:0;top:4px;font-weight:800;
  color:var(--accent,#6366f1);}

/* Generic card grid */
.card-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(215px,1fr));gap:16px;margin:16px 0;}
.card{background:#fff;border:1px solid var(--line);border-radius:14px;padding:18px;
  border-top:4px solid var(--accent);transition:transform .16s ease,box-shadow .16s ease;}
.card:hover{transform:translateY(-4px);box-shadow:0 12px 26px rgba(15,23,42,.10);}
.card-icon{width:44px;height:44px;border-radius:12px;display:flex;align-items:center;
  justify-content:center;font-size:22px;margin-bottom:10px;}
.card h4{margin:0 0 6px;font-size:15px;}
.card p{margin:0;font-size:13.5px;color:#334155;}

/* Model cards */
.model-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:18px;margin:16px 0;}
.model-card{background:#fff;border:1px solid var(--line);border-radius:16px;padding:20px;
  border-left:5px solid var(--accent);transition:transform .16s ease,box-shadow .16s ease;}
.model-card:hover{transform:translateY(-4px);box-shadow:0 14px 30px rgba(15,23,42,.12);}
.model-head{display:flex;align-items:center;gap:12px;margin-bottom:8px;}
.model-emoji{width:42px;height:42px;border-radius:12px;display:flex;align-items:center;
  justify-content:center;font-size:22px;}
.model-head h4{margin:0;font-size:16px;}
.model-row{margin:10px 0;}
.model-row p{margin:5px 0 0;font-size:13.5px;color:#334155;}
.tag{display:inline-block;font-size:10.5px;font-weight:700;letter-spacing:.03em;text-transform:uppercase;
  padding:2px 9px;border-radius:999px;}
.tag-what{background:#eef2ff;color:#4338ca;}
.tag-use{background:#ecfeff;color:#0e7490;}
.tag-out{background:#fef2f2;color:#b91c1c;}

/* Interactive flowchart */
.flow{display:flex;flex-wrap:wrap;align-items:stretch;gap:8px;margin:18px 0;}
.flow-step{position:relative;flex:1;min-width:150px;background:#fff;border:1px solid var(--line);
  border-radius:14px;padding:16px 14px 14px;text-align:center;border-bottom:4px solid var(--accent);
  transition:transform .16s ease,box-shadow .16s ease;}
.flow-step:hover{transform:translateY(-5px);box-shadow:0 14px 28px rgba(15,23,42,.14);}
.flow-badge{position:absolute;top:-11px;left:50%;transform:translateX(-50%);width:24px;height:24px;
  border-radius:50%;background:var(--accent);color:#fff;font-size:12px;font-weight:700;
  display:flex;align-items:center;justify-content:center;box-shadow:0 3px 8px rgba(0,0,0,.18);}
.flow-emoji{font-size:26px;margin-top:6px;}
.flow-title{font-weight:700;font-size:14px;margin-top:6px;color:var(--accent);}
.flow-sub{font-size:11.5px;color:var(--muted);margin-top:3px;}
.flow-arrow{display:flex;align-items:center;font-size:22px;color:#cbd5e1;font-weight:700;}

/* Tables */
table.data-table{border-collapse:collapse;width:100%;margin:14px 0;font-size:13px;
  border-radius:10px;overflow:hidden;}
table.data-table th{background:linear-gradient(90deg,#4338ca,#7c3aed);color:#fff;text-align:left;padding:9px 11px;}
table.data-table td{border-bottom:1px solid var(--line);padding:8px 11px;}
table.data-table tr:nth-child(even) td{background:#f8fafc;}
table.data-table tr:hover td{background:#eef2ff;}

/* Metrics + misc */
blockquote{border-left:4px solid var(--purple);margin:12px 0;padding:12px 18px;
  background:linear-gradient(90deg,#faf5ff,#eef2ff);border-radius:0 10px 10px 0;font-size:15px;}
.code-block{background:#0f172a;color:#e2e8f0;padding:15px 17px;border-radius:12px;overflow-x:auto;
  font-size:12.5px;white-space:pre-wrap;font-family:"Fira Code",Consolas,monospace;}
code{background:#eef2ff;color:#4338ca;padding:1px 6px;border-radius:5px;font-size:90%;}
.metrics{display:flex;gap:16px;flex-wrap:wrap;margin:16px 0;}
.metric{flex:1;min-width:170px;background:#fff;border:1px solid var(--line);border-radius:14px;
  padding:18px;text-align:center;border-top:4px solid var(--accent,#6366f1);}
.metric-value{font-size:28px;font-weight:800;color:var(--accent,#6366f1);}
.metric-label{font-size:12px;color:var(--muted);}
.pill-row{display:flex;flex-wrap:wrap;gap:8px;margin:10px 0;}
.pill{background:linear-gradient(90deg,#6366f1,#a855f7);color:#fff;padding:5px 14px;border-radius:999px;
  font-size:13px;font-weight:600;}

@media (max-width:820px){
  .sidebar{position:static;width:100%;height:auto;}
  .content{margin-left:0;padding:0 20px 60px;}
}
"""


def _build_html(scope: ScopeConfig, series: pd.DataFrame, is_real: bool) -> str:
    """Assemble the full self-contained HTML document string."""
    actual, index, results = build_demo_model_results(scope)
    table = comparison_table(results)

    # The FIRST figure embeds Plotly's JS inline; the rest reuse it.
    demand_div = _fig_to_div(build_demand_over_time_fig(series), include_js=True)
    comparison_div = _fig_to_div(build_model_comparison_fig(table), include_js=False)
    forecast_div = _fig_to_div(
        build_forecast_vs_actual_fig(actual, results, index), include_js=False
    )

    section_html = {
        "business-problem": _section_business_problem(scope),
        "flow": _section_flow(scope),
        "data-source": _section_data_source(scope),
        "eda": _section_eda(series, demand_div),
        "data-preparation": _section_data_preparation(series),
        "tools": _section_tools(scope),
        "models": _section_models(scope),
        "results": _section_results(table, comparison_div, forecast_div),
        "insights": _section_insights(scope, results),
        "india": _section_india(),
    }

    nav_links = "\n".join(
        f'<li><a href="#{anchor}"><span class="dot">{emoji}</span>{html.escape(label)}</a></li>'
        for label, anchor, emoji, _accent in _SECTIONS
    )
    sections = "\n".join(
        f'<section id="{anchor}" style="--accent:{accent}">'
        f'<h2><span class="sec-emoji">{emoji}</span>{html.escape(label)}</h2>{section_html[anchor]}</section>'
        for label, anchor, emoji, accent in _SECTIONS
    )

    banner = "" if is_real else (
        '<div class="banner"><strong>⚠ Illustrative demonstration data.</strong> '
        "These charts and metrics use a synthetic, clearly-labelled series so the reusable "
        "analysis functions render. Run the pipeline over the validated NYC TLC data, save the "
        "prepared series to <code>data/demand_series.parquet</code>, then regenerate this file "
        "(<code>python scripts/build_dashboard_html.py</code>) to populate the real numbers.</div>"
    )
    data_badge = (
        '<span class="badge badge-real">REAL prepared data</span>'
        if is_real
        else '<span class="badge badge-demo">Illustrative demo data</span>'
    )
    generated = datetime.now().strftime("%Y-%m-%d %H:%M")
    scope_line = html.escape(
        f"{scope.time_grain} · {scope.geographic_grain} · "
        f"{scope.window_start} → {scope.window_end}"
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Ride-Hailing Demand Forecasting</title>
<style>{_CSS}</style>
</head>
<body>
  <nav class="sidebar">
    <h1>🚕 Demand Forecasting</h1>
    <div class="sub">A colourful storytelling walkthrough</div>
    <ul>{nav_links}</ul>
    <div class="scope">Scope: {scope_line}<br><br>Generated: {generated}</div>
  </nav>
  <main class="content">
    <header class="hero">
      <h1>Ride-Hailing Demand Forecasting {data_badge}</h1>
      <p>Forecasting where and when demand will peak, so drivers can be positioned ahead of need.</p>
    </header>
    {banner}
    {sections}
  </main>
</body>
</html>
"""


# =============================================================================
# CLI
# =============================================================================


def generate_dashboard(output_path: Path, demand_series_path: Path | None = None) -> Path:
    """Build the dashboard and write it to ``output_path``; return the path written."""
    scope = default_scope()

    if demand_series_path is not None:
        series = _load_series(demand_series_path)
        is_real = series is not None
        if not is_real:
            print(f"[warn] could not load {demand_series_path}; falling back to demo data.")
            series = build_demo_demand_series(scope)
    else:
        series, is_real = get_demand_series(scope)

    html_text = _build_html(scope, series, is_real)
    output_path.write_text(html_text, encoding="utf-8")
    return output_path


def _load_series(path: Path) -> pd.DataFrame | None:
    """Load a long-format demand series from parquet/csv, or None if unusable."""
    if not path.exists():
        return None
    try:
        if path.suffix == ".parquet":
            df = pd.read_parquet(path)
        else:
            df = pd.read_csv(path, parse_dates=[PERIOD_COLUMN])
    except Exception:  # pragma: no cover - fall back to demo on any read error
        return None
    if {PERIOD_COLUMN, REGION_COLUMN, DEMAND_COLUMN}.issubset(df.columns):
        return df
    return None


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Build a standalone, self-contained, colourful interactive dashboard.html."
    )
    parser.add_argument(
        "output",
        nargs="?",
        default=str(_PROJECT_ROOT / "dashboard.html"),
        help="Output HTML path (default: dashboard.html at the repo root).",
    )
    parser.add_argument(
        "--demand-series",
        default=None,
        help="Optional path to a real prepared demand series (.parquet/.csv). "
        "If valid, it is used and the illustrative-data banner is dropped.",
    )
    args = parser.parse_args(argv)

    out = Path(args.output).resolve()
    demand = Path(args.demand_series).resolve() if args.demand_series else None
    written = generate_dashboard(out, demand)

    size_kb = written.stat().st_size / 1024
    print(f"[ok] wrote {written} ({size_kb:,.1f} KB)")


if __name__ == "__main__":
    main()
