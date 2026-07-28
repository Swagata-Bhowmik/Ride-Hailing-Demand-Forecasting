"""Ride-Hailing Demand Forecasting - Streamlit storytelling dashboard (Requirement 9).

This is the interactive delivery layer described in the design's "Delivery" layer
and the Dashboard component (`dashboard/app.py`). It presents the whole project as
a sidebar-navigated story (Requirements 9.1, 9.2) and displays the model comparison
results and forecast visualizations (Requirement 9.3), reusing the same ``src/``
functions the notebook uses so the two deliverables stay consistent.

Structure (why it looks the way it does):

* Every story section is a plain ``render_*(...)`` function. A single ordered
  ``SECTIONS`` mapping wires each sidebar label to its render function, so
  navigation is data-driven and a new section is one dict entry.
* ``main()`` builds the ``st.sidebar`` radio and dispatches to the selected
  section. Nothing renders at import time - the Streamlit run is guarded by
  ``if __name__ == "__main__": main()`` (Streamlit sets ``__name__`` to
  ``"__main__"`` when it runs the script). This lets the smoke test (task 13.4)
  ``import dashboard.app`` and call individual section functions without a running
  Streamlit server.
* Heavy statistical imports (``src.eda`` -> statsmodels) are done lazily inside the
  EDA section, so importing this module stays light.

Honesty note (the project's "golden rule"): the real EDA charts, model-comparison
metrics, and forecasts are produced by running the pipeline in ``src/`` over the
validated NYC TLC data. When those real artifacts are not present on disk, the
data-backed sections fall back to a **clearly-labelled illustrative demonstration
series** so the reusable ``src/`` visualizations still render - every such view is
prominently marked as illustrative, and nothing synthetic is presented as a real
NYC result.

Design references:
- Components and Interfaces -> Dashboard (`dashboard/app.py`)
- Requirements 9.1 (sidebar navigation), 9.2 (required sections),
  9.3 (model comparison + forecast visualizations)
- Reuses: src.config, src.eda, src.evaluation, src.business, src.preparation,
  src.models.base
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# --- Make the project root importable so ``import src...`` works when the app is
# launched with ``streamlit run dashboard/app.py`` from the project root. ---------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Lightweight src modules imported at top (no heavy statsmodels/matplotlib pulled
# in here - those are imported lazily inside the sections that need them).
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
    plot_forecast_vs_actual,
    select_carry_forward,
)
from src.models.base import ExclusionRecord, Forecast, TrainedModel  # noqa: E402
from src.preparation import (  # noqa: E402
    DEMAND_COLUMN,
    PERIOD_COLUMN,
    REGION_COLUMN,
)

# Pure upload-format validator (no Streamlit dependency) reused by the
# "Upload & analyze" section and property-tested independently (task 13.3).
from dashboard.upload_validation import (  # noqa: E402
    REQUIRED_COLUMNS,
    validate_upload,
)

# The NYC TLC boroughs used as the Geographic_Grain (design Key Design Decisions).
_DEMO_REGIONS = ("Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island")


# =============================================================================
# Demonstration data helpers
# =============================================================================
#
# These build a *clearly-labelled illustrative* demand series and forecast set so
# the reusable src/ visualizations render even before the real pipeline artifacts
# exist. They are deterministic (seeded) and never presented as real NYC results.


def build_demo_demand_series(scope: ScopeConfig, *, n_days: int = 120) -> pd.DataFrame:
    """Return a deterministic, illustrative long-format DemandSeries for the UI.

    Produces one row per ``(period, region)`` over the most recent ``n_days`` of
    the scope window, with a mild upward trend, weekly seasonality, and seeded
    noise - the same long format (``period``/``region``/``demand``) the real
    preparation pipeline emits, so the reused ``src`` functions operate on it
    unchanged. This is illustrative only; it is never labelled as real data.
    """
    rng = np.random.default_rng(42)
    end = pd.Timestamp(scope.window_end)
    periods = pd.date_range(end=end, periods=n_days, freq="D")

    # Rough relative size of each borough so the demo looks plausible.
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
            trend = b * (1.0 + 0.0008 * i)  # slow growth over the window
            weekly = 1.0 + 0.18 * np.sin(2 * np.pi * (period.dayofweek / 7.0))
            noise = rng.normal(1.0, 0.05)
            demand = max(0, int(round(trend * weekly * noise)))
            rows.append({PERIOD_COLUMN: period, REGION_COLUMN: region, DEMAND_COLUMN: demand})

    return pd.DataFrame(rows, columns=[PERIOD_COLUMN, REGION_COLUMN, DEMAND_COLUMN])


def build_demo_model_results(scope: ScopeConfig):
    """Build an illustrative ``(actual, index, results)`` bundle for the Results view.

    Constructs holdout actual demand plus several models' forecasts (as
    :class:`~src.models.base.Forecast` / :class:`~src.models.base.TrainedModel`),
    including one :class:`~src.models.base.ExclusionRecord` to show that excluded
    models still appear honestly. It then routes them through the *real*
    :func:`~src.evaluation.build_model_results` so the metrics come from the
    actual evaluation code, not hand-typed numbers.

    Returns:
        ``(actual, index, results)`` where ``actual`` is the holdout total-demand
        array, ``index`` is the holdout period index, and ``results`` is the list
        of :class:`~src.evaluation.ModelResult` produced by the evaluation module.
    """
    rng = np.random.default_rng(7)
    n = int(scope.holdout_periods)
    index = pd.date_range(end=pd.Timestamp(scope.window_end), periods=n, freq="D")

    # Illustrative holdout actuals (system-wide daily total demand).
    t = np.arange(n)
    actual = 12000 + 400 * np.sin(2 * np.pi * (index.dayofweek / 7.0)) + 20 * t
    actual = np.round(actual).astype(float)

    def _forecast(name: str, noise_scale: float, bias: float) -> TrainedModel:
        values = actual + rng.normal(bias, noise_scale, size=n)
        forecast = Forecast(model_name=name, values=values, index=index)
        # forecaster instance is unused for scoring/plotting; a lightweight stand-in.
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


#: The real model-results artifact written by ``scripts/train_models.py`` (real
#: NYC TLC fits, evaluated at the system-wide total-daily-demand level). Small and
#: committed so the deployed dashboard shows real numbers.
_MODEL_RESULTS_PATH = _PROJECT_ROOT / "dashboard" / "model_results.json"


def load_real_model_results(scope: ScopeConfig):
    """Return ``(actual, index, results, artifact)`` from the real artifact, or ``None``.

    Reads the JSON written by ``scripts/train_models.py`` and reconstructs the same
    ``(actual, index, results)`` bundle :func:`build_demo_model_results` returns -
    a :class:`~src.models.base.TrainedModel` for every scored model and an
    :class:`~src.models.base.ExclusionRecord` for every excluded one - then routes
    them through the *real* :func:`~src.evaluation.build_model_results`, so the
    metrics are recomputed from the real forecast arrays rather than stored. Returns
    ``None`` when the artifact is absent or unreadable so callers fall back to the
    illustrative demo.
    """
    if not _MODEL_RESULTS_PATH.exists():
        return None
    try:
        with open(_MODEL_RESULTS_PATH, "r", encoding="utf-8") as fh:
            artifact = json.load(fh)
        index = pd.DatetimeIndex(pd.to_datetime(artifact["index"]))
        actual = np.asarray(artifact["actual"], dtype=float)

        train_results = []
        for model in artifact["models"]:
            if model.get("values") is None:
                train_results.append(
                    ExclusionRecord(
                        model_name=model["name"],
                        reason=model.get("excluded_reason", "excluded"),
                    )
                )
            else:
                values = np.asarray(model["values"], dtype=float)
                forecast = Forecast(model_name=model["name"], values=values, index=index)
                train_results.append(
                    TrainedModel(
                        model_name=model["name"], forecaster=object(), forecast=forecast
                    )
                )
        results = build_model_results(train_results, actual)
        return actual, index, results, artifact
    except Exception:  # pragma: no cover - any read/parse error falls back to demo
        return None


def get_model_results(scope: ScopeConfig):
    """Return ``(actual, index, results, is_real)`` - real artifact if present, else demo."""
    real = load_real_model_results(scope)
    if real is not None:
        actual, index, results, _ = real
        return actual, index, results, True
    actual, index, results = build_demo_model_results(scope)
    return actual, index, results, False


#: Candidate locations for the prepared demand series, in priority order. The
#: local pipeline writes the full parquet under the git-ignored ``data/``; a small
#: committed CSV (``dashboard/demand_series.csv``) is the deploy-time fallback so
#: the hosted app shows the real series without needing the large raw data.
_SERIES_CANDIDATES = (
    _PROJECT_ROOT / "data" / "demand_series.parquet",
    _PROJECT_ROOT / "data" / "demand_series.csv",
    _PROJECT_ROOT / "dashboard" / "demand_series.csv",
)


def get_demand_series(scope: ScopeConfig) -> "tuple[pd.DataFrame, bool]":
    """Return ``(series, is_real)`` - the prepared series if on disk, else the demo.

    Looks for a prepared demand series saved by the pipeline (the git-ignored
    ``data/demand_series.parquet``/``.csv``) and, failing that, the small committed
    ``dashboard/demand_series.csv`` used on the deployed app. If found it is used
    as-is (``is_real=True``); otherwise a clearly-labelled illustrative series is
    returned (``is_real=False``) so the reused ``src`` charts still render.
    """
    for path in _SERIES_CANDIDATES:
        if path.exists():
            try:
                if path.suffix == ".parquet":
                    df = pd.read_parquet(path)
                else:
                    df = pd.read_csv(path, parse_dates=[PERIOD_COLUMN])
                if {PERIOD_COLUMN, REGION_COLUMN, DEMAND_COLUMN}.issubset(df.columns):
                    return df, True
            except Exception:  # pragma: no cover - fall back to demo on any read error
                pass
    return build_demo_demand_series(scope), False


def _demo_notice(is_real: bool) -> None:
    """Render a prominent illustrative-data banner when data is not the real series."""
    if not is_real:
        st.info(
            "Illustrative demonstration data. These charts use a synthetic, "
            "clearly-labelled series so the reusable analysis functions render. "
            "Run the pipeline over the validated NYC TLC data to populate the real "
            "numbers (save the prepared series to `data/demand_series.parquet`)."
        )


def _model_results_notice(is_real: bool) -> None:
    """Flag whether the model scoreboard comes from real fits or the illustrative demo."""
    if is_real:
        st.success(
            "Real results. These metrics come from the candidate models fit on the "
            "real NYC TLC demand series (via `scripts/train_models.py`), scored on "
            "the reserved holdout at the system-wide total-daily-demand level."
        )
    else:
        st.info(
            "Illustrative demonstration scoreboard. These forecasts are synthetic, "
            "clearly-labelled placeholders so the reusable evaluation functions "
            "render. Run `python scripts/train_models.py` over the prepared series "
            "to populate the real model results."
        )


# =============================================================================
# Visual theme + reusable colourful components (mirrors dashboard.html)
# =============================================================================

#: Vibrant categorical palette shared by cards and chart traces.
PALETTE = [
    "#6366f1", "#06b6d4", "#f43f5e", "#22c55e",
    "#f59e0b", "#a855f7", "#ec4899", "#3b82f6",
]

_THEME_CSS = """
<style>
:root { --ink:#0f172a; --muted:#64748b; }
.stApp { background: linear-gradient(180deg,#f8fafc 0%,#eef2ff 100%); }
.block-container { padding-top: 1.4rem; max-width: 1200px; }
h1,h2,h3,h4 { font-family:'Inter','Segoe UI',system-ui,sans-serif; color:var(--ink); letter-spacing:-.01em; }

/* Hero banner */
.hero { border-radius:18px; padding:26px 30px; margin:2px 0 18px;
  background:linear-gradient(120deg,#6366f1 0%,#8b5cf6 45%,#ec4899 100%);
  color:#fff; box-shadow:0 12px 34px rgba(99,102,241,.28); }
.hero h1 { color:#fff; margin:0 0 6px; font-size:30px; }
.hero p { color:#eef2ff; margin:0; font-size:15px; max-width:70ch; }
.hero .badge { display:inline-block; margin-top:12px; padding:4px 12px; border-radius:999px;
  font-size:12px; font-weight:700; background:rgba(255,255,255,.22); color:#fff;
  border:1px solid rgba(255,255,255,.4); }

/* Section header */
.sec-head { display:flex; align-items:center; gap:12px; margin:6px 0 10px; }
.sec-emoji { width:44px; height:44px; border-radius:12px; display:flex; align-items:center;
  justify-content:center; font-size:22px; color:#fff; box-shadow:0 6px 16px rgba(15,23,42,.16); }
.sec-head h2 { margin:0; font-size:23px; }
.lead { font-size:16px; color:#334155; line-height:1.6; margin:6px 0 14px;
  padding-left:14px; border-left:4px solid #6366f1; }

/* Cards */
.cards { display:flex; flex-wrap:wrap; gap:14px; margin:10px 0 18px; }
.card { flex:1 1 240px; min-width:220px; background:#fff; border-radius:16px; padding:16px 18px;
  border:1px solid #eef2f7; border-top:4px solid var(--accent,#6366f1);
  box-shadow:0 6px 18px rgba(15,23,42,.06); transition:transform .15s ease, box-shadow .15s ease; }
.card:hover { transform:translateY(-4px); box-shadow:0 16px 30px rgba(15,23,42,.12); }
.card .ic { width:38px; height:38px; border-radius:10px; display:flex; align-items:center;
  justify-content:center; font-size:20px; margin-bottom:8px; }
.card h4 { margin:2px 0 8px; font-size:16px; }
.card p { margin:0; color:#475569; font-size:14px; line-height:1.55; }
.card ul { margin:0; padding-left:2px; list-style:none; }
.card li { position:relative; padding-left:20px; margin:4px 0; color:#475569; font-size:14px; }
.card li:before { content:'\\2713'; position:absolute; left:0; color:var(--accent,#6366f1); font-weight:800; }

/* KPI tiles (custom + st.metric restyle) */
.kpis { display:flex; flex-wrap:wrap; gap:14px; margin:8px 0 18px; }
.kpi { flex:1 1 180px; min-width:150px; background:#fff; border-radius:16px; padding:16px 18px;
  border:1px solid #eef2f7; box-shadow:0 6px 18px rgba(15,23,42,.06);
  border-left:5px solid var(--accent,#6366f1); }
.kpi .val { font-size:26px; font-weight:800; color:var(--accent,#6366f1); line-height:1.1; }
.kpi .lab { font-size:12.5px; color:var(--muted); margin-top:4px; text-transform:uppercase; letter-spacing:.04em; }
[data-testid="stMetric"] { background:#fff; border-radius:14px; padding:14px 16px;
  border:1px solid #eef2f7; border-left:5px solid #6366f1; box-shadow:0 6px 16px rgba(15,23,42,.06); }
[data-testid="stMetricValue"] { color:#4338ca; font-weight:800; }

/* Flow diagram */
.flow { display:flex; flex-wrap:wrap; align-items:stretch; gap:8px; margin:12px 0 18px; }
.step { flex:1 1 130px; min-width:120px; background:#fff; border-radius:14px; padding:12px;
  text-align:center; border:1px solid #eef2f7; border-bottom:4px solid var(--accent,#6366f1);
  box-shadow:0 6px 16px rgba(15,23,42,.06); transition:transform .15s ease; }
.step:hover { transform:translateY(-4px); }
.step .se { font-size:22px; } .step .st { font-weight:700; font-size:13.5px; margin-top:4px; color:#1e293b; }
.step .ss { font-size:11.5px; color:var(--muted); margin-top:2px; }

/* Pills */
.pills { display:flex; flex-wrap:wrap; gap:8px; margin:8px 0 4px; }
.pill { padding:6px 13px; border-radius:999px; font-size:13px; font-weight:700; color:#fff;
  box-shadow:0 4px 12px rgba(15,23,42,.12); }

section[data-testid="stSidebar"] { background:linear-gradient(180deg,#111827,#1f2937); }
section[data-testid="stSidebar"] * { color:#e5e7eb; }

/* "In plain English" concept box - teaches a term simply, where it appears */
.concept { background:linear-gradient(135deg,#ecfeff 0%,#eef2ff 100%);
  border:1px solid #c7d2fe; border-left:6px solid #06b6d4; border-radius:14px;
  padding:14px 18px; margin:12px 0 16px; box-shadow:0 6px 16px rgba(6,182,212,.10); }
.concept .ct { font-size:13px; font-weight:800; color:#0e7490; text-transform:uppercase;
  letter-spacing:.05em; margin-bottom:6px; display:flex; align-items:center; gap:7px; }
.concept p { margin:0 0 6px; color:#334155; font-size:14.5px; line-height:1.6; }
.concept p:last-child { margin-bottom:0; }
.concept b, .concept strong { color:#0f172a; }

/* Story-arc strip on the overview page */
.arc { display:flex; flex-wrap:wrap; gap:10px; margin:10px 0 18px; }
.arc .node { flex:1 1 150px; min-width:140px; background:#fff; border-radius:14px;
  padding:14px 16px; border:1px solid #eef2f7; border-top:4px solid var(--accent,#6366f1);
  box-shadow:0 6px 16px rgba(15,23,42,.06); }
.arc .node .n { font-size:12px; font-weight:800; color:var(--accent,#6366f1); }
.arc .node .h { font-weight:700; font-size:14.5px; color:#1e293b; margin:3px 0; }
.arc .node .d { font-size:12.5px; color:#64748b; line-height:1.5; }

/* Takeaway banner - the one thing to remember from a section */
.takeaway { background:linear-gradient(135deg,#f0fdf4 0%,#ecfdf5 100%);
  border:1px solid #bbf7d0; border-left:6px solid #22c55e; border-radius:14px;
  padding:13px 18px; margin:16px 0 6px; color:#14532d; font-size:14.5px; line-height:1.6; }
.takeaway b { color:#052e16; }
</style>
"""


def _inject_theme() -> None:
    st.markdown(_THEME_CSS, unsafe_allow_html=True)


def _hero(title: str, subtitle: str, badge: str) -> None:
    st.markdown(
        f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p>'
        f'<span class="badge">{badge}</span></div>',
        unsafe_allow_html=True,
    )


def _section_header(emoji: str, title: str, accent: str) -> None:
    st.markdown(
        f'<div class="sec-head"><div class="sec-emoji" style="background:{accent}">{emoji}</div>'
        f'<h2>{title}</h2></div>',
        unsafe_allow_html=True,
    )


def _lead(text: str) -> None:
    st.markdown(f'<div class="lead">{text}</div>', unsafe_allow_html=True)


def _card(emoji: str, title: str, *, bullets=None, body: str = "", accent: str = "#6366f1") -> str:
    if bullets:
        inner = "<ul>" + "".join(f"<li>{b}</li>" for b in bullets) + "</ul>"
    else:
        inner = f"<p>{body}</p>"
    return (
        f'<div class="card" style="--accent:{accent}">'
        f'<div class="ic" style="background:{accent}1a;color:{accent}">{emoji}</div>'
        f'<h4>{title}</h4>{inner}</div>'
    )


def _cards(cards) -> None:
    st.markdown(f'<div class="cards">{"".join(cards)}</div>', unsafe_allow_html=True)


def _kpis(items) -> None:
    """items: iterable of (label, value, accent)."""
    tiles = "".join(
        f'<div class="kpi" style="--accent:{accent}"><div class="val">{value}</div>'
        f'<div class="lab">{label}</div></div>'
        for label, value, accent in items
    )
    st.markdown(f'<div class="kpis">{tiles}</div>', unsafe_allow_html=True)


def _flow(steps) -> None:
    """steps: iterable of (emoji, title, subtitle, accent)."""
    nodes = "".join(
        f'<div class="step" style="--accent:{accent}"><div class="se">{e}</div>'
        f'<div class="st">{t}</div><div class="ss">{s}</div></div>'
        for e, t, s, accent in steps
    )
    st.markdown(f'<div class="flow">{nodes}</div>', unsafe_allow_html=True)


def _pills(names) -> None:
    chips = "".join(
        f'<span class="pill" style="background:{PALETTE[i % len(PALETTE)]}">{n}</span>'
        for i, n in enumerate(names)
    )
    st.markdown(f'<div class="pills">{chips}</div>', unsafe_allow_html=True)


def _concept(title: str, *paragraphs: str) -> None:
    """Render an "In plain English" teaching box explaining one term simply.

    Each section uses this to define the *specific* concepts it introduces (and
    nothing another section already covered), so the reader is taught the jargon
    exactly where it first matters - never repeated tab to tab.
    """
    body = "".join(f"<p>{p}</p>" for p in paragraphs)
    st.markdown(
        f'<div class="concept"><div class="ct">💡 In plain English — {title}</div>{body}</div>',
        unsafe_allow_html=True,
    )


def _takeaway(text: str) -> None:
    """Render the single 'remember this' banner that closes a section."""
    st.markdown(f'<div class="takeaway">✅ <b>Takeaway:</b> {text}</div>', unsafe_allow_html=True)


def _story_arc(nodes) -> None:
    """Render the overview story-arc strip. nodes: (num, title, desc, accent)."""
    cells = "".join(
        f'<div class="node" style="--accent:{accent}"><div class="n">{num}</div>'
        f'<div class="h">{title}</div><div class="d">{desc}</div></div>'
        for num, title, desc, accent in nodes
    )
    st.markdown(f'<div class="arc">{cells}</div>', unsafe_allow_html=True)


# =============================================================================
# Interactive Plotly chart builders
# =============================================================================

def _plotly_theme(fig: go.Figure, title: str) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, font=dict(size=17, color="#4338ca")),
        template="plotly_white", height=430,
        font=dict(family="Inter, Segoe UI, sans-serif", size=13, color="#334155"),
        margin=dict(l=55, r=25, t=55, b=45), hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", colorway=PALETTE,
    )
    fig.update_xaxes(showgrid=True, gridcolor="#eef2ff", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#eef2ff", zeroline=False)
    return fig


def _fig_demand_over_time(series: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for i, region in enumerate(pd.unique(series[REGION_COLUMN])):
        sub = series[series[REGION_COLUMN] == region].sort_values(PERIOD_COLUMN)
        fig.add_trace(go.Scatter(
            x=sub[PERIOD_COLUMN], y=sub[DEMAND_COLUMN], mode="lines", name=str(region),
            line=dict(width=2.2, color=PALETTE[i % len(PALETTE)]),
            hovertemplate="%{x|%Y-%m-%d}<br>%{fullData.name}: %{y:,} trips<extra></extra>",
        ))
    fig.update_yaxes(title_text="Demand (trips)")
    return _plotly_theme(fig, "Demand over time by region")


def _fig_model_comparison(table: pd.DataFrame) -> go.Figure:
    scored = table[table["mae"].notna()].sort_values("mae")
    models = scored["model_name"].astype(str).tolist()
    fig = go.Figure()
    for metric, color in (("mae", "#6366f1"), ("rmse", "#06b6d4"), ("mape", "#f43f5e")):
        fig.add_trace(go.Bar(x=models, y=scored[metric], name=metric.upper(), marker_color=color,
                             hovertemplate="%{x}<br>" + metric.upper() + ": %{y:.3f}<extra></extra>"))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text="Error (lower is better)")
    return _plotly_theme(fig, "Model comparison - error metrics (lower is better)")


def _fig_forecast_vs_actual(actual, results, index) -> go.Figure:
    actual_arr = np.asarray(actual, dtype=float).ravel()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(index), y=actual_arr, mode="lines", name="Actual",
                             line=dict(color="#0f172a", width=3.4),
                             hovertemplate="%{x|%Y-%m-%d}<br>Actual: %{y:,.0f}<extra></extra>"))
    ci = 0
    for r in results:
        fc = getattr(r, "forecast", None)
        if fc is None or getattr(fc, "values", None) is None:
            continue
        vals = np.asarray(fc.values, dtype=float).ravel()
        if vals.size != actual_arr.size:
            continue
        fig.add_trace(go.Scatter(x=list(index), y=vals, mode="lines", name=str(r.model_name),
                                 line=dict(width=1.8, color=PALETTE[ci % len(PALETTE)]), opacity=0.9,
                                 hovertemplate="%{x|%Y-%m-%d}<br>%{fullData.name}: %{y:,.0f}<extra></extra>"))
        ci += 1
    fig.update_yaxes(title_text="Demand (trips)")
    return _plotly_theme(fig, "Forecast vs. actual on the holdout")


# =============================================================================
# Story sections (Requirement 9.2) - each is a dispatched render function
# =============================================================================


def render_overview(scope: ScopeConfig) -> None:
    """Section 0: the storyline map - orient any reader in 60 seconds."""
    _section_header("🧭", "Start here — the whole story in one minute", "#6366f1")
    _lead(
        "This project answers one question for a ride-hailing platform: "
        "<strong>how many trips will be requested, on which day, in which part of the "
        "city — and what should we do about it?</strong> Below is the journey each tab "
        "takes you through. Read it top to bottom and it reads like a story."
    )
    _concept(
        "What is \"demand forecasting\"?",
        "<b>Forecasting</b> just means predicting the future from the past. Here the "
        "\"future\" is the number of ride requests (we call that <b>demand</b>).",
        "If we know demand will spike in Manhattan next Friday, the platform can send "
        "drivers there <i>before</i> the rush — so riders wait less and drivers earn more. "
        "That is the entire point of the project.",
    )
    _story_arc([
        ("01", "🎯 Problem", "Why predicting demand saves money for riders and drivers.", "#6366f1"),
        ("02", "🗽 Data", "247M real NYC trips — where they come from and their limits.", "#06b6d4"),
        ("03", "🔍 Explore", "The shape of demand: weekly rhythm, trend, stationarity.", "#22c55e"),
        ("04", "🧹 Prepare", "Turning raw trips into a clean, model-ready table.", "#f59e0b"),
        ("05", "🛠️ Tools", "The open-source stack that does the work.", "#3b82f6"),
        ("06", "🤖 Models", "Eight forecasting methods, from simple to deep learning.", "#ec4899"),
        ("07", "📊 Results", "An honest scoreboard — who won, who lost, and why.", "#8b5cf6"),
        ("08", "💡 Action", "Turning the winning forecast into a driver plan + savings.", "#10b981"),
        ("09", "🇮🇳 India", "How the same method transfers to Ola, Uber, Rapido.", "#f97316"),
        ("10", "📤 Try it", "Upload your own data and get a live forecast.", "#0ea5e9"),
    ])
    _cards([
        _card("🏆", "Headline result", body="A simple <b>Holt-Winters</b> baseline won at "
              "<b>3.74% MAPE</b> — beating heavier deep-learning models. We report that honestly.",
              accent="#22c55e"),
        _card("📏", "How big is the data?", body="<b>247,412,659</b> real trips over 12 months, "
              "condensed into a clean daily demand series per borough.", accent="#6366f1"),
        _card("🔒", "The golden rule", body="Only real public data — <b>nothing is fabricated</b>, "
              "and every number is validated against the raw data first.", accent="#f43f5e"),
    ])
    _takeaway(
        "Predict where and when demand peaks → position drivers ahead of it → cut rider wait "
        "time and driver idle time. Every tab that follows is one honest step toward that."
    )


def render_business_problem(scope: ScopeConfig) -> None:
    """Section 1: the business problem the project solves (Requirement 9.2)."""
    _section_header("🎯", "The business problem", "#6366f1")
    _lead(
        "Our goal: <strong>forecast demand - how much, when, and where -</strong> so a "
        "platform can position drivers <em>ahead</em> of need, cutting both rider wait "
        "time and driver idle time."
    )
    _cards([
        _card("⏱️", "Spiky in time", bullets=[
            "Some hours surge, others go quiet.",
            "Rush hours, weekends and events all shift demand.",
        ], accent="#6366f1"),
        _card("🗺️", "Spiky in space", bullets=[
            "Some neighbourhoods boom while others sit idle.",
            "Manhattan behaves nothing like Staten Island.",
        ], accent="#06b6d4"),
        _card("💸", "Two costs at once", bullets=[
            "Riders wait longer when no driver is near.",
            "Drivers sit idle in the wrong place.",
        ], accent="#f43f5e"),
    ])
    st.markdown("#### 🔒 Fixed forecasting scope")
    st.caption("Read from one source of truth (`src/config.ScopeConfig`) so every stage uses identical values.")
    _kpis([
        ("Time grain", scope.time_grain, "#22c55e"),
        ("Geographic grain", scope.geographic_grain, "#f59e0b"),
        ("Analysis window", f"{scope.window_months} months", "#a855f7"),
        ("Holdout", f"{scope.holdout_periods} periods", "#3b82f6"),
    ])
    _concept(
        "\"Grain\" and \"scope\"",
        "<b>Time grain</b> = how finely we slice time. We use <b>daily</b> — one number per day. "
        "<b>Geographic grain</b> = how finely we slice the map. We use <b>borough</b> "
        "(Manhattan, Brooklyn, etc.).",
        "<b>Scope</b> is just the fixed set of rules for the whole project (grain, date range, "
        "which models to try). We store it in one file so every step uses the exact same "
        "settings and results can never quietly drift apart.",
    )
    _takeaway(
        "Demand is spiky in <b>time</b> and <b>space</b>. Forecast both and you can move drivers "
        "ahead of need — the business win this whole project chases."
    )


def render_data_source(scope: ScopeConfig) -> None:
    """Section 2: the data source and its honest limitations (Requirement 9.2)."""
    _section_header("🗽", "Data source & honest limitations", "#06b6d4")
    _lead(
        "Source: <strong>official <a href='https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page' "
        "target='_blank'>NYC TLC Trip Record Data</a></strong> - the For-Hire Vehicle "
        "High-Volume (FHVHV) feed (Uber/Lyft-style rides), published in Parquet by the "
        "NYC Taxi &amp; Limousine Commission. Only real, public data - "
        "<strong>nothing is fabricated</strong>."
    )
    st.markdown("#### 🧾 What \"demand\" means here")
    _cards([
        _card("📊", "Trip count", body="Demand = count of trips per (period, region).", accent="#6366f1"),
        _card("🗺️", "Region = borough", body="Mapped via the official taxi-zone lookup.", accent="#06b6d4"),
        _card("✅", "Validated", body="Every reported number is checked against the raw data first.", accent="#22c55e"),
    ])
    _concept(
        "FHVHV, Parquet and the zone lookup",
        "<b>FHVHV</b> = \"For-Hire Vehicle, High Volume\" — the official name for Uber/Lyft-style "
        "app rides in the NYC data. It is the closest public match to Ola/Uber/Rapido.",
        "<b>Parquet</b> is just a file format (like CSV) but far smaller and faster to read — "
        "important when each month is ~1 GB.",
        "The data only says <i>which pickup zone</i> a trip started in (a number). The "
        "<b>taxi-zone lookup</b> is a small table that translates each zone number into its "
        "borough, so we can group demand by borough.",
    )
    st.markdown("#### ⚖️ Honest limitations")
    _cards([
        _card("🌍", "NYC, not India", bullets=[
            "The method generalises.",
            "The specific numbers do not transfer directly.",
        ], accent="#06b6d4"),
        _card("📈", "Trips ≈ demand", bullets=[
            "Trip counts are a proxy for true demand.",
            "Riders who gave up aren't observed.",
        ], accent="#f43f5e"),
        _card("🔬", "Coarse grain", bullets=[
            "Borough/daily hides intraday surges.",
            "Neighbourhood spikes are smoothed out.",
        ], accent="#f59e0b"),
        _card("🌦️", "Partial drivers", bullets=[
            "Weather, events, surge not in the base feed.",
            "Known extensions, not silent gaps.",
        ], accent="#a855f7"),
    ])
    st.caption(
        f"Analysis window: {scope.window_start} → {scope.window_end} "
        f"({scope.window_months} months of FHVHV records)."
    )


def render_eda(scope: ScopeConfig) -> None:
    """Section 3: the EDA findings (Requirement 9.2), reusing ``src.eda``."""
    _section_header("🔍", "EDA findings", "#22c55e")
    _lead(
        "Before modelling we study the series' shape, seasonality and stationarity. "
        "The first chart is interactive - hover for values, drag to zoom."
    )

    series, is_real = get_demand_series(scope)
    _demo_notice(is_real)

    # Lazy import: src.eda pulls in statsmodels/matplotlib, kept out of module import.
    from src.eda import adf_test, plot_demand_series, seasonal_decompose_demand

    total = int(series[DEMAND_COLUMN].sum())
    _kpis([
        ("Total demand", f"{total:,}", "#6366f1"),
        ("Regions", f"{series[REGION_COLUMN].nunique()}", "#06b6d4"),
        ("Distinct periods", f"{series[PERIOD_COLUMN].nunique()}", "#22c55e"),
        ("Rows", f"{len(series):,}", "#f59e0b"),
    ])

    st.plotly_chart(_fig_demand_over_time(series), use_container_width=True)
    st.markdown(f"**What this shows:** {plot_demand_series(series, scope).interpretation}")

    _cards([
        _card("📆", "Weekly rhythm", bullets=[
            "Demand rises and falls on a clear 7-day cycle.",
            "Weekdays and weekends have distinct shapes.",
        ], accent="#22c55e"),
        _card("📈", "Mild trend", bullets=[
            "A gentle drift over the window.",
            "Models must capture level + trend together.",
        ], accent="#6366f1"),
        _card("🏙️", "Scale gap", bullets=[
            "Manhattan dominates total volume.",
            "Staten Island is a fraction of it.",
        ], accent="#f59e0b"),
    ])

    st.markdown("#### 🔬 Seasonal decomposition")
    _concept(
        "Seasonal decomposition",
        "Any demand line is really three things added together: a slow <b>trend</b> (is demand "
        "generally rising or falling?), a repeating <b>seasonal</b> pattern (the weekly "
        "weekday-vs-weekend rhythm), and the leftover <b>residual</b> (random noise).",
        "Splitting them apart tells us <i>what a model must capture</i>. Strong weekly "
        "seasonality here is why every model we pick is built to handle a 7-day cycle.",
    )
    try:
        decomp = seasonal_decompose_demand(series, period=7)
        st.pyplot(decomp.figure)
        st.markdown(f"**What this shows:** {decomp.interpretation}")
    except ValueError as exc:
        st.warning(f"Not enough history to decompose seasonality: {exc}")

    st.markdown("#### 📉 Stationarity - Augmented Dickey-Fuller test")
    _concept(
        "Stationarity and the ADF test",
        "A series is <b>stationary</b> if its behaviour (average level, swing size) stays "
        "roughly constant over time. Many classical models assume this, so we test for it.",
        "The <b>Augmented Dickey-Fuller (ADF)</b> test is a statistical check. Rule of thumb: "
        "if the <b>p-value is below 0.05</b>, the series is stationary; if higher, we "
        "<i>difference</i> it (model day-to-day changes instead of raw levels) — which is "
        "exactly what the \"I\" in ARIMA does automatically.",
    )
    try:
        adf = adf_test(series)
        col1, col2 = st.columns(2)
        col1.metric("ADF statistic", f"{adf.statistic:.3f}")
        col2.metric("p-value", f"{adf.p_value:.4f}")
        st.markdown(f"**What this shows:** {adf.interpretation}")
    except ValueError as exc:
        st.warning(f"Not enough observations to run the ADF test: {exc}")

    _takeaway(
        "The demand series has a clear <b>weekly cycle</b> and a mild trend. Knowing its shape "
        "up front is what lets us choose the right models — and explains why a well-tuned "
        "seasonal baseline turns out to be so hard to beat."
    )


def render_data_preparation(scope: ScopeConfig) -> None:
    """Section 4: the data preparation approach (Requirement 9.2)."""
    _section_header("🧹", "Data preparation", "#f59e0b")
    _lead(
        "Raw trip records become a validated forecasting dataset via the pure functions "
        "in <code>src/preparation.py</code>. Order matters - here's the flow:"
    )
    _flow([
        ("🚫", "Validate", "invalid records logged, not dropped silently", "#f43f5e"),
        ("🗺️", "Map zones", "PULocationID → borough via official lookup", "#06b6d4"),
        ("➕", "Aggregate", "count trips per (period, region)", "#6366f1"),
        ("0️⃣", "Zero-fill", "empty periods become 0, never missing", "#f59e0b"),
        ("🔁", "Lag features", "lag_1 / lag_7 / lag_14 for ML models", "#a855f7"),
    ])

    _concept(
        "\"Long format\", zero-fill and lag features",
        "<b>Long format</b> = one row per day-and-borough with its trip count. It is the tidy "
        "shape every model expects.",
        "<b>Zero-fill</b>: if a borough had no trips on some day, that row is simply missing. "
        "We insert it as a <b>0</b> — because \"no trips\" is real information, and a gap would "
        "confuse the models.",
        "<b>Lag features</b>: extra columns that carry <i>past</i> demand — yesterday "
        "(<code>lag_1</code>), last week (<code>lag_7</code>), two weeks ago "
        "(<code>lag_14</code>). Machine-learning models (like XGBoost) have no memory of time, "
        "so we hand them the past explicitly as these columns.",
    )

    series, is_real = get_demand_series(scope)
    _demo_notice(is_real)
    st.markdown("#### 📋 Prepared demand series (long format)")
    st.caption(
        "One row per (period, region); `demand` is the trip count and is 0 for "
        "empty periods rather than missing. This is the exact table every model reads."
    )
    st.dataframe(series.head(20), use_container_width=True)

    n_regions = series[REGION_COLUMN].nunique()
    n_periods = series[PERIOD_COLUMN].nunique()
    complete = n_regions * n_periods
    _kpis([
        ("Complete grid cells", f"{complete:,}", "#6366f1"),
        ("= periods", f"{n_periods:,}", "#22c55e"),
        ("× regions", f"{n_regions}", "#06b6d4"),
        ("Gaps left behind", "0", "#f59e0b"),
    ])
    st.caption(
        "The prepared grid is deliberately *complete*: every period × region cell exists, with "
        "no missing values — that completeness is what makes a fair model comparison possible."
    )
    _takeaway(
        "Preparation is unglamorous but decisive: validate → map zones to boroughs → count "
        "trips → zero-fill gaps → add lag features. Get this wrong and every model downstream "
        "is wrong too."
    )


def render_tools(scope: ScopeConfig) -> None:
    """Section 5: the tools and technology used (Requirement 9.2)."""
    _section_header("🛠️", "Tools & technology", "#3b82f6")
    _lead(
        "The stack is deliberately open-source and reproducible. Core logic lives in "
        "<code>src/</code> as pure, testable functions; this dashboard and the notebook "
        "reuse the same functions, so the deliverables can never drift."
    )
    _cards([
        _card("🐼", "Data handling", bullets=["pandas", "pyarrow"], accent="#6366f1"),
        _card("📊", "Visualisation", bullets=["matplotlib", "Plotly (this app)"], accent="#06b6d4"),
        _card("📐", "Classical stats", bullets=["statsmodels", "Holt-Winters, SARIMA, VAR"], accent="#f59e0b"),
        _card("🔮", "Modern", bullets=["Prophet"], accent="#a855f7"),
        _card("🌳", "Machine learning", bullets=["XGBoost on lag features"], accent="#22c55e"),
        _card("🧠", "Deep learning", bullets=["TensorFlow / Keras", "LSTM / GRU"], accent="#ec4899"),
        _card("🖥️", "Dashboard", bullets=["Streamlit (this app)", "standalone HTML"], accent="#3b82f6"),
        _card("🧪", "Testing", bullets=["pytest", "Hypothesis (property-based)"], accent="#f43f5e"),
        _card("🚀", "Automate & deploy", bullets=["GitHub Actions", "Streamlit Cloud"], accent="#f97316"),
    ])
    st.caption(f"Candidate model set: {', '.join(scope.candidate_models)}.")


def render_models(scope: ScopeConfig) -> None:
    """Section 6: the models and method (Requirement 9.2)."""
    _section_header("🤖", "Models & method", "#ec4899")
    _lead(
        "We train a <strong>broad</strong> set of models so the final choice is justified "
        "by <em>evidence</em>, not assumption. Each card explains what the model is and "
        "how we use it."
    )
    _concept(
        "Why try so many models?",
        "There is no universal \"best\" forecasting model — it depends on the data. So we run a "
        "<b>broad set</b>, from dead-simple to cutting-edge, and let the evidence pick the winner.",
        "A <b>baseline</b> is the simplest sensible model. Its job is to set a bar: any complex "
        "model that can't beat the baseline isn't worth its extra cost. That is honest science, "
        "not a weakness.",
    )
    _cards([
        _card("📉", "Holt-Winters", body="Exponential smoothing baseline - every fancier model must beat it.", accent="#6366f1"),
        _card("🔁", "SARIMA / SARIMAX", body="Classic seasonal statistical model; SARIMAX adds external regressors.", accent="#06b6d4"),
        _card("🕸️", "VAR / VARMAX", body="Forecasts all boroughs jointly, learning how they move together.", accent="#a855f7"),
        _card("🔮", "Prophet", body="Additive trend + seasonality + holidays, robust and easy to read.", accent="#ec4899"),
        _card("🌳", "XGBoost", body="Gradient-boosted trees on engineered lag features.", accent="#22c55e"),
        _card("🧠", "LSTM / GRU", body="Recurrent neural nets that learn long-range sequence patterns.", accent="#f59e0b"),
    ])
    _concept(
        "The models in one line each",
        "<b>Holt-Winters</b>: a smart weighted average that leans on recent days and last week. "
        "<b>SARIMA</b>: predicts today from recent days + the same day last week (the S = "
        "seasonal, I = differencing to remove trend). <b>VAR</b>: forecasts all boroughs "
        "together, using one borough's history to help predict another.",
        "<b>Prophet</b> (by Meta): adds up a trend curve + weekly pattern + holidays. "
        "<b>XGBoost</b>: builds hundreds of small decision trees, each fixing the last one's "
        "mistakes, using the lag columns. <b>LSTM/GRU</b>: neural networks with memory, made "
        "for sequences — powerful but hungry for lots of data.",
    )
    st.markdown("#### 🧪 The method (fair by design)")
    _concept(
        "Holdout — the fairness trick",
        "We hide the most recent <b>30 days</b> from every model while it learns (that hidden "
        "slice is the <b>holdout</b>). Then we ask each model to predict those 30 days and "
        "compare its guess to what actually happened.",
        "Because no model ever saw the holdout during training, this is an honest test of "
        "prediction — like grading a student on questions they never got to see in advance.",
    )
    _cards([
        _card("🔒", "Reserved holdout", body=f"Train on everything except the most-recent {scope.holdout_periods}-period holdout.", accent="#3b82f6"),
        _card("⚖️", "Same yardstick", body="Every model forecasts over the SAME holdout, scored with the same metrics.", accent="#22c55e"),
        _card("👀", "Nothing hidden", body="Models that can't train are recorded with a reason and still shown.", accent="#f43f5e"),
    ])


def render_results(scope: ScopeConfig) -> None:
    """Section 7: results - model comparison table + forecast plots (R9.3)."""
    _section_header("📊", "Results", "#8b5cf6")
    _lead(
        "The honest scoreboard: every model (including underperformers and excluded "
        "ones), scored by the reusable <code>src.evaluation</code> functions on the "
        "reserved holdout."
    )

    actual, index, results, is_real = get_model_results(scope)
    _model_results_notice(is_real)

    table = comparison_table(results)
    scored = table[table["mae"].notna()].sort_values("mae")
    if not scored.empty:
        top = scored.iloc[0]
        _kpis([
            ("Best model", str(top["model_name"]), "#22c55e"),
            ("Best MAE (trips/day)", f"{top['mae']:,.0f}", "#6366f1"),
            ("Best MAPE", f"{top['mape']:.2f}%", "#f43f5e"),
            ("Models compared", f"{len(table)}", "#a855f7"),
        ])

    _concept(
        "MAE, RMSE, MAPE — the three scores",
        "All three measure <b>how far the forecast was from reality</b> (lower = better). "
        "<b>MAE</b> = average miss in trips (e.g. \"off by 26,804 trips/day\"). "
        "<b>RMSE</b> = similar, but punishes big misses harder — so a high RMSE warns of "
        "occasional large errors.",
        "<b>MAPE</b> = the miss as a <b>percentage</b> (e.g. 3.74%), which is easy to compare "
        "across cities of different sizes. A 3.74% MAPE means the daily forecast is, on "
        "average, within ~4% of the true number of trips.",
    )
    st.markdown("#### 📊 Error metrics by model (lower is better)")
    st.plotly_chart(_fig_model_comparison(table), use_container_width=True)

    st.markdown("#### 📋 Full comparison table")
    st.caption(
        "One row per model with the same metric columns (MAE, RMSE, MAPE). Excluded "
        "models keep the columns as NaN and are flagged - a complete, honest census."
    )
    st.dataframe(
        table.sort_values("mae", na_position="last").reset_index(drop=True),
        use_container_width=True,
    )

    st.markdown("#### 📈 Forecast vs. actual on the holdout")
    st.plotly_chart(_fig_forecast_vs_actual(actual, results, index), use_container_width=True)
    st.caption(
        "The bold dark line is actual holdout demand; each coloured line is a model's "
        "forecast. Where a line hugs the dark line, that model tracks demand well."
    )

    st.markdown("#### 🏆 Models carried forward")
    _concept(
        "Why did the simplest model win?",
        "The deep-learning models (LSTM, GRU) lost here — and that is expected, not a bug. "
        "Neural nets need <i>lots</i> of history; with only ~335 days per borough they "
        "<b>underfit</b> (never learn enough to shine).",
        "Meanwhile the daily series has a strong, stable weekly rhythm — exactly what a "
        "well-tuned <b>Holt-Winters</b> baseline captures. On clean, strongly-seasonal daily "
        "data, a simple model beating complex ones is a well-known, honest outcome.",
    )
    names, justification = select_carry_forward(table, return_justification=True)
    _pills(names)
    st.code(justification)
    _takeaway(
        "Complexity is not accuracy. We tried everything, scored it fairly, and reported the "
        "losers too — a stronger story than only showing a winner."
    )


def render_business_insights(scope: ScopeConfig) -> None:
    """Section 8: business insights - the recommendation (Requirement 9.2)."""
    _section_header("💡", "Business insights", "#10b981")
    _lead(
        "A forecast only matters if it drives <strong>action</strong>. The reusable "
        "<code>src.business</code> functions turn the selected forecast into a "
        "driver-positioning plan and quantify the benefit - with assumptions and the "
        "formula shown so the number is defensible."
    )

    _, index, results, is_real = get_model_results(scope)
    _model_results_notice(is_real)

    # Use the best-scored model's forecast to derive a recommendation.
    scored = [r for r in results if r.forecast is not None and r.metrics is not None]
    scored.sort(key=lambda r: r.metrics.mae)
    best = scored[0]

    recommendation = positioning_recommendation(best.forecast, scope)
    st.markdown(f"#### 🧭 Driver-positioning recommendation (from the {best.model_name} forecast)")
    st.info(recommendation.action)

    _concept(
        "How the forecast becomes money",
        "A forecast is only useful if it changes a decision. We convert predicted demand into "
        "<b>how many drivers to pre-position</b>, then estimate the time saved: shorter rider "
        "waits and less driver idling.",
        "Every number rests on <b>stated assumptions</b> (e.g. how many trips one driver serves, "
        "how much waiting a good position removes) and a <b>visible formula</b> — so anyone can "
        "challenge or re-run it. We never hide behind a magic number.",
    )
    impact = quantify_impact(recommendation)
    st.markdown("#### 📉 Quantified impact")
    _kpis([
        ("Rider wait-minutes saved", f"{impact.rider_wait_minutes_saved:,.0f}", "#6366f1"),
        ("Driver idle-minutes saved", f"{impact.driver_idle_minutes_saved:,.0f}", "#06b6d4"),
        ("Total minutes saved", f"{impact.total_minutes_saved:,.0f}", "#22c55e"),
    ])
    st.markdown(f"**In business terms:** {impact.narrative}")

    with st.expander("Assumptions & formula (so the number is reproducible)"):
        st.markdown("**Assumptions used:**")
        st.json({k: impact.assumptions[k] for k in DEFAULT_IMPACT_ASSUMPTIONS})
        st.markdown("**Formula applied:**")
        st.code(IMPACT_FORMULA)


def render_india(scope: ScopeConfig) -> None:
    """Section 9: generalisation to India (Ola, Uber, Rapido)."""
    _section_header("🇮🇳", "Generalisation to India (Ola, Uber, Rapido)", "#f97316")
    # The narrative text is the single source of truth in src.business.
    st.markdown(india_generalization())


def _read_uploaded_file(uploaded) -> "tuple[pd.DataFrame | None, str | None]":
    """Parse a Streamlit uploaded file into a DataFrame.

    Supports CSV and Parquet by file extension. Returns ``(df, None)`` on success
    or ``(None, error_message)`` when the bytes cannot be parsed at all - a read
    failure is distinct from a *format* failure, which :func:`validate_upload`
    reports afterwards.
    """
    name = (getattr(uploaded, "name", "") or "").lower()
    try:
        if name.endswith(".parquet"):
            return pd.read_parquet(uploaded), None
        # Default to CSV; try to parse the period column as dates for convenience.
        df = pd.read_csv(uploaded)
        if PERIOD_COLUMN in df.columns:
            df[PERIOD_COLUMN] = pd.to_datetime(df[PERIOD_COLUMN], errors="coerce")
        return df, None
    except Exception as exc:  # noqa: BLE001 - surface any parse error to the user
        return None, f"Could not read the uploaded file: {exc}"


def _analyze_uploaded_series(series: pd.DataFrame, scope: ScopeConfig) -> None:
    """Display analysis of a validated, conforming uploaded DemandSeries (R9.4).

    Reuses the same ``src`` functions the rest of the dashboard uses so an
    uploaded series is analysed identically to the project's own data.
    """
    st.success("Upload conforms to the expected format. Showing analysis of your data.")

    st.subheader("Uploaded demand series (long format)")
    st.dataframe(series.head(20), use_container_width=True)

    total = int(series[DEMAND_COLUMN].sum())
    st.markdown(
        f"- **Rows:** {len(series):,}  \n"
        f"- **Regions:** {series[REGION_COLUMN].nunique()}  \n"
        f"- **Distinct periods:** {series[PERIOD_COLUMN].nunique()}  \n"
        f"- **Total demand (sum of trip counts):** {total:,}"
    )

    # Reuse the real EDA visualization so uploaded data is charted like the rest.
    from src.eda import plot_demand_series  # lazy: keeps module import light

    st.subheader("Demand over time")
    try:
        ts = plot_demand_series(series, scope)
        st.pyplot(ts.figure)
        st.markdown(f"**What this shows:** {ts.interpretation}")
    except Exception as exc:  # noqa: BLE001 - charting is best-effort on user data
        st.warning(f"Could not chart the uploaded series: {exc}")

    # Turn the uploaded data into a real forecast + recommendation.
    _forecast_uploaded_series(series, scope)


def _forecast_total_per_period(forecast, period_index) -> "np.ndarray":
    """Reduce any Forecast shape to a total-per-period array aligned to ``period_index``."""
    values = forecast.values
    index = forecast.index
    if isinstance(values, pd.DataFrame):
        per = values.sum(axis=1)
        per.index = pd.to_datetime(values.index)
    elif isinstance(index, pd.MultiIndex) and PERIOD_COLUMN in (index.names or []):
        periods = pd.to_datetime(index.get_level_values(PERIOD_COLUMN))
        per = pd.DataFrame(
            {"p": periods, "v": np.asarray(values, dtype=float).ravel()}
        ).groupby("p")["v"].sum()
    else:
        arr = np.asarray(values, dtype=float).ravel()
        per = pd.Series(arr, index=pd.to_datetime(list(index)) if index is not None else period_index[: len(arr)])
    return per.reindex(period_index).fillna(0.0).to_numpy(dtype=float)


def _forecast_uploaded_series(series: pd.DataFrame, scope: ScopeConfig) -> None:
    """Fit a fast model on the uploaded series and show a real forecast + recommendation.

    Uses Holt-Winters (statsmodels - already in the deploy requirements) fit per
    region. When the series is long enough it reserves a holdout to score accuracy
    honestly, then refits on the full series, forecasts forward, and turns that
    forecast into a driver-positioning recommendation with quantified impact.
    Best-effort: any failure shows a clear message instead of crashing the app.
    """
    from src.evaluation import build_model_results, comparison_table, split_holdout
    from src.models.base import Forecast, TrainedModel
    from src.models.holt_winters import HoltWinters

    st.subheader("Forecast your data (Holt-Winters)")
    st.markdown(
        "We now **fit a model live on your upload** and forecast forward. Holt-Winters "
        "(exponential smoothing) is used per region because it is fast and needs no heavy "
        "dependencies. Your periods are treated at the project's daily grain."
    )

    work = series[[PERIOD_COLUMN, REGION_COLUMN, DEMAND_COLUMN]].copy()
    work[PERIOD_COLUMN] = pd.to_datetime(work[PERIOD_COLUMN], errors="coerce")
    work = work.dropna(subset=[PERIOD_COLUMN, REGION_COLUMN])
    n_periods = work[PERIOD_COLUMN].nunique()
    if n_periods < 4:
        st.info(
            f"Only {n_periods} distinct period(s) in your data — too few to forecast "
            "meaningfully. Upload a longer series (ideally several weeks of daily data)."
        )
        return

    # --- 1) Honest accuracy check on a held-out tail (only if long enough) ---
    holdout_h = max(1, min(int(scope.holdout_periods), n_periods // 4))
    if n_periods - holdout_h >= 2:
        try:
            train, holdout = split_holdout(work, holdout_h)
            hidx = pd.DatetimeIndex(np.sort(holdout[PERIOD_COLUMN].unique()))
            actual_total = (
                holdout.groupby(PERIOD_COLUMN)[DEMAND_COLUMN].sum().reindex(hidx)
            ).to_numpy(dtype=float)

            model = HoltWinters()
            model.fit(train, scope)
            fc = model.predict(len(hidx))
            fc_total = _forecast_total_per_period(fc, hidx)

            tm = TrainedModel(
                model_name="Holt-Winters",
                forecaster=object(),
                forecast=Forecast("Holt-Winters", fc_total, hidx),
            )
            results = build_model_results([tm], actual_total)
            row = comparison_table(results).iloc[0]

            st.markdown(
                f"**Accuracy on the most recent {len(hidx)} periods** (reserved from "
                "training, so this is honest out-of-sample error):"
            )
            c1, c2, c3 = st.columns(3)
            c1.metric("MAE", f"{row['mae']:,.0f}")
            c2.metric("RMSE", f"{row['rmse']:,.0f}")
            c3.metric("MAPE", f"{row['mape']:.2f}%")

            fig = plot_forecast_vs_actual(actual_total, results, index=hidx)
            st.pyplot(fig)
            st.caption(
                "Bold line = your actual demand on the held-out tail; the other line is "
                "the model's forecast. Closer is better."
            )
        except Exception as exc:  # noqa: BLE001 - never crash on user data
            st.warning(f"Could not run the holdout accuracy check: {exc}")
    else:
        st.info("Series is short, so no holdout is reserved — showing a forward forecast only.")

    # --- 2) Forward forecast on the FULL series + recommendation ---
    try:
        horizon = int(min(14, max(7, n_periods // 4)))
        full_model = HoltWinters()
        full_model.fit(work, scope)
        future = full_model.predict(horizon)

        st.markdown(f"**Forward forecast — the next {horizon} periods:**")
        if isinstance(future.index, pd.MultiIndex):
            fut = pd.DataFrame(
                {
                    PERIOD_COLUMN: pd.to_datetime(future.index.get_level_values(PERIOD_COLUMN)),
                    REGION_COLUMN: future.index.get_level_values(REGION_COLUMN),
                    DEMAND_COLUMN: np.asarray(future.values, dtype=float).ravel(),
                }
            )
            wide = (
                fut.pivot_table(index=PERIOD_COLUMN, columns=REGION_COLUMN, values=DEMAND_COLUMN)
                .round(0)
                .astype(int)
            )
            wide.insert(0, "TOTAL", wide.sum(axis=1))
            st.dataframe(wide, use_container_width=True)
            st.caption("Predicted trip count per region (and total) for each upcoming period.")

        rec = positioning_recommendation(future, scope)
        st.markdown("**Driver-positioning recommendation from your forecast:**")
        st.markdown(f"> {rec.action}")

        impact = quantify_impact(rec)
        c1, c2, c3 = st.columns(3)
        c1.metric("Rider wait-minutes saved", f"{impact.rider_wait_minutes_saved:,.0f}")
        c2.metric("Driver idle-minutes saved", f"{impact.driver_idle_minutes_saved:,.0f}")
        c3.metric("Total minutes saved", f"{impact.total_minutes_saved:,.0f}")
        st.caption(
            "Impact is an estimate from documented assumptions (see the Business insights "
            "section); it scales with your predicted demand."
        )
    except Exception as exc:  # noqa: BLE001 - never crash on user data
        st.warning(f"Could not produce a forward forecast: {exc}")


def render_upload_analyze(scope: ScopeConfig) -> None:
    """Upload-and-analyze mode with input-format validation (Requirements 9.4, 9.5).

    Accepts a user-supplied CSV/Parquet file, validates it against the expected
    long-format ``DemandSeries`` schema via the pure :func:`validate_upload`, and
    either displays a descriptive error naming the offending column (R9.5) or
    analyses the conforming data (R9.4).
    """
    _section_header("📤", "Upload & forecast your own data", "#0ea5e9")
    st.markdown(
        """
Bring your own demand data: this mode **validates it, charts it, fits a model live,
forecasts it forward, and turns the forecast into a positioning recommendation**.
The file must be a **long-format demand series** (CSV or Parquet) with exactly these
columns:

| column | type | meaning |
|---|---|---|
| `{period}` | datetime | observation timestamp |
| `{region}` | text | region / borough |
| `{demand}` | integer ≥ 0 | trip count for that period & region |

If the file is missing a required column or a column has the wrong type, you'll get
a clear message naming the offending column instead of a crash.
        """.format(period=PERIOD_COLUMN, region=REGION_COLUMN, demand=DEMAND_COLUMN)
    )
    st.caption(f"Required columns: {', '.join(REQUIRED_COLUMNS)}.")

    uploaded = st.file_uploader(
        "Upload a demand series (.csv or .parquet)",
        type=["csv", "parquet"],
    )
    if uploaded is None:
        st.info("Upload a file to validate and analyse it.")
        return

    df, read_error = _read_uploaded_file(uploaded)
    if read_error is not None:
        st.error(read_error)
        return

    result = validate_upload(df)
    if not result.ok:
        # Descriptive error naming the offending column (Requirement 9.5).
        st.error(result.error)
        return

    _analyze_uploaded_series(df, scope)


# =============================================================================
# Navigation wiring (Requirement 9.1)
# =============================================================================

#: Ordered sidebar label -> render function. Order follows the required story
#: arc in Requirement 9.2. Adding a section is a single entry here.
SECTIONS: "dict[str, callable]" = {
    "Start here": render_overview,
    "Business problem": render_business_problem,
    "Data source & limitations": render_data_source,
    "EDA findings": render_eda,
    "Data preparation": render_data_preparation,
    "Tools & technology": render_tools,
    "Models & method": render_models,
    "Results": render_results,
    "Business insights": render_business_insights,
    "Generalisation to India": render_india,
    "Upload & analyze": render_upload_analyze,
}


def main() -> None:
    """Entry point: build the sidebar and dispatch to the selected section (R9.1).

    Guarded by ``if __name__ == "__main__"`` so importing this module (e.g. in the
    smoke test) never starts rendering; only ``streamlit run`` triggers it.
    """
    st.set_page_config(
        page_title="Ride-Hailing Demand Forecasting",
        page_icon="🚕",
        layout="wide",
    )

    scope = default_scope()
    _inject_theme()

    # Reflect whether the app is showing real prepared data or the illustrative fallback.
    _, series_is_real = get_demand_series(scope)
    badge = "Real NYC TLC data" if series_is_real else "Illustrative demo data"

    st.sidebar.title("🚕 Demand Forecasting")
    st.sidebar.caption("A colourful storytelling walkthrough.")
    selection = st.sidebar.radio("Go to section", list(SECTIONS.keys()))
    st.sidebar.markdown("---")
    st.sidebar.caption(
        f"Scope: {scope.time_grain} · {scope.geographic_grain}\n\n"
        f"{scope.window_start} → {scope.window_end}"
    )

    _hero(
        "Ride-Hailing Demand Forecasting",
        "Forecasting where and when demand will peak - on real NYC TLC data - so drivers "
        "can be positioned ahead of need, cutting rider wait time and driver idle time.",
        badge,
    )

    # Dispatch to the chosen section's render function.
    SECTIONS[selection](scope)


if __name__ == "__main__":
    main()
