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


def get_demand_series(scope: ScopeConfig) -> "tuple[pd.DataFrame, bool]":
    """Return ``(series, is_real)`` - the prepared series if on disk, else the demo.

    Looks for a prepared demand series saved by the pipeline (``data/demand_series
    .parquet`` or ``.csv``). If found it is used as-is (``is_real=True``); otherwise
    a clearly-labelled illustrative series is returned (``is_real=False``) so the
    reused ``src`` charts still render.
    """
    data_dir = _PROJECT_ROOT / "data"
    for candidate in ("demand_series.parquet", "demand_series.csv"):
        path = data_dir / candidate
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
# Story sections (Requirement 9.2) - each is a dispatched render function
# =============================================================================


def render_business_problem(scope: ScopeConfig) -> None:
    """Section 1: the business problem the project solves (Requirement 9.2)."""
    st.header("The business problem")
    st.markdown(
        """
Ride-hailing demand is spiky in **time** and **space**: some hours and some
neighbourhoods surge while others go quiet. When drivers are not where riders will
be, two costs appear at once:

- **Riders wait longer** (worse experience, cancelled trips, lost revenue).
- **Drivers sit idle** (lower earnings, lower platform utilisation).

**The goal of this project is to forecast demand - how much, when, and where -**
so a platform can *position drivers ahead of need*, cutting both rider wait time
and driver idle time.

We frame it as a full data-science lifecycle: validate real data, explore it,
prepare it, compare a broad set of forecasting models honestly, pick the best, and
translate that forecast into an operational driver-positioning recommendation.
        """
    )
    st.subheader("Fixed forecasting scope")
    st.markdown(
        f"""
The scope is fixed up front and read from a single source of truth
(`src/config.ScopeConfig`) so every stage uses the same values:

- **Time grain:** `{scope.time_grain}`
- **Geographic grain:** `{scope.geographic_grain}`
- **Analysis window:** `{scope.window_start}` → `{scope.window_end}`
  ({scope.window_months} months)
- **Holdout reserved for evaluation:** {scope.holdout_periods} periods
        """
    )


def render_data_source(scope: ScopeConfig) -> None:
    """Section 2: the data source and its honest limitations (Requirement 9.2)."""
    st.header("Data source & honest limitations")
    st.markdown(
        """
**Source: official [NYC TLC Trip Record Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)**,
specifically the **For-Hire Vehicle High Volume (FHVHV)** feed - the Uber/Lyft-style
rides - published by the NYC Taxi & Limousine Commission in Parquet format.

Only **real, public** data is used. Demand is defined as the **count of trips** per
`(period, region)`, where region is a borough (via the official taxi-zone lookup).
        """
    )
    st.subheader("Honest limitations")
    st.markdown(
        """
- **It is NYC, not India.** The models are trained on New York boroughs. The
  *method* generalises to Ola/Uber/Rapido (see the India section), but the specific
  numbers do not transfer directly.
- **Trip counts are a proxy for demand.** The TLC feed records *completed* trips, so
  unmet demand (riders who gave up, or requests with no driver) is not directly
  observed.
- **Borough/daily grain is coarse.** It suits stable multivariate forecasting but
  hides intraday and neighbourhood-level surges that matter operationally.
- **Exogenous drivers are partial.** Weather, events, and surge pricing are not in
  the base feed; where used they are added as candidate explanatory variables.
- **Every reported number is validated** against the raw data before it is shown -
  nothing is fabricated. Where a limitation cannot be resolved, it is documented
  rather than hidden.
        """
    )
    st.caption(
        f"Analysis window: {scope.window_start} → {scope.window_end} "
        f"({scope.window_months} months of FHVHV records)."
    )


def render_eda(scope: ScopeConfig) -> None:
    """Section 3: the EDA findings (Requirement 9.2), reusing ``src.eda``."""
    st.header("EDA findings")
    st.markdown(
        "Exploring the demand series *before* modelling: its shape over time, its "
        "seasonality, and its stationarity. Charts and their plain-language readings "
        "come straight from the reusable `src.eda` functions."
    )

    series, is_real = get_demand_series(scope)
    _demo_notice(is_real)

    # Lazy import: src.eda pulls in statsmodels/matplotlib, kept out of module import.
    from src.eda import adf_test, plot_demand_series, seasonal_decompose_demand

    st.subheader("Demand over time (Requirement 3.1)")
    ts = plot_demand_series(series, scope)
    st.pyplot(ts.figure)
    st.markdown(f"**What this shows:** {ts.interpretation}")

    st.subheader("Seasonal decomposition (Requirement 3.2)")
    try:
        decomp = seasonal_decompose_demand(series, period=7)
        st.pyplot(decomp.figure)
        st.markdown(f"**What this shows:** {decomp.interpretation}")
    except ValueError as exc:
        st.warning(f"Not enough history to decompose seasonality: {exc}")

    st.subheader("Stationarity - Augmented Dickey-Fuller test (Requirement 3.3)")
    try:
        adf = adf_test(series)
        col1, col2 = st.columns(2)
        col1.metric("ADF statistic", f"{adf.statistic:.3f}")
        col2.metric("p-value", f"{adf.p_value:.4f}")
        st.markdown(f"**What this shows:** {adf.interpretation}")
    except ValueError as exc:
        st.warning(f"Not enough observations to run the ADF test: {exc}")


def render_data_preparation(scope: ScopeConfig) -> None:
    """Section 4: the data preparation approach (Requirement 9.2)."""
    st.header("Data preparation")
    st.markdown(
        """
Raw trip records are reshaped into a validated forecasting dataset by the pure
functions in `src/preparation.py`. Every step is validated against ground truth
(the "golden rule"). The pipeline, in order:

1. **Invalid-record handling** (`apply_validity_rules`) - records that fail the
   validation checks from Phase 1 (e.g. pickups outside the file's month, negative
   fares/counts) get a *documented* handling rule and a `HandlingLog`.
2. **Zone → borough mapping** (`map_zones_to_regions`) - `PULocationID` is joined to
   the official taxi-zone lookup to materialise the borough (Geographic grain).
3. **Aggregation** (`aggregate_demand`) - trips are counted per `(period, region)`
   at the daily grain. Totals *reconcile* with the raw valid record count.
4. **Zero-fill** (`fill_missing_periods`) - every `(period, region)` in the window
   is present; periods with no trips are explicit **0**, never omitted.
5. **Lag features** (`add_lag_features`) - `lag_1`, `lag_7`, `lag_14` per region for
   the machine-learning models (NaN for the first *k* periods of each region).
        """
    )

    series, is_real = get_demand_series(scope)
    _demo_notice(is_real)
    st.subheader("Prepared demand series (long format)")
    st.caption(
        "One row per (period, region); `demand` is the trip count and is 0 for "
        "empty periods rather than missing."
    )
    st.dataframe(series.head(20), use_container_width=True)

    total = int(series[DEMAND_COLUMN].sum())
    st.markdown(
        f"- **Rows:** {len(series):,}  \n"
        f"- **Regions:** {series[REGION_COLUMN].nunique()}  \n"
        f"- **Distinct periods:** {series[PERIOD_COLUMN].nunique()}  \n"
        f"- **Total demand (sum of trip counts):** {total:,}"
    )


def render_tools(scope: ScopeConfig) -> None:
    """Section 5: the tools and technology used (Requirement 9.2)."""
    st.header("Tools & technology")
    st.markdown(
        """
The stack is deliberately open-source and reproducible:

| Layer | Tools | Why |
|---|---|---|
| Data handling | **pandas**, **pyarrow** | Read the large FHVHV Parquet files; reshape to the demand series. |
| Visualisation | **matplotlib** | EDA charts and forecast-vs-actual overlays. |
| Classical / statistical | **statsmodels** (Holt-Winters, SARIMA/SARIMAX, VAR/VARMAX; ADF, decomposition, ACF/PACF) | Baseline + classical + multivariate forecasting and diagnostics. |
| Modern forecasting | **Prophet** | Trend/seasonality/holiday decomposition out of the box. |
| Machine learning | **XGBoost** | Gradient-boosted trees on lag features. |
| Deep learning | **TensorFlow / Keras** (LSTM, GRU) | Sequence models for demand. |
| Dashboard | **Streamlit** | This interactive storytelling app. |
| Testing | **pytest**, **Hypothesis** | Example tests + property-based tests for the 13 correctness properties. |
| Automation / deploy | **GitHub Actions**, **Streamlit Community Cloud** | Free-tier scheduled refresh and public hosting. |

The core logic lives in `src/` as pure, testable functions; both this dashboard
and the notebook **reuse the same functions**, so the two deliverables can never
drift apart.
        """
    )
    st.caption(f"Candidate model set: {', '.join(scope.candidate_models)}.")


def render_models(scope: ScopeConfig) -> None:
    """Section 6: the models and method (Requirement 9.2)."""
    st.header("Models & method")
    st.markdown(
        """
A **broad** set of models is trained so the final choice is justified by evidence,
not assumption. The candidate set spans every major family:

- **Baseline:** Holt-Winters exponential smoothing.
- **Classical univariate:** SARIMA (and a SARIMAX exogenous variant).
- **Multivariate:** VAR / VARMAX - jointly forecasts all boroughs at once.
- **Modern:** Prophet.
- **Machine learning:** XGBoost on lag features.
- **Deep learning:** LSTM (and a GRU variant).

**Method:** each model is trained on everything *except* the most-recent
`{holdout}`-period **holdout**, which is reserved for out-of-sample scoring. Every
model forecasts over the *same* holdout so the comparison is fair. Models that
cannot train on the prepared data are **not silently dropped** - they are recorded
with a reason and still shown in the comparison.
        """.format(holdout=scope.holdout_periods)
    )
    st.markdown(
        "All models share one `Forecaster` interface (`src/models/base.py`), and "
        "`train_all` trains every candidate, catching per-model failures so one "
        "failure never aborts the others."
    )


def render_results(scope: ScopeConfig) -> None:
    """Section 7: results - model comparison table + forecast plots (R9.3)."""
    st.header("Results")
    st.markdown(
        "The honest scoreboard: the **model-comparison table** (every model, "
        "including underperformers and excluded ones) and the **forecast-vs-actual** "
        "overlay on the holdout. Both are produced by the reusable `src.evaluation` "
        "functions."
    )

    actual, index, results, is_real = get_model_results(scope)
    _model_results_notice(is_real)

    st.subheader("Model comparison table (Requirements 6.3, 9.3)")
    table = comparison_table(results)
    st.caption(
        "One row per model with the same metric columns (MAE, RMSE, MAPE). Excluded "
        "models keep the columns as NaN and are flagged, so the comparison is a "
        "complete, honest census - lower error is better."
    )
    st.dataframe(
        table.sort_values("mae", na_position="last").reset_index(drop=True),
        use_container_width=True,
    )

    st.subheader("Forecast vs. actual on the holdout (Requirements 6.4, 9.3)")
    fig = plot_forecast_vs_actual(actual, results, index=index)
    st.pyplot(fig)
    st.caption(
        "The bold black line is the actual holdout demand; each coloured line is a "
        "model's forecast. Where a line hugs the black line, that model tracks real "
        "demand well."
    )

    st.subheader("Models carried forward (Requirement 6.6)")
    names, justification = select_carry_forward(table, return_justification=True)
    st.markdown(f"**Selected:** {', '.join(names)}")
    st.code(justification)


def render_business_insights(scope: ScopeConfig) -> None:
    """Section 8: business insights - the recommendation (Requirement 9.2)."""
    st.header("Business insights")
    st.markdown(
        "The point of the forecast is action. The reusable `src.business` functions "
        "turn the selected forecast into a **driver-positioning recommendation** and "
        "quantify the benefit, showing the assumptions and the formula so the number "
        "is defensible."
    )

    _, index, results, is_real = get_model_results(scope)
    _model_results_notice(is_real)

    # Use the best-scored model's forecast to derive a recommendation.
    scored = [r for r in results if r.forecast is not None and r.metrics is not None]
    scored.sort(key=lambda r: r.metrics.mae)
    best = scored[0]
    st.markdown(f"Deriving the recommendation from the **{best.model_name}** forecast.")

    recommendation = positioning_recommendation(best.forecast, scope)
    st.subheader("Driver-positioning recommendation (Requirement 7.1)")
    st.markdown(f"> {recommendation.action}")

    impact = quantify_impact(recommendation)
    st.subheader("Quantified impact (Requirements 7.2, 7.3)")
    col1, col2, col3 = st.columns(3)
    col1.metric("Rider wait-minutes saved", f"{impact.rider_wait_minutes_saved:,.0f}")
    col2.metric("Driver idle-minutes saved", f"{impact.driver_idle_minutes_saved:,.0f}")
    col3.metric("Total minutes saved", f"{impact.total_minutes_saved:,.0f}")
    st.markdown(f"**In business terms:** {impact.narrative}")

    with st.expander("Assumptions & formula (so the number is reproducible)"):
        st.markdown("**Assumptions used:**")
        st.json({k: impact.assumptions[k] for k in DEFAULT_IMPACT_ASSUMPTIONS})
        st.markdown("**Formula applied:**")
        st.code(IMPACT_FORMULA)


def render_india(scope: ScopeConfig) -> None:
    """Section 9: generalisation to India (Ola, Uber, Rapido)."""
    st.header("Generalisation to India (Ola, Uber, Rapido)")
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


def render_upload_analyze(scope: ScopeConfig) -> None:
    """Upload-and-analyze mode with input-format validation (Requirements 9.4, 9.5).

    Accepts a user-supplied CSV/Parquet file, validates it against the expected
    long-format ``DemandSeries`` schema via the pure :func:`validate_upload`, and
    either displays a descriptive error naming the offending column (R9.5) or
    analyses the conforming data (R9.4).
    """
    st.header("Upload & analyze your own data")
    st.markdown(
        """
Bring your own demand data and analyse it with the same pipeline. The file must be
a **long-format demand series** (CSV or Parquet) with exactly these columns:

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

    st.sidebar.title("🚕 Demand Forecasting")
    st.sidebar.caption("A storytelling walkthrough of the project.")
    selection = st.sidebar.radio("Go to section", list(SECTIONS.keys()))
    st.sidebar.markdown("---")
    st.sidebar.caption(
        f"Scope: {scope.time_grain} · {scope.geographic_grain} · "
        f"{scope.window_start} → {scope.window_end}"
    )

    # Dispatch to the chosen section's render function.
    SECTIONS[selection](scope)


if __name__ == "__main__":
    main()
