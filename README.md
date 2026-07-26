# 🚕 Ride-Hailing Demand Forecasting

> Forecasting ride-hailing demand — **how much, when, and where** — on real,
> official **NYC TLC** trip data, so a platform can position drivers *ahead of need*
> and cut both rider wait time and driver idle time.

[![Live app](https://img.shields.io/badge/🚀_Live_app-Open_the_dashboard-FF4B4B?style=for-the-badge)](https://ride-hailing-demand-forecasting-swagata-bhowmik.streamlit.app/)
&nbsp;
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
&nbsp;
![Data](https://img.shields.io/badge/Data-Real_NYC_TLC_(247M+_trips)-06b6d4?style=for-the-badge)
&nbsp;
![Tests](https://img.shields.io/badge/Tests-131_passing-22c55e?style=for-the-badge)

The project runs the **full data-science lifecycle** end to end: validate real data →
explore it → prepare it → compare a broad set of forecasting models honestly →
select the best → translate the forecast into a business recommendation → and deliver
it four ways (below).

> [!IMPORTANT]
> **Golden rule:** only real public data is used, nothing is fabricated, and every
> reported number is validated against the raw data before it is shown.

---

## 🔎 Explore this project — four ways, pick any

| | Deliverable | What it is |
|---|---|---|
| 🌐 | **[Live interactive app](https://ride-hailing-demand-forecasting-swagata-bhowmik.streamlit.app/)** | A hosted Streamlit dashboard — click and explore, **no setup**. Includes an **upload-your-own-data → live forecast** mode. |
| 📄 | **[Interactive report (HTML)](https://swagata-bhowmik.github.io/Ride-Hailing-Demand-Forecasting/)** | A self-contained visual report, hosted via GitHub Pages so it opens in any browser. Also downloadable as [`dashboard.html`](dashboard.html) to view completely offline (no Python, no internet). |
| 📓 | **[`notebook/demand_forecasting.ipynb`](notebook/demand_forecasting.ipynb)** | The deep, guided **technical notebook** — every step explained and run, with real charts, tables and numbers saved inline. |
| 📖 | **This README** | The whole story in one read — understand the project **without opening anything**. |

> [!TIP]
> **Headline result:** across a broad candidate set, the honest holdout winner is a
> **Holt-Winters** baseline at **3.74% MAPE** on system-wide daily demand — deep-learning
> models were trained too and reported honestly, even though they underperformed on this
> short daily series ([full scoreboard below](#-real-holdout-results)).

---

## 🧭 How it works — the pipeline

```mermaid
flowchart LR
    A["📥 Ingest<br/>NYC TLC FHVHV"] --> B["✅ Validate"]
    B --> C["🧹 Prepare<br/>zone→borough · zero-fill · lags"]
    C --> D["🔍 Explore<br/>trend · seasonality · ADF"]
    D --> E["🤖 Model<br/>8 candidates"]
    E --> F["📊 Evaluate<br/>holdout MAE/RMSE/MAPE"]
    F --> G["🏆 Select"]
    G --> H["💡 Recommend<br/>driver positioning"]
    H --> I["🚀 Deliver<br/>app · report · notebook"]
    classDef s1 fill:#6366f1,color:#fff,stroke:#4338ca;
    classDef s2 fill:#06b6d4,color:#fff,stroke:#0e7490;
    classDef s3 fill:#22c55e,color:#fff,stroke:#15803d;
    classDef s4 fill:#f59e0b,color:#fff,stroke:#b45309;
    classDef s5 fill:#ec4899,color:#fff,stroke:#be185d;
    class A,B s1; class C,D s2; class E,F s3; class G s4; class H,I s5;
```

Every arrow is real, tested code in `src/`. Nothing is hand-edited between steps, so
the numbers always trace back to the raw data.

---

## 🎯 The business problem

Ride-hailing demand is spiky in **time** and **space**: some hours and some
neighbourhoods surge while others go quiet. When drivers are not where riders will
be, two costs appear at once:

- **Riders wait longer** — worse experience, cancelled trips, lost revenue.
- **Drivers sit idle** — lower earnings, lower platform utilisation.

If we can forecast demand per region ahead of time, a platform can pre-position
supply where it will be needed and reduce both costs. The analysis is framed on New
York but written to **generalise to Ola / Uber / Rapido operations in India** (see
the *Generalisation to India* section of the dashboard).

---

## 🗽 Data source: NYC TLC FHVHV

- **Source:** official [NYC TLC Trip Record Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page),
  the **For-Hire Vehicle High Volume (FHVHV)** feed — the Uber/Lyft-style rides —
  published by the NYC Taxi & Limousine Commission in **Parquet** format.
- **Demand** is defined as the **count of trips** per `(period, region)`, where
  region is a **borough** (via the official taxi-zone lookup).
- **Scope** (fixed in one source of truth, `src/config.py::ScopeConfig`):
  - **Time grain:** daily
  - **Geographic grain:** borough (Manhattan, Brooklyn, Queens, Bronx, Staten Island, EWR)
  - **Analysis window:** 12 months ending 2026-04 (`2025-05` → `2026-04`)
  - **Holdout:** most-recent 30 days reserved for out-of-sample evaluation

### ⚖️ Honest limitations

- **It is NYC, not India.** Models train on New York boroughs. The *method*
  generalises; the specific numbers do not transfer directly.
- **Trip counts are a proxy for demand.** The feed records *completed* trips, so
  unmet demand (riders who gave up, requests with no driver) is not directly observed.
- **Borough/daily grain is coarse.** It suits stable multivariate forecasting but
  hides intraday and neighbourhood-level surges that matter operationally.
- **Exogenous drivers are partial.** Weather, events, and surge pricing are not in
  the base feed; where used they are added as candidate explanatory variables.
- Where a limitation cannot be resolved, it is **documented rather than hidden**.

> ⚠️ FHVHV monthly files are large (~1 GB each). Raw data lives in the git-ignored
> `data/` directory and is **never committed**. Downloads and heavy model fits are
> long-running jobs you run in your own terminal.

---

## 🗂️ Project structure

```
ride-hailing-demand-forecasting/
├── README.md
├── .gitignore                    # excludes data/, venvs, secrets, brief file
├── requirements.txt              # lightweight runtime (deployed app + CI auto-install this)
├── requirements-full.txt         # full stack (training, deep learning, notebook, tests)
├── .streamlit/config.toml        # dashboard server/theme config (committed, no secrets)
├── notebook/
│   └── demand_forecasting.ipynb  # deep guided-story notebook
├── src/
│   ├── config.py                 # ScopeConfig — single source of truth for scope
│   ├── validation.py             # Data_Validator (profiling + reconciliation)
│   ├── preparation.py            # Data_Preparation_Pipeline (pure functions)
│   ├── eda.py                    # EDA_Module (charts + plain-language readings)
│   ├── models/                   # one module per model family
│   ├── evaluation.py             # Evaluation_Framework (holdout, metrics, selection)
│   └── business.py               # Business_Module (recommendation + impact)
├── dashboard/
│   └── app.py                    # Streamlit storytelling dashboard
├── scripts/
│   └── validate_raw.py           # Phase-1 raw-data validation runner
├── tests/                        # pytest + Hypothesis property tests
├── .github/workflows/            # scheduled auto-refresh (free-tier)
└── data/                         # git-ignored; local Parquet lives here
```

---

## 💻 Running locally

Prerequisites: **Python 3.10–3.11** and `pip`.

```bash
# 1. (Recommended) create and activate a virtual environment
python -m venv .venv
# Windows:  .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate

# 2. Install the full dependency stack (training, deep learning, notebook, tests)
pip install -r requirements-full.txt

# 3. Put the real data in place (git-ignored)
#    Download FHVHV Parquet from the NYC TLC page above into data/,
#    e.g. data/fhvhv_2026-04.parquet

# 4. Phase 1 — validate the raw data first (real numbers, before any modeling)
python scripts/validate_raw.py

# 5. Explore the full analysis in the notebook
jupyter notebook notebook/demand_forecasting.ipynb

# 6. Launch the interactive dashboard
streamlit run dashboard/app.py
```

The dashboard runs even before the real pipeline artifacts exist: data-backed
sections fall back to a **clearly-labelled illustrative series** so the reusable
`src/` visualizations still render. Save a prepared series to
`data/demand_series.parquet` to populate the real numbers.

Run the test suite (example + property-based tests) with:

```bash
pytest
```

---

## 🤖 Models & results approach

A **broad** candidate set is trained so the final choice is justified by evidence,
not assumption. It spans every major forecasting family:

| Family | Model(s) |
|---|---|
| Baseline | Holt-Winters exponential smoothing |
| Classical univariate | SARIMA (+ SARIMAX exogenous variant) |
| Multivariate | VAR / VARMAX (joint multi-borough forecasting) |
| Modern | Prophet |
| Machine learning | XGBoost on lag features |
| Deep learning | LSTM (+ GRU variant) |

**Method:** the most-recent 30-day **holdout** is reserved and excluded from all
training. Every model forecasts over the *same* holdout, and all are scored with
the **same metrics** — **MAE, RMSE, MAPE** — so the comparison is fair. Results are
presented as an honest scoreboard:

- A **comparison table** with one row per model, including underperformers and any
  models that could not train (recorded with a reason, never silently dropped).
- **Forecast-vs-actual** overlays on the holdout.
- **Error-by-period** reporting so accuracy is not hidden behind a single aggregate.
- **3–5 models carried forward** for deep explanation, with justification based on
  the reported metrics.

### 📊 Real holdout results

Run on the real 12-month series (247,412,659 trips) via
`python scripts/train_models.py`, scored on the reserved 30-day holdout at the
system-wide total-daily-demand level (~700k trips/day):

| Model | MAE (trips/day) | RMSE | MAPE | |
|---|---|---|---|---|
| **Holt-Winters** | **26,804** | **36,574** | **3.74%** | 🥇 |
| Prophet | 30,585 | 40,476 | 4.42% | 🥈 |
| XGBoost | 32,179 | 45,384 | 4.43% | 🥉 |
| VAR | 32,614 | 47,765 | 4.47% | |
| SARIMA | 46,440 | 56,689 | 6.49% | |
| SARIMAX | 46,440 | 56,689 | 6.49% | |
| LSTM | 53,935 | 75,907 | 7.23% | |
| GRU | 88,056 | 111,740 | 12.01% | |

The simple **Holt-Winters** baseline is the most accurate here — an honest result:
a well-tuned baseline beating heavier models is common on a clean, strongly-weekly
daily series, and it is reported rather than hidden. The deep-learning models
(**LSTM, GRU**) underperform the classical ones, which is expected on a short daily
series (≈335 training days per borough): recurrent nets are data-hungry, so they
underfit here — again, reported rather than hidden. Carry-forward set:
Holt-Winters, Prophet, XGBoost, VAR, SARIMA. These real numbers are persisted to
`dashboard/model_results.json` and rendered by both dashboards.

> LSTM/GRU require `tensorflow`, which currently has no build for Python 3.14. They
> were trained in a separate Python 3.11 environment via
> `scripts/add_deep_learning.py`, which merges their real results into the artifact.

The selected forecast is then turned into a **driver-positioning recommendation**
with a **quantified impact** (rider wait-minutes and driver idle-minutes saved),
showing the assumptions and formula so the number is reproducible.

---

## 🚀 Deployment

The dashboard is deployed on **Streamlit Community Cloud** and runs at a public URL
with zero setup for the visitor:

> **🌐 https://ride-hailing-demand-forecasting-swagata-bhowmik.streamlit.app/**

The deployment is deliberately lean. Streamlit serves `dashboard/app.py` and installs
from the lightweight root `requirements.txt` (pandas, pyarrow, matplotlib, statsmodels,
plotly, streamlit), so the free-tier build stays fast. The app reads the committed
prepared artifacts (`dashboard/demand_series.csv`, `dashboard/model_results.json`) and
fits only the fast Holt-Winters model at runtime — it never needs the heavy training
packages (tensorflow, prophet, xgboost), which live in `requirements-full.txt` for
local work. Server and theme settings ship in `.streamlit/config.toml` (headless mode,
a 50 MB upload cap for the upload-and-forecast mode, taxi-yellow theme); secrets are
kept out of the repository.

> **🌐 Live dashboard:** https://ride-hailing-demand-forecasting-swagata-bhowmik.streamlit.app/

---

## ⚙️ Automation

A free-tier **GitHub Actions** workflow (`.github/workflows/`) re-runs the
prepare → forecast pipeline on a schedule and records success/failure in the run
logs, demonstrating operational maturity without paid infrastructure.

---

## ✅ What's delivered

- [x] 🗽 Validated **247M+ real NYC TLC trips** into a clean 12-month daily borough demand series
- [x] 🤖 Compared **8 forecasting models** on a reserved holdout (MAE/RMSE/MAPE) — honest scoreboard, losers included
- [x] 💡 Turned the forecast into a **driver-positioning recommendation** with quantified impact
- [x] 🌐 Shipped a **live Streamlit app** (with upload-and-forecast), a 📄 **hosted HTML report**, and a 📓 **guided notebook**
- [x] 🧪 **131 automated tests** (example + property-based) and an ⚙️ auto-refresh GitHub Action
- [x] 📦 Published prepared artifacts so the deployed dashboard shows **real numbers**

## 🔭 Roadmap (future work)

- [ ] ⏳ Expand the analysis window to **18–24 months** for a longer seasonal history
- [ ] 🕐 Add an **hourly** forecasting grain to capture intraday surges
- [ ] 📍 Drill the geographic grain down to **taxi zone** where data density allows
- [ ] 🌦️ Incorporate **exogenous signals** (weather, holidays, events) as model inputs
- [ ] 📉 Add **prediction intervals** (uncertainty), not just point estimates
- [ ] 🇮🇳 Pilot the **India generalisation** on an equivalent open dataset

---

## 📄 License & data attribution

Analysis code is provided for portfolio/educational use. Trip data is © the
**NYC Taxi & Limousine Commission**, used under their public data terms; see the
[NYC TLC Trip Record Data page](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page).
