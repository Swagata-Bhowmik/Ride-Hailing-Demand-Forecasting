# 🚕 Ride-Hailing Demand Forecasting

Forecasting ride-hailing demand — **how much, when, and where** — on real,
official **NYC TLC** trip data, so a platform can position drivers *ahead of need*
and cut both rider wait time and driver idle time.

The project runs the full data-science lifecycle end to end: validate real data →
explore it → prepare it → compare a broad set of forecasting models honestly →
select the best → translate the forecast into a business recommendation → deliver
it as a deep notebook, an interactive dashboard, free-tier automation, and a live
public deployment.

> **Golden rule:** only real public data is used, nothing is fabricated, and every
> reported number is validated against the raw data before it is shown.

---

## The business problem

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

## Data source: NYC TLC FHVHV

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

### Honest limitations

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

## Project structure

```
ride-hailing-demand-forecasting/
├── README.md
├── .gitignore                    # excludes data/, venvs, secrets, brief file
├── requirements.txt              # full stack (training + deep learning)
├── requirements-dashboard.txt    # lightweight subset for cloud deployment
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

## Running locally

Prerequisites: **Python 3.10–3.11** and `pip`.

```bash
# 1. (Recommended) create and activate a virtual environment
python -m venv .venv
# Windows:  .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate

# 2. Install the full dependency stack
pip install -r requirements.txt

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

## Models & results approach

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

### Real holdout results

Run on the real 12-month series (247,412,659 trips) via
`python scripts/train_models.py`, scored on the reserved 30-day holdout at the
system-wide total-daily-demand level (~700k trips/day):

| Model | MAE (trips/day) | RMSE | MAPE |
|---|---|---|---|
| Holt-Winters | 26,804 | 36,574 | 3.74% |
| Prophet | 30,585 | 40,476 | 4.42% |
| XGBoost | 32,179 | 45,384 | 4.43% |
| VAR | 32,614 | 47,765 | 4.47% |
| SARIMA | 46,440 | 56,689 | 6.49% |
| SARIMAX | 46,440 | 56,689 | 6.49% |
| LSTM / GRU | — | — | excluded where `tensorflow` is not installed |

The simple **Holt-Winters** baseline is the most accurate here — an honest result:
a well-tuned baseline beating heavier models is common on a clean, strongly-weekly
daily series, and it is reported rather than hidden. Carry-forward set:
Holt-Winters, Prophet, XGBoost, VAR, SARIMA. These real numbers are persisted to
`dashboard/model_results.json` and rendered by both dashboards.

The selected forecast is then turned into a **driver-positioning recommendation**
with a **quantified impact** (rider wait-minutes and driver idle-minutes saved),
showing the assumptions and formula so the number is reproducible.

---

## Deployment (Streamlit Community Cloud)

The dashboard is designed to be hosted for free at a public URL so anyone can open
it **without any local setup** (Requirement 11).

**The deploy is performed by you (the project owner) from your own Streamlit
Community Cloud account.** Steps:

1. Push this repository to GitHub (public).
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **New app** and select this repository and branch.
4. Set **Main file path** to `dashboard/app.py`.
5. Under **Advanced settings**, set the **Python version** to 3.10 or 3.11 and set
   the **requirements file** to **`requirements-dashboard.txt`**.
   - Why: the dashboard itself only needs pandas, pyarrow, matplotlib, statsmodels,
     and streamlit. The full `requirements.txt` pulls in heavy training packages
     (tensorflow, prophet, xgboost) that can exceed the free tier's build/memory
     limits. Use the full file only for local training and the notebook.
6. Click **Deploy**. Streamlit installs the dependencies and serves the app at a
   public `*.streamlit.app` URL.

The committed `.streamlit/config.toml` supplies sensible server and theme settings
(headless mode, a 50 MB upload cap for the *Upload & analyze* mode, and a
taxi-yellow theme). Secrets are **not** stored there — they belong in
`.streamlit/secrets.toml`, which is git-ignored.

> After deploying, add the live URL here:
> **Live dashboard:** _`https://<your-app>.streamlit.app`_ (add after deploy)

---

## Automation

A free-tier **GitHub Actions** workflow (`.github/workflows/`) re-runs the
prepare → forecast pipeline on a schedule and records success/failure in the run
logs, demonstrating operational maturity without paid infrastructure.

---

## Roadmap

- [ ] Expand the Analysis_Window to 18–24 months for a longer seasonal history.
- [ ] Add an **hourly** forecasting grain alongside daily to capture intraday surges.
- [ ] Drill the geographic grain down to **taxi zone** where data density allows.
- [ ] Incorporate **exogenous signals** (weather, holidays, events) as model inputs.
- [ ] Add **prediction intervals** (uncertainty) to forecasts, not just point estimates.
- [x] Publish the prepared demand series artifact so the deployed dashboard shows
      real numbers instead of the illustrative fallback (committed
      `dashboard/demand_series.csv` for the series and `dashboard/model_results.json`
      for the real model scoreboard).
- [ ] Pilot the India generalisation on an equivalent open dataset where available.

---

## License & data attribution

Analysis code is provided for portfolio/educational use. Trip data is © the
**NYC Taxi & Limousine Commission**, used under their public data terms; see the
[NYC TLC Trip Record Data page](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page).
