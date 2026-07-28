# 🚕 Ride-Hailing Demand Forecasting

> Forecasting ride-hailing demand — **how much, when, and where** — on **247M+ real NYC TLC
> trips**, so a platform can position drivers *ahead of need* and cut both rider wait time and
> driver idle time.

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
&nbsp;
![Data](https://img.shields.io/badge/Data-Real_NYC_TLC_(247M+_trips)-06b6d4?style=for-the-badge)
&nbsp;
![Best model](https://img.shields.io/badge/Best_model-Holt--Winters_3.74%25_MAPE-22c55e?style=for-the-badge)
&nbsp;
![Tests](https://img.shields.io/badge/Tests-131_passing-f43f5e?style=for-the-badge)

---

## 🔎 Explore this project — three ways

<table>
<tr>
<td width="33%" align="center">

### 📓 Notebook
The deep, guided technical walkthrough — every step explained and run, with real charts,
tables and numbers inline.

**➡️ [`notebook/demand_forecasting.ipynb`](notebook/demand_forecasting.ipynb)**

</td>
<td width="33%" align="center">

### 🚀 Live dashboard
A hosted, interactive Streamlit app — click and explore, **no setup**. Includes an
**upload-your-own-data → live forecast** mode.

**➡️ [Open the live app](https://ride-hailing-demand-forecasting-swagata-bhowmik.streamlit.app/)**

</td>
<td width="33%" align="center">

### 📄 Offline dashboard
A self-contained visual report — **open in any browser, no Python, no internet**. Also hosted
via GitHub Pages.

**➡️ [Open the report](https://swagata-bhowmik.github.io/Ride-Hailing-Demand-Forecasting/)** ·
[`dashboard.html`](dashboard.html)

</td>
</tr>
</table>

---

## 🎯 What this is

A full, end-to-end **time-series demand-forecasting** project on the official
[NYC TLC FHVHV feed](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page) (the
Uber/Lyft-style rides). It runs the whole data-science lifecycle: **validate → explore →
prepare → compare a broad set of models honestly → select the best → turn the forecast into a
business recommendation → deliver it**.

> **Golden rule:** only real public data, nothing fabricated, and every reported number is
> validated against the raw data before it is shown.

**Data source:** the official NYC TLC Trip Record Data (FHVHV feed).

## 🏆 Headline results

Across **8 forecasting models** (Holt-Winters, SARIMA/SARIMAX, VAR, Prophet, XGBoost, LSTM, GRU),
scored on a reserved 30-day holdout, the honest winner is a **Holt-Winters baseline at 3.74%
MAPE** — beating the heavier deep-learning models, which is expected on a short, strongly-weekly
daily series and is reported rather than hidden.

| Model | MAE (trips/day) | RMSE | MAPE | |
|---|---|---|---|---|
| **Holt-Winters** | **26,804** | **36,574** | **3.74%** | 🥇 |
| Prophet | 30,585 | 40,476 | 4.42% | 🥈 |
| XGBoost | 32,179 | 45,384 | 4.43% | 🥉 |
| VAR | 32,614 | 47,765 | 4.47% | |
| SARIMA / SARIMAX | 46,440 | 56,689 | 6.49% | |
| LSTM | 53,935 | 75,907 | 7.23% | |
| GRU | 88,056 | 111,740 | 12.01% | |

## 🧭 How it works

```
📥 Ingest → ✅ Validate → 🧹 Prepare → 🔍 Explore → 🤖 Model → 📊 Evaluate → 🏆 Select → 💡 Recommend → 🚀 Deliver
```

The selected forecast is turned into a **driver-positioning recommendation** with a
**quantified impact** (rider wait-minutes and driver idle-minutes saved), showing the assumptions
and formula so the number is reproducible.

## 🛠️ Tech stack

Python · pandas · pyarrow · statsmodels · Prophet · XGBoost · TensorFlow/Keras · matplotlib ·
Plotly · Streamlit · pytest + Hypothesis · GitHub Actions.

## 💻 Run locally

```bash
python -m venv .venv && .venv\Scripts\activate      # Windows
pip install -r requirements-full.txt                # full stack (training, DL, notebook, tests)
python scripts/validate_raw.py                      # validate real data first
python scripts/build_demand_series.py               # build the prepared series
python scripts/train_models.py                      # fit all models
streamlit run dashboard/app.py                      # launch the live dashboard
pytest                                              # run the 131 tests
```

The deployed app installs the lightweight root `requirements.txt` and fits only the fast
Holt-Winters model at runtime, reading committed artifacts for the rest; the heavy training stack
lives in `requirements-full.txt`.

## ⚖️ Honest limitations

- **NYC, not India** — the method generalises to Ola/Uber/Rapido; the specific numbers do not.
- **Trips ≈ demand** — only completed trips are recorded; unmet demand isn't observed.
- **Borough/daily grain is coarse** — hides intraday and street-level surges.
- **Point forecasts only** — no uncertainty bands yet.

## 🔭 Roadmap (future work)

- Expand the window to **18–24 months** for a longer seasonal history.
- Add an **hourly** grain to capture intraday surges.
- Drill geography down to **taxi zone** where data density allows.
- Incorporate **exogenous signals** (weather, holidays, events).
- Add **prediction intervals** (uncertainty), not just point estimates.
- Pilot the **India generalisation** on an equivalent open dataset.

## 📄 License & attribution

Analysis code is for portfolio/educational use. Trip data is © the **NYC Taxi & Limousine
Commission**, used under their public
[data terms](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page).
