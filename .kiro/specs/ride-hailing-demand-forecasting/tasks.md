# Implementation Plan: Ride-Hailing Demand Forecasting

## Overview

This plan converts the design into incremental Python coding tasks for a data-science pipeline delivered as a deep Jupyter notebook, a Streamlit dashboard, free-tier GitHub Actions automation, and a live Streamlit Community Cloud deployment backed by a clean GitHub repo.

The ordering honors the brief's explicit next step: **Phase 1 validates the already-downloaded `data/fhvhv_2026-04.parquet` first** (schema, columns, row count, nulls, date range, duplicates, domain violations) and presents real numbers to the user before anything else is built. Every later phase builds on validated inputs.

Pure-logic components (validation, preparation, evaluation, business impact, upload validation) are implemented as deterministic functions and covered by **Hypothesis property-based tests** for the 13 correctness properties in the design. Model fitting and third-party wrappers are covered by example/integration/smoke tests instead.

Working method: phase-by-phase git commits. Long-running jobs (large FHVHV Parquet downloads, heavy model fits, notebook full runs) are **written and verified by Kiro but executed by the user in their own terminal**; those tasks note this explicitly.

## Tasks

- [x] 1. Project scaffolding and scope configuration
  - [x] 1.1 Create repository structure and dependency/hygiene files
    - Create `src/`, `src/models/`, `dashboard/`, `notebook/`, `tests/`, `.github/workflows/`, and git-ignored `data/` directories
    - Write `requirements.txt` (pandas, pyarrow, statsmodels, prophet, xgboost, tensorflow/keras, streamlit, hypothesis, pytest, nbconvert/papermill)
    - Write `.gitignore` excluding `data/`, `.venv/`, secrets, and `PROJECT2_BRIEF_Ride_Hailing_Forecasting.md`
    - _Requirements: 12.3, 12.4_

  - [x] 1.2 Implement ScopeConfig single source of truth
    - Create `src/config.py` with a frozen `ScopeConfig` dataclass (time_grain, geographic_grain, window_start, window_end, candidate_models, holdout_periods, lags, change_log)
    - Set proposed defaults (daily, borough, 2025-05-01 → 2026-04-30, full candidate set, 30-day holdout, lags [1,7,14])
    - Add a window validator that accepts 12-24 months and rejects otherwise, and a `record_scope_change` helper that appends to change_log with rationale
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5_

  - [x] 1.3 Write unit tests for ScopeConfig
    - Test window validator accepts 12-24 months and rejects shorter/longer spans (R2.3)
    - Test scope-change logging records value and rationale (R2.5)
    - _Requirements: 2.3, 2.5_

- [x] 2. Data acquisition and validation (Phase 1 — validate the downloaded file first)
  - [x] 2.1 Implement core profiling functions
    - Create `src/validation.py` with `load_parquet` (raises a clear error naming the path when missing/unreadable), `profile_schema` (row count, column names, dtype per column), `profile_nulls` (count + percentage per column), `pickup_date_range` (min/max pickup timestamp), and `count_duplicates`
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.8_

  - [x] 2.2 Implement domain-violation flagging and report aggregation
    - Add `flag_domain_violations` (pickups outside the file's stated month, negative counts/fares) returning count + example per violation type
    - Add `build_validation_report` aggregating schema, nulls, date range, duplicates, and violations into a `ValidationReport`
    - Add multi-file orchestration that applies profiling to each additional month file when the window is expanded
    - _Requirements: 1.6, 1.7, 1.8_

  - [x] 2.3 Create the raw-data validation runner and present real numbers
    - Add `scripts/validate_raw.py` (and matching notebook cell) that loads `data/fhvhv_2026-04.parquet`, builds the ValidationReport, and prints all findings with real numbers
    - Note: Kiro writes and verifies this code; the USER executes it in their terminal against the real ~1 GB Parquet file and reviews the reported numbers before modeling proceeds
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.8, 13.1_

  - [x] 2.4 Write property test for profiling accuracy
    - **Property 1: Profiling accuracy** — every reported statistic (row count, columns/dtypes, per-column null count/percentage, min/max pickup timestamp) equals ground truth computed directly from the DataFrame
    - **Validates: Requirements 1.2, 1.3, 1.4**

  - [x] 2.5 Write property test for duplicate and domain-violation flagging
    - **Property 2: Duplicate and domain-violation flagging** — with injected duplicates and out-of-domain records, reported duplicate count and flagged-violation count equal the injected counts, and each flagged item is genuinely violating
    - **Validates: Requirements 1.5, 1.6**

  - [x] 2.6 Write unit tests for report aggregation and multi-file orchestration
    - Test `build_validation_report` structure and `load_parquet` error on missing path (R1.8)
    - Test multi-file orchestration applies criteria 2-6 per file (R1.7)
    - _Requirements: 1.7, 1.8_

- [x] 3. Checkpoint - Validation complete
  - Ensure all tests pass and the raw-data validation numbers have been reviewed by the user, ask the user if questions arise.

- [x] 4. Data preparation pipeline
  - [x] 4.1 Implement zone mapping and demand aggregation
    - Create `src/preparation.py` with `map_zones_to_regions` (PULocationID → borough via taxi_zone_lookup) and `aggregate_demand` (count trips per (period, region) at Time_Grain/Geographic_Grain)
    - _Requirements: 4.1_

  - [x] 4.2 Implement zero-fill for missing periods
    - Add `fill_missing_periods` so every (period, region) in the Analysis_Window is present with demand 0 when no records exist, never omitted
    - _Requirements: 4.3_

  - [x] 4.3 Implement documented invalid-record handling
    - Add `apply_validity_rules` that applies a documented handling rule to records failing Requirement 1 checks and returns a `HandlingLog`
    - _Requirements: 4.4_

  - [x] 4.4 Implement lag feature generation
    - Add `add_lag_features` producing `lag_k[t] = demand[t-k]` per region, NaN for the first k periods of each region
    - _Requirements: 4.6_

  - [x] 4.5 Implement the prepare orchestrator with before/after examples
    - Add `prepare` that runs validity handling → zone mapping → aggregation → zero-fill → lag features, and records real before-and-after examples for each transformation
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.6_

  - [x] 4.6 Implement prepared-dataset reconciliation
    - Add `revalidate_prepared` in `src/validation.py` confirming aggregated totals reconcile with raw valid record counts before proceeding
    - _Requirements: 4.5, 13.4_

  - [x] 4.7 Write property test for aggregation correctness and reconciliation
    - **Property 3: Aggregation correctness and reconciliation** — per-bucket counts equal a direct group-by, and total demand equals the number of valid raw records (conservation)
    - **Validates: Requirements 4.1, 4.5, 13.4**

  - [x] 4.8 Write property test for zero-fill completeness
    - **Property 4: Zero-fill completeness** — exactly one row per (period, region) in the window, none omitted, demand exactly 0 where no input records existed
    - **Validates: Requirements 4.3, 13.4**

  - [x] 4.9 Write property test for invalid-record handling
    - **Property 5: Invalid-record handling** — after handling, no injected invalid record remains and the HandlingLog count equals the number injected
    - **Validates: Requirements 4.4**

  - [x] 4.10 Write property test for lag feature correctness
    - **Property 6: Lag feature correctness** — within each region `lag_k` at t equals demand at t−k, and is NaN for the first k periods
    - **Validates: Requirements 4.6**

- [x] 5. Checkpoint - Preparation complete
  - Ensure all tests pass and reconciliation holds, ask the user if questions arise.

- [x] 6. Exploratory data analysis
  - [x] 6.1 Implement demand series visualization
    - Create `src/eda.py` with `plot_demand_series` at the defined Time_Grain over the Analysis_Window, with plain-language interpretation output
    - _Requirements: 3.1, 3.7_

  - [x] 6.2 Implement stationarity and autocorrelation analysis
    - Add `seasonal_decompose_demand` (trend/seasonal/residual), `adf_test` (statistic + p-value), and `acf_pacf` plots, each with interpretation
    - _Requirements: 3.2, 3.3, 3.4, 3.7_

  - [x] 6.3 Implement anomaly detection and correlation analysis
    - Add `detect_anomalies` (identify affected period + describe observation) and `demand_correlations` against candidate explanatory variables, each with interpretation
    - _Requirements: 3.5, 3.6, 3.7_

  - [x] 6.4 Write unit tests for EDA wrappers
    - Test decomposition returns three components of correct length (R3.2), ADF returns statistic and p-value in valid range (R3.3), correlation matrix within [-1, 1] (R3.6)
    - _Requirements: 3.2, 3.3, 3.6_

- [x] 7. Forecasting system
  - [x] 7.1 Implement Forecaster interface and train_all orchestration
    - Create `src/models/base.py` with the `Forecaster` protocol (fit/predict), a `TrainedModel`/`ExclusionRecord`/`Forecast` structures, and `train_all` that trains every candidate, catches per-model failures, and records an ExclusionRecord with reason so one failure never aborts others
    - _Requirements: 5.7, 5.8_

  - [x] 7.2 Implement Holt-Winters baseline
    - Create `src/models/holt_winters.py` implementing the Forecaster interface for Exponential Smoothing
    - Note: fitting may be run by the USER for the full series; Kiro verifies on a small fixture
    - _Requirements: 5.1_

  - [x] 7.3 Implement SARIMA/SARIMAX
    - Create `src/models/sarima.py` with a SARIMA model and a SARIMAX exogenous variant
    - _Requirements: 5.2_

  - [x] 7.4 Implement VAR/VARMAX multivariate model
    - Create `src/models/var.py` for joint multi-region forecasting
    - _Requirements: 5.3_

  - [x] 7.5 Implement Prophet model
    - Create `src/models/prophet_model.py` implementing the Forecaster interface
    - _Requirements: 5.4_

  - [x] 7.6 Implement XGBoost with lag features
    - Create `src/models/xgboost_lags.py` consuming the lag features from preparation
    - _Requirements: 5.5_

  - [x] 7.7 Implement LSTM/GRU deep-learning model
    - Create `src/models/lstm_gru.py` with an LSTM model and a GRU variant
    - Note: training is heavy and run by the USER; Kiro verifies wiring on a small fixture
    - _Requirements: 5.6_

  - [x] 7.8 Write integration tests for each model family
    - Fit each family on a small fixture series and assert a forecast of expected length aligned to the holdout index (R5.7); assert exclusion recorded on a forced failure (R5.8)
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8_

- [x] 8. Checkpoint - Models train and forecast
  - Ensure all model integration tests pass on fixtures; note which full fits the user must run, ask the user if questions arise.

- [x] 9. Evaluation framework
  - [x] 9.1 Implement holdout split
    - Create `src/evaluation.py` with `split_holdout` reserving the most-recent contiguous n periods, disjoint from training, exactly reconstructing the series on concatenation
    - _Requirements: 6.1_

  - [x] 9.2 Implement error metrics
    - Add `error_metrics` computing MAE, RMSE, MAPE (all ≥ 0), with a documented MAPE-zero convention
    - _Requirements: 6.2_

  - [x] 9.3 Implement comparison table and forecast-vs-actual plots
    - Add `comparison_table` (one row per model incl. underperformers/excluded, same metric columns) and a plot of each model's forecast against holdout actuals
    - _Requirements: 6.3, 6.4_

  - [x] 9.4 Implement error-by-period reporting
    - Add `error_by_period` reporting non-negative error across distinct period buckets that partition the holdout, not only aggregate error
    - _Requirements: 6.5_

  - [x] 9.5 Implement carry-forward model selection
    - Add `select_carry_forward` returning 3-5 model names present in the table, with documented justification based on reported metrics
    - _Requirements: 6.6_

  - [x] 9.6 Write property test for holdout split
    - **Property 7: Holdout split has no leakage** — holdout equals most-recent n periods, train is the earlier remainder, disjoint, concatenation reconstructs the original exactly
    - **Validates: Requirements 6.1**

  - [x] 9.7 Write property test for error metrics
    - **Property 8: Error-metric correctness** — MAE, RMSE, MAPE all non-negative; all exactly 0 when forecast equals actual; RMSE ≥ MAE
    - **Validates: Requirements 6.2**

  - [x] 9.8 Write property test for comparison table completeness
    - **Property 9: Comparison table completeness** — exactly one row per model with the same metric column set for each (incl. excluded)
    - **Validates: Requirements 6.3**

  - [x] 9.9 Write property test for error-by-period partition
    - **Property 10: Error-by-period partition** — each per-bucket error non-negative and buckets partition the holdout periods (each period in exactly one bucket)
    - **Validates: Requirements 6.5**

  - [x] 9.10 Write property test for carry-forward selection count
    - **Property 11: Carry-forward selection count** — for a table with ≥3 models, returns 3-5 names all present in the table
    - **Validates: Requirements 6.6**

- [x] 10. Business impact recommendation
  - [x] 10.1 Implement business module
    - Create `src/business.py` with `positioning_recommendation` (driver positioning at Time_Grain/Geographic_Grain), `quantify_impact` (reduced rider wait / driver idle time with shown assumptions and calculation), and `india_generalization` (Ola/Uber/Rapido narrative)
    - _Requirements: 7.1, 7.2, 7.3, 7.4_

  - [x] 10.2 Write property test for impact calculation reproducibility
    - **Property 12: Impact calculation reproducibility** — quantified benefit equals recomputing the documented formula from the same assumptions
    - **Validates: Requirements 7.3**

  - [x] 10.3 Write unit test for recommendation generation
    - Assert a recommendation is produced for a sample forecast at the defined grain (R7.1)
    - _Requirements: 7.1_

- [x] 11. Checkpoint - Core pipeline complete
  - Ensure all property and unit tests pass end to end, ask the user if questions arise.

- [x] 12. Jupyter notebook deliverable
  - [x] 12.1 Build the guided-story notebook
    - Create `notebook/demand_forecasting.ipynb` presenting the full analysis top-to-bottom with structured markdown headers, reusing `src/` functions; each code cell has an explanation above and interpretation below, new tools explained on first use, and validation cells confirming each transformation produced the intended result
    - Note: the full end-to-end run over real data is executed by the USER; Kiro verifies structure and light cells
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 13.1, 13.2, 13.3_

  - [x] 12.2 Write notebook execution smoke test
    - Add a CI test running the notebook top-to-bottom via `nbconvert`/`papermill` on a small fixture to confirm it runs without errors from a clean state (R8.6)
    - _Requirements: 8.6_

- [x] 13. Interactive dashboard deliverable
  - [x] 13.1 Build the Streamlit storytelling dashboard
    - Create `dashboard/app.py` with sidebar navigation across all required sections (business problem, data source + honest limitations, EDA findings, data preparation, tools/technology, models/method, results, business insights, India generalization) and model-comparison + forecast visualizations, reusing `src/` functions
    - _Requirements: 9.1, 9.2, 9.3_

  - [x] 13.2 Implement upload-and-analyze mode with input validation
    - Add `validate_upload` returning a descriptive error naming the missing/invalid column for non-conforming input, and passing for conforming input; wire the upload-and-analyze flow to display results
    - _Requirements: 9.4, 9.5_

  - [x] 13.3 Write property test for upload input validation
    - **Property 13: Upload input validation** — missing required column or corrupted dtype yields an error referencing the offending column; a fully conforming dataset passes
    - **Validates: Requirements 9.5**

  - [x] 13.4 Write dashboard smoke test
    - Assert the app imports and its sections render, and upload of a valid fixture returns results (R9.1-9.4)
    - _Requirements: 9.1, 9.2, 9.3, 9.4_

- [x] 14. Automation
  - [x] 14.1 Implement scheduled auto-refresh workflow
    - Create `.github/workflows/refresh.yml` (free-tier GitHub Actions) that re-runs the prepare→forecast pipeline on a cron schedule and records success/failure in run logs
    - _Requirements: 10.1, 10.2, 10.3, 10.4_

  - [x] 14.2 Write forced-failure automation test
    - Assert a forced-failure run records the failure in logs (R10.4)
    - _Requirements: 10.4_

- [x] 15. Deployment and repository finalization
  - [x] 15.1 Prepare deployment and README
    - Add Streamlit Community Cloud deployment config so the dashboard serves at a public URL without visitor local setup, and write `README.md` describing the project, NYC TLC data source, results, and roadmap
    - Note: the actual public deploy is performed by the USER via their Streamlit Community Cloud account; Kiro provides config and instructions
    - _Requirements: 11.1, 11.2, 11.3, 12.1_

  - [x] 15.2 Write repository hygiene checks
    - Assert README contains data source/results/roadmap (R12.1), and `.gitignore` excludes `data/`, virtual envs, secrets, and the brief file (R12.3, R12.4)
    - _Requirements: 12.1, 12.3, 12.4_

- [x] 16. Final checkpoint - Ensure all tests pass
  - Ensure all property, unit, integration, and smoke tests pass; confirm phase-by-phase commits are in place (R12.2) and limitations are documented (R13.2, R13.3), ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional test tasks and can be skipped for a faster MVP, but each maps to a design property or acceptance criterion for traceability.
- **Phase 1 (task 2) runs first**: the already-downloaded `fhvhv_2026-04.parquet` is validated with real numbers before any modeling, per the brief's explicit next step.
- **Long-running jobs** (large FHVHV downloads, heavy model fits, full notebook/deploy runs) are written and verified by Kiro but executed by the USER in their own terminal; those tasks note this.
- Property tests use **Hypothesis** at a minimum of 100 iterations each; the 13 properties map one-to-one to design properties and cover the pure-logic layers.
- Model fitting, third-party statistical wrappers, notebook, dashboard, automation, and deployment are covered by example/integration/smoke tests rather than property tests.
- Requirement coverage: R1→task 2, R2→1.2, R3→6, R4→4, R5→7, R6→9, R7→10, R8→12, R9→13, R10→14, R11→15.1, R12→1.1/15/16, R13 threaded through 2.3/4.6/12.1/16.
- Checkpoints (tasks 3, 5, 8, 11, 16) ensure incremental validation and align with phase-by-phase commits.

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1"] },
    { "id": 1, "tasks": ["1.2"] },
    { "id": 2, "tasks": ["1.3", "2.1"] },
    { "id": 3, "tasks": ["2.2"] },
    { "id": 4, "tasks": ["2.3", "2.4"] },
    { "id": 5, "tasks": ["2.5", "4.1"] },
    { "id": 6, "tasks": ["2.6", "4.2"] },
    { "id": 7, "tasks": ["4.3"] },
    { "id": 8, "tasks": ["4.4"] },
    { "id": 9, "tasks": ["4.5"] },
    { "id": 10, "tasks": ["4.6", "4.7", "6.1"] },
    { "id": 11, "tasks": ["4.8", "6.2"] },
    { "id": 12, "tasks": ["4.9", "6.3"] },
    { "id": 13, "tasks": ["4.10", "6.4", "7.1"] },
    { "id": 14, "tasks": ["7.2", "7.3", "7.4", "7.5", "7.6", "7.7", "9.1"] },
    { "id": 15, "tasks": ["7.8", "9.2"] },
    { "id": 16, "tasks": ["9.3"] },
    { "id": 17, "tasks": ["9.4"] },
    { "id": 18, "tasks": ["9.5"] },
    { "id": 19, "tasks": ["9.6", "10.1"] },
    { "id": 20, "tasks": ["9.7", "10.2"] },
    { "id": 21, "tasks": ["9.8", "10.3"] },
    { "id": 22, "tasks": ["9.9"] },
    { "id": 23, "tasks": ["9.10"] },
    { "id": 24, "tasks": ["12.1", "13.1"] },
    { "id": 25, "tasks": ["12.2", "13.2"] },
    { "id": 26, "tasks": ["13.3", "14.1", "15.1"] },
    { "id": 27, "tasks": ["13.4", "14.2", "15.2"] }
  ]
}
```
