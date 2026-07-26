# Requirements Document

## Introduction

This document defines the requirements for **Ride-Hailing Demand Forecasting**, a portfolio data science project that forecasts ride-hailing demand (how much, when, and where) so that platforms can position drivers ahead of need, reducing rider wait time and driver idle time. The project is built on real, official **NYC TLC (Taxi & Limousine Commission) Trip Record Data** — specifically the For-Hire Vehicle High Volume (FHVHV = Uber/Lyft-style) feed — and framed as generalizing to Ola/Uber/Rapido India.

The project spans the full lifecycle: data acquisition and validation, exploratory data analysis (EDA), data preparation, comparison of a broad set of forecasting models, honest evaluation, translation of the best forecast into a business recommendation, and delivery through a deep Jupyter notebook, an interactive Streamlit dashboard, automation, live public deployment, and clean GitHub repository hygiene.

The overriding constraint (the "golden rule") is that only real public data is used, nothing is fabricated, and every result is validated and defensible at every step.

## Glossary

- **Project**: The overall ride-hailing demand forecasting deliverable, encompassing the notebook, dashboard, automation, and deployment.
- **Data_Validator**: The process/component responsible for profiling and validating raw and transformed data (schema, completeness, validity, quality).
- **EDA_Module**: The component that performs exploratory data analysis, including trend, seasonality, anomaly, stationarity, and autocorrelation analysis.
- **Data_Preparation_Pipeline**: The component that cleans, aggregates, and reshapes raw trip records into the forecasting dataset at the chosen time and geographic grain.
- **Forecasting_System**: The component that trains and produces forecasts from the candidate model set.
- **Evaluation_Framework**: The component that scores and compares model forecasts using defined metrics on held-out data.
- **Business_Module**: The component that translates the selected forecast into a driver-positioning / cost-savings recommendation.
- **Notebook**: The deep, self-contained, aesthetic Jupyter notebook deliverable.
- **Dashboard**: The interactive Streamlit storytelling dashboard deliverable.
- **Automation_Pipeline**: The upload-and-analyze and/or scheduled auto-refresh mechanism (free-tier, e.g. GitHub Actions).
- **Deployment**: The live public hosting of the Dashboard (e.g. Streamlit Community Cloud).
- **Repository**: The public GitHub repository containing the project code and documentation.
- **FHVHV**: For-Hire Vehicle High Volume trip records (Uber/Lyft-style rides) from the NYC TLC feed.
- **NYC TLC Data**: The official NYC Taxi & Limousine Commission Trip Record Data, published in Parquet format.
- **Time_Grain**: The temporal aggregation level of the forecasting series (e.g. hourly or daily).
- **Geographic_Grain**: The spatial aggregation level of the forecasting series (e.g. borough or taxi zone).
- **Analysis_Window**: The contiguous span of historical months of data selected for the forecasting work.
- **Candidate_Model_Set**: The full set of forecasting approaches trained and reported (baseline, classical, multivariate, modern, ML, deep learning).
- **ADF_Test**: Augmented Dickey-Fuller test for stationarity.
- **ACF/PACF**: Autocorrelation Function / Partial Autocorrelation Function analysis.
- **Holdout_Set**: The most recent contiguous portion of the series reserved for out-of-sample evaluation (never used for training).
- **Forecast_Error_Metric**: A quantitative accuracy measure such as MAE, RMSE, or MAPE.

## Requirements

### Requirement 1: Data Acquisition and Validation

**User Story:** As a data scientist, I want the downloaded NYC TLC data validated before any modeling, so that every downstream result rests on trustworthy, understood raw data.

#### Acceptance Criteria

1. THE Data_Validator SHALL use only official NYC TLC Trip Record Data obtained from the NYC TLC source in Parquet format.
2. WHEN the `fhvhv_2026-04.parquet` file is loaded, THE Data_Validator SHALL report the row count, column names, and data type of each column.
3. WHEN the `fhvhv_2026-04.parquet` file is profiled, THE Data_Validator SHALL report the count and percentage of null values for each column.
4. WHEN the `fhvhv_2026-04.parquet` file is profiled, THE Data_Validator SHALL report the minimum and maximum pickup timestamp to establish the date range.
5. THE Data_Validator SHALL report the count of duplicate records in the loaded data.
6. IF a column contains values outside its valid domain, such as a pickup timestamp outside the file's stated month or a negative trip count, THEN THE Data_Validator SHALL flag the affected records with a count and an example.
7. WHERE additional months are required to satisfy the selected Analysis_Window, THE Data_Validator SHALL apply Acceptance Criteria 2 through 6 to each additional month file.
8. THE Data_Validator SHALL present all validation findings to the user with real numbers before any modeling work begins.

### Requirement 2: Scope Definition

**User Story:** As a data scientist, I want the forecasting scope explicitly fixed, so that the time grain, geographic grain, and data window are unambiguous throughout the project.

#### Acceptance Criteria

1. THE Project SHALL define the Time_Grain as a single documented value used consistently across EDA, preparation, modeling, and evaluation.
2. THE Project SHALL define the Geographic_Grain as a single documented value used consistently across EDA, preparation, modeling, and evaluation.
3. THE Project SHALL define the Analysis_Window as a contiguous span between 12 and 24 months of NYC TLC Data.
4. THE Project SHALL document the Candidate_Model_Set that will be trained and reported.
5. WHERE a scope value is changed after initial definition, THE Project SHALL record the change and its rationale in the project documentation.

### Requirement 3: Exploratory Data Analysis

**User Story:** As a data scientist, I want the demand series explored statistically and visually, so that trends, seasonality, and anomalies are understood before modeling.

#### Acceptance Criteria

1. WHEN EDA is performed, THE EDA_Module SHALL produce a time-series plot of demand at the defined Time_Grain over the Analysis_Window.
2. THE EDA_Module SHALL perform a seasonal decomposition of the demand series into trend, seasonal, and residual components.
3. THE EDA_Module SHALL assess stationarity of the demand series using the ADF_Test and report the test statistic and p-value.
4. THE EDA_Module SHALL produce ACF/PACF plots of the demand series to inform model order selection.
5. WHEN an anomaly such as a demand spike or drop is detected in the series, THE EDA_Module SHALL identify the affected time period and describe the observation.
6. THE EDA_Module SHALL report correlation between demand and any available candidate explanatory variables.
7. THE EDA_Module SHALL accompany each chart and statistical output with a plain-language interpretation of what it shows.

### Requirement 4: Data Preparation

**User Story:** As a data scientist, I want raw trip records cleaned and reshaped into a validated forecasting dataset, so that models are trained on correct inputs at the chosen grain.

#### Acceptance Criteria

1. THE Data_Preparation_Pipeline SHALL aggregate raw trip records into a demand series at the defined Time_Grain and Geographic_Grain.
2. WHEN a data transformation is applied, THE Data_Preparation_Pipeline SHALL present a real before-and-after example of the affected data.
3. IF a time period within the Analysis_Window has no trip records, THEN THE Data_Preparation_Pipeline SHALL represent that period explicitly as zero demand rather than omitting it.
4. WHEN records fail a validity check defined in Requirement 1, THE Data_Preparation_Pipeline SHALL apply a documented handling rule to those records.
5. WHEN preparation is complete, THE Data_Validator SHALL re-validate the prepared dataset and confirm that aggregated totals reconcile with the raw record counts.
6. THE Data_Preparation_Pipeline SHALL produce lag features required by the machine-learning models in the Candidate_Model_Set.

### Requirement 5: Model Comparison

**User Story:** As a data scientist, I want a broad set of forecasting models trained and reported, so that the model choice is justified by honest evidence rather than assumption.

#### Acceptance Criteria

1. THE Forecasting_System SHALL train a simple baseline model using Exponential Smoothing (Holt-Winters).
2. THE Forecasting_System SHALL train at least one classical univariate model from the ARIMA/SARIMA family.
3. THE Forecasting_System SHALL train a multivariate model from the VAR/VARMAX family for joint multi-region forecasting.
4. THE Forecasting_System SHALL train a Prophet model.
5. THE Forecasting_System SHALL train a machine-learning model using XGBoost or LightGBM with lag features.
6. THE Forecasting_System SHALL train a deep-learning model using LSTM.
7. THE Forecasting_System SHALL produce forecasts from every trained model over the same Holdout_Set.
8. WHERE a candidate model cannot be trained on the prepared data, THE Forecasting_System SHALL record the reason the model was excluded.

### Requirement 6: Model Evaluation

**User Story:** As a data scientist, I want all models scored honestly on held-out data, so that results are comparable and defensible.

#### Acceptance Criteria

1. THE Evaluation_Framework SHALL reserve the most recent contiguous portion of the demand series as the Holdout_Set and exclude it from all model training.
2. THE Evaluation_Framework SHALL compute the same set of Forecast_Error_Metrics for every model in the Candidate_Model_Set on the Holdout_Set.
3. THE Evaluation_Framework SHALL present a comparison table containing the Forecast_Error_Metrics for every trained model, including underperforming models.
4. THE Evaluation_Framework SHALL plot each model's forecast against the actual Holdout_Set values.
5. THE Evaluation_Framework SHALL report whether forecast error varies across distinct time periods rather than reporting only an aggregate error.
6. WHEN model comparison is complete, THE Evaluation_Framework SHALL identify between three and five models to carry forward for deep explanation, with a documented justification based on the reported metrics.

### Requirement 7: Business Impact Recommendation

**User Story:** As a business stakeholder, I want the best forecast translated into a driver-positioning recommendation, so that the analysis produces actionable value.

#### Acceptance Criteria

1. THE Business_Module SHALL derive a driver-positioning recommendation from the selected forecast at the defined Time_Grain and Geographic_Grain.
2. THE Business_Module SHALL express the expected impact in business terms such as reduced rider wait time or reduced driver idle time.
3. WHERE a quantified benefit is stated, THE Business_Module SHALL show the assumptions and calculation used to derive it.
4. THE Business_Module SHALL describe how the NYC-based approach generalizes to Ola, Uber, or Rapido operations in India.

### Requirement 8: Jupyter Notebook Deliverable

**User Story:** As a reviewer, I want a deep, self-contained notebook, so that a zero-context reader can understand the full analysis top to bottom.

#### Acceptance Criteria

1. THE Notebook SHALL present the analysis as a top-to-bottom guided story using structured markdown headers.
2. WHERE a code cell performs an action, THE Notebook SHALL include an explanation above the cell describing what the code does, the meaning of key terms, and the effect of key parameters.
3. WHEN a cell produces an output such as a number, table, or chart, THE Notebook SHALL include an interpretation of what the output means.
4. WHEN a new tool, library, or model is introduced, THE Notebook SHALL explain in plain language what it is and why it is used.
5. WHEN a data transformation is shown, THE Notebook SHALL include a validation cell confirming the transformation produced the intended result.
6. THE Notebook SHALL run end to end without errors from a clean state.

### Requirement 9: Interactive Dashboard Deliverable

**User Story:** As a reviewer, I want an interactive storytelling dashboard, so that I can explore the project narrative without reading code.

#### Acceptance Criteria

1. THE Dashboard SHALL provide sidebar navigation between distinct sections of the project story.
2. THE Dashboard SHALL include sections covering the business problem, the data source and its honest limitations, the EDA findings, the data preparation, the tools and technology, the models and method, the results, the business insights, and the generalization to India.
3. THE Dashboard SHALL display the model comparison results and forecast visualizations.
4. WHERE an upload-your-data mode is provided, THE Dashboard SHALL analyze user-supplied data conforming to the expected input format and display results.
5. IF user-supplied data does not conform to the expected input format, THEN THE Dashboard SHALL display a descriptive error message identifying the problem.

### Requirement 10: Automation

**User Story:** As a maintainer, I want the project to support automated analysis or refresh, so that it demonstrates operational maturity using free-tier tooling.

#### Acceptance Criteria

1. THE Automation_Pipeline SHALL provide an upload-and-analyze capability, a scheduled auto-refresh capability, or both.
2. THE Automation_Pipeline SHALL operate using free-tier services only.
3. WHERE a scheduled auto-refresh is provided, THE Automation_Pipeline SHALL run on a defined schedule using GitHub Actions.
4. IF an automated run fails, THEN THE Automation_Pipeline SHALL record the failure in its run logs.

### Requirement 11: Deployment

**User Story:** As a reviewer, I want the dashboard available at a live public link, so that the project is accessible beyond a local machine.

#### Acceptance Criteria

1. THE Deployment SHALL host the Dashboard at a publicly accessible URL.
2. THE Deployment SHALL use a free-tier hosting service such as Streamlit Community Cloud.
3. WHEN a visitor opens the public URL, THE Deployment SHALL serve the Dashboard without requiring local setup by the visitor.

### Requirement 12: GitHub Repository Hygiene

**User Story:** As a reviewer, I want a clean, well-documented repository, so that the project history and quality are immediately evident.

#### Acceptance Criteria

1. THE Repository SHALL include a README describing the project, its data source, its results, and its roadmap.
2. THE Repository SHALL record project progress through phase-by-phase commits with descriptive commit messages.
3. THE Repository SHALL exclude large data files, virtual environments, secrets, and personal planning files by means of a `.gitignore` file.
4. THE Repository SHALL exclude the project brief file from version control.

### Requirement 13: Data Integrity and Validation Discipline

**User Story:** As a data scientist, I want validation and honesty enforced throughout, so that every result is real and defensible.

#### Acceptance Criteria

1. THE Project SHALL use only real public NYC TLC Data and SHALL derive every reported result from that data.
2. WHEN any result appears inconsistent with expectation, THE Project SHALL investigate the cause and document the finding rather than omit the result.
3. WHERE a limitation cannot be resolved, THE Project SHALL document the limitation explicitly.
4. WHEN a data transformation output is produced, THE Data_Validator SHALL re-validate the output against the intended result before the Project proceeds to the next step.
