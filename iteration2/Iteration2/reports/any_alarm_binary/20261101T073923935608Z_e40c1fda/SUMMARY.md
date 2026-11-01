# Two-hour history: will any alarm start in the next 10, 20 or 30 minutes?

Completed a first sampled-window experiment on all 21 physical turbines using the team's already preprocessed ten-minute data. Logistic regression, random forest and XGBoost were trained independently for each horizon, with a constant-prevalence baseline. Sensor summaries from exactly the preceding 12 rows are the only predictors. Alarm inputs, current/future sensor readings and future-fitted transformations are excluded.

## Final test results for validation-selected models

Model choices were frozen using validation average precision. Each model's yes/no threshold was selected using validation F1. The table below uses 2024–2025 windows with those unchanged choices and thresholds.

| horizon_minutes | model | precision | recall | f1 | average_precision | roc_auc |
| --- | --- | --- | --- | --- | --- | --- |
| 10 | xgboost | 0.357 | 0.574 | 0.440 | 0.384 | 0.827 |
| 20 | xgboost | 0.439 | 0.625 | 0.516 | 0.487 | 0.816 |
| 30 | xgboost | 0.499 | 0.640 | 0.561 | 0.550 | 0.809 |

Precision is the share of predicted-yes windows that really contain a new alarm start. Recall is the share of actual-yes windows flagged. These are window metrics, not independent-event detection rates. False positives and missed alarms remain material; this is not a deployment-ready system.

## Validation comparison

| horizon_minutes | model | precision | recall | f1 | average_precision |
| --- | --- | --- | --- | --- | --- |
| 10 | prevalence_baseline | 0.000 | 0.000 | 0.000 | 0.110 |
| 10 | logistic_regression | 0.286 | 0.529 | 0.371 | 0.290 |
| 10 | random_forest | 0.326 | 0.629 | 0.430 | 0.380 |
| 10 | xgboost | 0.354 | 0.572 | 0.438 | 0.380 |
| 20 | prevalence_baseline | 0.000 | 0.000 | 0.000 | 0.167 |
| 20 | logistic_regression | 0.366 | 0.603 | 0.455 | 0.390 |
| 20 | random_forest | 0.432 | 0.626 | 0.511 | 0.484 |
| 20 | xgboost | 0.431 | 0.632 | 0.512 | 0.491 |
| 30 | prevalence_baseline | 0.000 | 0.000 | 0.000 | 0.211 |
| 30 | logistic_regression | 0.416 | 0.624 | 0.500 | 0.451 |
| 30 | random_forest | 0.482 | 0.643 | 0.551 | 0.545 |
| 30 | xgboost | 0.487 | 0.651 | 0.557 | 0.551 |

## Population and sample sizes

| Split | Eligible windows | Sampled windows |
| --- | --- | --- |
| train | 6,501,772 | 120,000 |
| validation | 1,096,358 | 50,000 |
| test | 2,156,436 | 50,000 |

Training: 2016–2022 (2019 absent). Validation: 2023. Test: 2024–2025. Each split uses a reproducible uniform sample independent of labels, preserving the natural class mix in expectation. The full development audit covers 147 turbine-years and 7,732,368 grid rows. All 120 minutes of input history and the maximum 30-minute target must be inside their split. No feature cache or modified source data is written.

Two training sensors (`wtc_InvUpTmp_mean`, `wtc_TrafoTMa_mean`) are missing in about 88% of observed training rows. Training-only median imputation handles their absent values. Source alarm-label completeness is inherited from the supplied Parquets; this experiment cannot reconstruct omitted raw alarm events. Earlier team experiments have inspected these years, so the chronological test is not historically untouched. Scores are not calibrated probabilities. Overlapping windows and shared turbines mean these results do not establish independent-event or unseen-turbine generalisation.

## Figures

- [Source coverage and missingness](01_coverage_missingness.png)
- [Alarm-start prevalence by horizon and year](02_alarm_prevalence.png)
- [Sensors versus any new alarm starts](03_sensors_and_alarm_starts.png)
- [Validation precision–recall](04_validation_precision_recall.png)
- [Test precision–recall](04_test_precision_recall.png)
- [Test confusion matrices](05_test_confusion_matrices.png)
- [Continuous validation forecast example](06_validation_forecast_timeline.png)
- [Selected-model feature importance](07_selected_feature_importance.png)

## Reuse

Open `notebooks/07_any_alarm_binary/02_review_saved_results.ipynb` to review this run without retraining. Use `03_predict_with_saved_models.ipynb` for a historical yes/no forecast example. Use notebook 01 to create a new experiment after finalising development settings; do not tune against these saved test results. The config, frozen selections, packages, sampled prediction tables and model bundles are saved beside this report.

All seven new tests passed, including exact label boundaries, exclusion of future values, gap handling, split purging, deterministic bounded sampling, and inference without alarm/future data. Every code cell in the review and inference notebooks also executed successfully. `example_yes_no_forecast.csv` contains the actual three-horizon example outputs.
