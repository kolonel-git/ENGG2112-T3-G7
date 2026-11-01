# Untwisting cables

Target: **10105**. [Training guide](../README.md). Choose one notebook below; horizons are independent.

| Horizon | Start with all features | Other versions |
|---|---|---|
| 10 minutes | [Run notebook](01_horizon_10min/01_train_all_features.ipynb) | [Selected](01_horizon_10min/02_train_selected_features.ipynb), [variant](01_horizon_10min/03_train_selected_features_variant.ipynb) |
| 20 minutes | [Run notebook](02_horizon_20min/01_train_all_features.ipynb) | [Selected](02_horizon_20min/02_train_selected_features.ipynb), [variant](02_horizon_20min/03_train_selected_features_variant.ipynb) |
| 30 minutes | [Run notebook](03_horizon_30min/01_train_all_features.ipynb) | [Selected](03_horizon_30min/02_train_selected_features.ipynb), [variant](03_horizon_30min/03_train_selected_features_variant.ipynb) |
| 1 hour | [Run notebook](04_horizon_1hour/01_train_all_features.ipynb) | [Selected](04_horizon_1hour/02_train_selected_features.ipynb), [variant](04_horizon_1hour/03_train_selected_features_variant.ipynb) |
| 6 hours | [Run notebook](05_horizon_6hours/01_train_all_features.ipynb) | [Selected](05_horizon_6hours/02_train_selected_features.ipynb), [variant](05_horizon_6hours/03_train_selected_features_variant.ipynb) |
| 1 day | [Run notebook](06_horizon_1day/01_train_all_features.ipynb) | [Selected](06_horizon_1day/02_train_selected_features.ipynb), [variant](06_horizon_1day/03_train_selected_features_variant.ipynb) |
| 1 week | [Run notebook](07_horizon_1week/01_train_all_features.ipynb) | [Selected](07_horizon_1week/02_train_selected_features.ipynb), [variant](07_horizon_1week/03_train_selected_features_variant.ipynb) |

## Shared defaults

Five turbines: 2304510–2304514. Years: train 2016–2018, validate 2020, test 2021. Inputs: previous two hours of sensor readings, summarized independently per turbine. Predictions every ten minutes. Target: at least one new target alarm start in `[t, t + horizon)`; currently active target alarms are excluded.

Models: constant prevalence, fixed random-weight score, lightweight logistic regression (`lbfgs`, 100 iterations, tolerance 1e-3), random forest, LightGBM and XGBoost. Training-only imputation/scaling; validation-only threshold selection; fixed thresholds on test. Both classes must exist in each split.

Restart Kernel → Run All after changing settings. No other horizon or feature variant is a prerequisite. Review printed timing and output paths. Results go to compact `reports/<target>_<minutes>m_<mode>_<hash>/` folders; `configuration.json` retains exact settings. The hash separates configurations; it does not automatically archive reruns with identical settings.

Metrics include precision, recall, F1/F2, AP, ROC-AUC and confusion matrices. Detailed binary summaries and ROC plots are also available in the 10/20/30-minute all-features notebooks. Prediction-history plots show one configured turbine at a time although fitting pools all five.

The evaluated years have been inspected in previous work; these are exploratory results. AP must be compared with prevalence. Different alarm definitions and horizons change the task.
