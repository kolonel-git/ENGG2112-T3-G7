# Stage 5: choose one training experiment

Prerequisite: processed `Data_Parquets_10min` files. [Full run guide](../../README.md).

## 1. Choose the alarm target

| Group | Target codes | Horizons |
|---|---|---|
| [Icing](icing_8230_8234_8235_8236/README.md) | 8230, 8234, 8235, 8236 | 10, 20, 30 minutes; 1 hour; 6 hours; 1 day; 1 week |
| [Pitch lubrication](pitch_lubrication_3130/README.md) | 3130 | 10, 20, 30 minutes; 1 hour; 6 hours; 1 day; 1 week |
| [Untwisting cables](untwisting_cables_10105/README.md) | 10105 | 10, 20, 30 minutes; 1 hour; 6 hours; 1 day; 1 week |
| [All recorded alarms](all_alarms/README.md) | Every code, including operational codes 20 and 25 | 10, 20, 30 minutes |

## 2. Choose the feature mode

- `01_train_all_features.ipynb`: recommended first run; every available sensor feature.
- `02_train_selected_features.ipynb`: edit `SELECTED_FEATURES` to choose sensors.
- `03_train_selected_features_variant.ipynb`: an independent alternative selected-sensor configuration. It is not a required next stage and may duplicate version 02 if settings match.

## 3. Choose one horizon

- `01_horizon_10min/`: next 10 minutes.
- `02_horizon_20min/`: next 20 minutes.
- `03_horizon_30min/`: next 30 minutes.
- `04_horizon_1hour/`: next 1 hour.
- `05_horizon_6hours/`: next 6 hours.
- `06_horizon_1day/`: next 1 day.
- `07_horizon_1week/`: next 1 week.

## Shared defaults

Five turbines: 2304510–2304514. Years: train 2016–2018, validate 2020, test 2021. Inputs: previous two hours of sensor readings, summarized independently per turbine. Predictions every ten minutes. Target: at least one new target alarm start in `[t, t + horizon)`; currently active target alarms are excluded.

Models: constant prevalence, fixed random-weight score, lightweight logistic regression (`lbfgs`, 100 iterations, tolerance 1e-3), random forest, LightGBM and XGBoost. Training-only imputation/scaling; validation-only threshold selection; fixed thresholds on test. Both classes must exist in each split.

Restart Kernel → Run All after changing settings. No other horizon or feature variant is a prerequisite. Review printed timing and output paths. Results go to compact `reports/<target>_<minutes>m_<mode>_<hash>/` folders; `configuration.json` retains exact settings. The hash separates configurations; it does not automatically archive reruns with identical settings.

Metrics include precision, recall, F1/F2, AP, ROC-AUC and confusion matrices. Detailed binary summaries and ROC plots are also available in the 10/20/30-minute all-features notebooks. Prediction-history plots show one configured turbine at a time although fitting pools all five.

The evaluated years have been inspected in previous work; these are exploratory results. AP must be compared with prevalence. Different alarm definitions and horizons change the task.

## Advanced alternative

[All-turbine temporal CV](../06_all_turbine_cross_validation/README.md) is separate and retains its own candidates and date ranges; Stage 5 is not a prerequisite.
