# untwisting_cables_10105

Each horizon folder contains the same three notebook roles:

- `01_train_all_features.ipynb`: all sensor features.
- `02_train_selected_features.ipynb`: editable sensor subset.
- `03_train_selected_features_variant.ipynb`: preserved alternate copy; identical settings give the same experiment/report directory.

Horizon folders: `01_horizon_10min`, `04_horizon_1hour`, `05_horizon_6hours`, `06_horizon_1day`, `07_horizon_1week`.

All use the previous hour of sensor inputs and predict every 10 minutes. Each notebook can run independently after preprocessing. Edit settings and Run All to train. Existing notebooks were preserved; newly created notebooks have not been trained.


## Prediction horizon order

- `01_horizon_10min/`
- `02_horizon_20min/`
- `03_horizon_30min/`
- `04_horizon_1hour/`
- `05_horizon_6hours/`
- `06_horizon_1day/`
- `07_horizon_1week/`

Each horizon contains all-features, selected-features, and selected-features variant notebooks. The sensor lookback remains two hours. New horizons require independent training.


Training defaults pool turbines **2304510, 2304511, 2304512, 2304513 and 2304514**. Histories are calculated separately per turbine; models are fitted on combined training rows. Lightweight logistic regression uses `lbfgs`, `max_iter=100`, and `tol=1e-3`, with training-only imputation and scaling. Reports include the turbine IDs in their folder name to preserve previous single-turbine results. Prediction-history plots still show one configured turbine at a time.


## Five-year training defaults

Only 2016, 2017, 2018, 2020 and 2021 are loaded. Train: 2016-2018; validation/threshold selection: 2020; test: 2021. All five turbines and the two-hour lookback are retained. Report folders include the selected years so previous experiments are preserved. This applies to `05_model_training`; the separate all-turbine cross-validation workflow is unchanged.


Report folders now use compact names such as `all_alarms_30m_all_<hash>`. The digest distinguishes targets, turbine IDs, selected years, horizons, split dates, seeds and feature selections. Exact settings remain in `configuration.json`; existing renamed reports are listed in `reports/report_folder_renames.json`.
