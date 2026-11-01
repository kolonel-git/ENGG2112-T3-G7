# 05_model_training

Choose an alarm-set folder and horizon. Current configurations all use the icing alarm group. Each model notebook is independent.

See [the pipeline guide](../README.md) for the full run order.


## Alarm-specific training folders

- `icing_8230_8234_8235_8236/` — existing grouped icing notebooks.
- `pitch_lubrication_3130/` — independent pitch-lubrication experiments.
- `untwisting_cables_10105/` — independent untwisting-cable experiments.

All three alarm-set folders contain matching horizon folders and training notebooks. Each alarm set is trained independently; existing icing settings are unchanged.

Pitch-lubrication (3130) and untwisting-cables (10105) now mirror all five icing horizon folders, including the all-feature and selected-feature variants available at each horizon. The new notebooks have not been trained.

Every alarm group now includes all three notebook roles in each of the five horizon folders: all features, selected features, and the selected-feature variant. New notebooks have not been trained.


## Prediction horizon order

- `01_horizon_10min/`
- `02_horizon_20min/`
- `03_horizon_30min/`
- `04_horizon_1hour/`
- `05_horizon_6hours/`
- `06_horizon_1day/`
- `07_horizon_1week/`

Each horizon contains all-features, selected-features, and selected-features variant notebooks. The sensor lookback remains two hours. New horizons require independent training.


## Training runtime settings

All seven horizons and all three notebook variants retain a two-hour sensor lookback. Lightweight logistic regression is enabled by default (`TRAIN_LOGISTIC_REGRESSION = True`) and uses `lbfgs`, 100 iterations and tolerance 1e-3. Random forest, LightGBM, histogram XGBoost, constant prevalence and random-weight baselines run by default. XGBoost uses 100 depth-4 trees and four CPU threads; reduce `TREE_ESTIMATORS` for a quicker pilot. Install the Iteration2 requirements in the notebook kernel before running. Training prints elapsed fit/scoring time per model. This change applies to `05_model_training`; the separate cross-validation experiment retains its declared candidates.


Training defaults pool turbines **2304510, 2304511, 2304512, 2304513 and 2304514**. Histories are calculated separately per turbine; models are fitted on combined training rows. Lightweight logistic regression uses `lbfgs`, `max_iter=100`, and `tol=1e-3`, with training-only imputation and scaling. Reports include the turbine IDs in their folder name to preserve previous single-turbine results. Prediction-history plots still show one configured turbine at a time.


## Five-year training defaults

Only 2016, 2017, 2018, 2020 and 2021 are loaded. Train: 2016-2018; validation/threshold selection: 2020; test: 2021. All five turbines and the two-hour lookback are retained. Report folders include the selected years so previous experiments are preserved. This applies to `05_model_training`; the separate all-turbine cross-validation workflow is unchanged.


All-alarm experiments: `all_alarms/` contains 10-, 20- and 30-minute horizons, each with all-features, selected-features and variant notebooks. See its README for the all-code target definition.


Report folders now use compact names such as `all_alarms_30m_all_<hash>`. The digest distinguishes targets, turbine IDs, selected years, horizons, split dates, seeds and feature selections. Exact settings remain in `configuration.json`; existing renamed reports are listed in `reports/report_folder_renames.json`.
