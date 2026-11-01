# All-alarm prediction

Run one of the following independent experiments after preprocessing:

- `01_horizon_10min/`
- `02_horizon_20min/`
- `03_horizon_30min/`

Each contains `01_train_all_features.ipynb`, `02_train_selected_features.ipynb`, and `03_train_selected_features_variant.ipynb`. Edit `SELECTED_FEATURES` in selected-feature notebooks as needed; the all-features version uses every sensor feature.

The label is 1 if **any recorded alarm code starts** in `[t, t + horizon)`, otherwise 0. All codes, including operational alarms 20 and 25, are retained; alarm flags and counts are never model inputs. `ALARM_CODES = ['ALL']` is a descriptive sentinel, not an actual source alarm code. Target events are intentionally unfiltered. Already-active alarms exclude prediction windows, consistent with the other onset-prediction notebooks. Frequent operational alarms can substantially reduce eligible windows and change prevalence; both classes must remain available in every split.

Defaults: turbines 2304510-2304514, a two-hour sensor history, train 2016-2018, validation 2020, test 2021. Models: constant prevalence, random-weight score, lightweight logistic regression, random forest, LightGBM and XGBoost. Run All independently for each horizon. Results are stored separately under the `all_alarms_ALL` target. Historical test-year reuse means these are exploratory evaluations.


Report folders now use compact names such as `all_alarms_30m_all_<hash>`. The digest distinguishes targets, turbine IDs, selected years, horizons, split dates, seeds and feature selections. Exact settings remain in `configuration.json`; existing renamed reports are listed in `reports/report_folder_renames.json`.
