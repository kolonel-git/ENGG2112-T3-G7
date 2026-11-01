# Machine-learning pipeline: run order

Run notebook cells top to bottom. Preparation is sequential; plotting and training notebooks are independent consumers of the prepared data.

| Order | Folder | Purpose |
|---|---|---|
| 1 | [01_data_export](01_data_export/) | Raw yearly CSVs to Excel workbooks. The second notebook is an optional alternative 10-minute Excel export. |
| 2 | [02_parquet_setup](02_parquet_setup/) | Exported Excel to source files in Data_Parquets. |
| 3 | [03_preprocessing](03_preprocessing/) | Regular 10-minute sensor/alarm rows in Data_Parquets_10min, including selected-alarm labels. |
| 4 | [04_data_visualisation](04_data_visualisation/) | Inspect individual features or day/week/month/year plots. Optional before training. |
| 5 | [05_model_training](05_model_training/) | Train and evaluate models by alarm group, horizon, and sensor selection. |

If processed data already exists, go directly to plotting or training. Existing data and report folders stay at the Iteration2 workspace root. Notebook paths are detected from their new locations. Python helpers in shared/ are imported automatically; do not run them separately.

## Current alarm group

All existing training notebooks currently target the icing group: **8230, 8234, 8235, 8236**. Their folder is [05_model_training/icing_8230_8234_8235_8236](05_model_training/icing_8230_8234_8235_8236/).

| Subfolder | Horizon | Versions |
|---|---|---|
| 01_horizon_10min | Next 10 minutes | All features; selected features; preserved selected-feature variant |
| 02_horizon_1hour | Next hour | All features; selected features; selected-feature variant |
| 03_horizon_6hours | Next 6 hours | All features; selected features; selected-feature variant |
| 04_horizon_1day | Next day | All features; selected features; selected-feature variant |
| 05_horizon_1week | Next week | All features; selected features; selected-feature variant |

Every model notebook still uses the preceding hour of sensor readings and predicts every 10 minutes. Each trains independently; you do not need to run earlier horizon notebooks first. Edit ALARM_CODES, TURBINE_IDS and (where applicable) SELECTED_FEATURES in settings. If you save a notebook for another alarm set, place that copy in a correspondingly named alarm-set folder.

The two 10-minute selected-feature versions are both retained, even though their current selected sensor lists match. No experiment settings, data files or saved models were changed by this reorganisation.

See [NOTEBOOK_LOCATIONS.md](NOTEBOOK_LOCATIONS.md) to find notebooks by their old filenames. Close old notebook tabs without saving them over the moved files, then reopen from the new folders.

All three alarm groups now have identical notebook roles at every horizon: `01_train_all_features.ipynb`, `02_train_selected_features.ipynb`, and `03_train_selected_features_variant.ipynb` (45 notebooks total). The variant may duplicate the selected-feature experiment when settings match.


## Stage 6: all-turbine temporal cross-validation

[06_all_turbine_cross_validation](06_all_turbine_cross_validation/README.md) contains an independent 6-hour 8200-family experiment using all eligible turbines. It develops on 2016–2023 and keeps 2024–2025 disabled until explicitly enabled for testing. Run after Stage 3; the notebooks in Stage 5 are not prerequisites. See its README for the weighted negative-sampling default and optional all-row training.
