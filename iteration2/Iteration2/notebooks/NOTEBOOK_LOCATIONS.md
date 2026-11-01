# Historical notebook filename lookup

For the current run order, use [Start Here](../README.md) and the [notebook directory](README.md). This table is only for locating older filenames. Some old names mention 3130 even when the notebook now targets icing; the current settings define the target.

| Previous filename | New location |
|---|---|
| 01_export_yearly_turbine_scada.ipynb | [01_data_export/01_export_yearly_turbine_scada.ipynb](01_data_export/01_export_yearly_turbine_scada.ipynb) |
| 10_optional_export_yearly_10min_alarm_scada.ipynb | [01_data_export/02_optional_export_10min_alarm_scada.ipynb](01_data_export/02_optional_export_10min_alarm_scada.ipynb) |
| 02_setup_yearly_parquets.ipynb | [02_parquet_setup/01_setup_yearly_parquets.ipynb](02_parquet_setup/01_setup_yearly_parquets.ipynb) |
| 03_preprocess_10min_parquets.ipynb | [03_preprocessing/01_preprocess_10min_parquets.ipynb](03_preprocessing/01_preprocess_10min_parquets.ipynb) |
| 04_plot_turbine_feature.ipynb | [04_data_visualisation/01_plot_turbine_feature.ipynb](04_data_visualisation/01_plot_turbine_feature.ipynb) |
| 05_all_features_day.ipynb | [04_data_visualisation/02_all_features_day.ipynb](04_data_visualisation/02_all_features_day.ipynb) |
| 06_all_features_week.ipynb | [04_data_visualisation/03_all_features_week.ipynb](04_data_visualisation/03_all_features_week.ipynb) |
| 07_all_features_month.ipynb | [04_data_visualisation/04_all_features_month.ipynb](04_data_visualisation/04_all_features_month.ipynb) |
| 08_all_features_year.ipynb | [04_data_visualisation/05_all_features_year.ipynb](04_data_visualisation/05_all_features_year.ipynb) |
| 09_train_alarm_3130.ipynb | [05_model_training/icing_8230_8234_8235_8236/01_horizon_10min/01_train_all_features.ipynb](05_model_training/icing_8230_8234_8235_8236/01_horizon_10min/01_train_all_features.ipynb) |
| 11_train_alarm_3130_selected_features.ipynb | [05_model_training/icing_8230_8234_8235_8236/01_horizon_10min/02_train_selected_features.ipynb](05_model_training/icing_8230_8234_8235_8236/01_horizon_10min/02_train_selected_features.ipynb) |
| 15_train_alarm_group_10min_selected_features.ipynb | [05_model_training/icing_8230_8234_8235_8236/01_horizon_10min/03_train_selected_features_variant.ipynb](05_model_training/icing_8230_8234_8235_8236/01_horizon_10min/03_train_selected_features_variant.ipynb) |
| 16_train_alarm_group_1hour_selected_features.ipynb | [05_model_training/icing_8230_8234_8235_8236/04_horizon_1hour/02_train_selected_features.ipynb](05_model_training/icing_8230_8234_8235_8236/04_horizon_1hour/02_train_selected_features.ipynb) |
| 12_train_alarm_group_6hour.ipynb | [05_model_training/icing_8230_8234_8235_8236/05_horizon_6hours/01_train_all_features.ipynb](05_model_training/icing_8230_8234_8235_8236/05_horizon_6hours/01_train_all_features.ipynb) |
| 13_train_alarm_group_1day.ipynb | [05_model_training/icing_8230_8234_8235_8236/06_horizon_1day/01_train_all_features.ipynb](05_model_training/icing_8230_8234_8235_8236/06_horizon_1day/01_train_all_features.ipynb) |
| 14_train_alarm_group_1week.ipynb | [05_model_training/icing_8230_8234_8235_8236/07_horizon_1week/01_train_all_features.ipynb](05_model_training/icing_8230_8234_8235_8236/07_horizon_1week/01_train_all_features.ipynb) |

New experiment: [All-turbine 6-hour 8200-family CV](06_all_turbine_cross_validation/01_cv_8200_family_all_turbines_6hour.ipynb). This is a new workflow, not a renamed single-turbine notebook.


All-alarm experiments: `05_model_training/all_alarms/` contains 10-, 20- and 30-minute horizons, each with all-features, selected-features and variant notebooks. See its README for the all-code target definition.
