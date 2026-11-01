# Stage 4: optional sensor and alarm plots

Requires processed `Data_Parquets_10min` from Stage 3. These notebooks are independent views; choose the duration you need.

| Notebook | View |
|---|---|
| [01_plot_turbine_feature](01_plot_turbine_feature.ipynb) | One selected feature |
| [02_all_features_day](02_all_features_day.ipynb) | All features for a day |
| [03_all_features_week](03_all_features_week.ipynb) | All features for a week |
| [04_all_features_month](04_all_features_month.ipynb) | All features for a month |
| [05_all_features_year](05_all_features_year.ipynb) | All features for a year |

Edit the turbine, year/date window and alarm-display settings, then Run All. Plot display filters are independent of model-training targets. To inspect raw table values and column names, use [preview_parquet](../02_parquet_setup/preview_parquet.ipynb).

Next: [training choices](../05_model_training/README.md). Plots are optional and do not prepare model inputs. [Full guide](../../README.md).
