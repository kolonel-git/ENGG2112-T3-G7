# Notebook directory

**Start with [the project run guide](../README.md) or [00_START_HERE](../00_START_HERE.ipynb).** Run cells top to bottom; restart the kernel before changing experiment settings.

| Order | Folder | When to run |
|---|---|---|
| 1 | [01_data_export](01_data_export/README.md) | You have raw yearly CSVs |
| 2 | [02_parquet_setup](02_parquet_setup/README.md) | You have yearly Excel exports |
| 3 | [03_preprocessing](03_preprocessing/README.md) | You have source sensor/alarm Parquets |
| 4 | [04_data_visualisation](04_data_visualisation/README.md) | Optional sensor/alarm inspection after preprocessing |
| 5 | [05_model_training](05_model_training/README.md) | Choose one alarm group, horizon and feature mode |
| 6 | [06_all_turbine_cross_validation](06_all_turbine_cross_validation/README.md) | Separate advanced temporal-CV experiment after preprocessing |

Stages 1–3 prepare data once. Stages 4 and 5 consume processed data independently. Stage 6 is an alternative experiment, not a continuation of every Stage 5 notebook. View saved CV results in [evaluate_cross_validation_results.ipynb](../evaluate_cross_validation_results.ipynb).

`shared/` contains imported helpers, not notebooks to run manually. [NOTEBOOK_LOCATIONS.md](NOTEBOOK_LOCATIONS.md) maps historical filenames; it is not the current run order.

There are 72 Stage 5 notebooks: three variants × seven horizons × three specific alarm groups, plus three variants × three horizons for all alarms. You only need to run the experiments you want.
