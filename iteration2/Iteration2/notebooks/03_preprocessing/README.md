# Stage 3: regular ten-minute data

[Full run guide](../../README.md). Prerequisite: source sensor/alarm Parquets from Stage 2. Skip if processed data already exists.

Open [01_preprocess_10min_parquets.ipynb](01_preprocess_10min_parquets.ipynb), check `SOURCE_ROOT`, `OUTPUT_ROOT`, `YEARS`, `TURBINE_IDS` and `REBUILD`, then Run All. For the current training scope, select 2016, 2017, 2018, 2020 and 2021. `None` selects all available years/turbines.

Output: `Data_Parquets_10min/<year>/Station_<ID>/data.parquet` and metadata. Each year has a regular ten-minute timeline. Missing sensors remain missing; `sensorObserved` distinguishes inserted rows. All alarm records are retained, including original timestamps used for future-start labels. Selected-alarm flags do not limit the target codes used by training.

Next: [optional plots](../04_data_visualisation/README.md) or [training](../05_model_training/README.md), independently. No need to repeat Excel conversion when changing training settings.
