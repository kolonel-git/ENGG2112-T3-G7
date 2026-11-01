# Stage 2: Excel to source Parquets

[Full run guide](../../README.md). Skip this stage when `Data_Parquets` or `Data_Parquets_10min` already exists in the required schema.

1. Put yearly `turbines_<year>.xlsx` in `Data/Interim/turbine_exports/`.
2. Open [01_setup_yearly_parquets.ipynb](01_setup_yearly_parquets.ipynb).
3. Set `YEARS` to existing workbooks, choose `TURBINE_IDS`, leave `REBUILD = False` to reuse valid outputs, then Run All.
4. Output: `Data_Parquets/<year>/Station_<ID>/sensors.parquet`, `alarms.parquet` and metadata.
5. Next: [Stage 3](../03_preprocessing/README.md).

[preview_parquet.ipynb](preview_parquet.ipynb) previews the **processed Stage 3 output**, not the Stage 2 source files. Run it after preprocessing, or immediately if processed files are already available. Edit `YEAR`, `TURBINE_ID` and `ROWS`.
