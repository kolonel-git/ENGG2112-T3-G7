# Stage 1: raw CSVs to yearly Excel

[Full run guide](../../README.md). Skip this stage if you already have source or processed Parquets.

1. Put original monthly turbine, pressure, temperature and alarm-log CSVs in `Data/Raw/<year>/`.
2. Open [01_export_yearly_turbine_scada.ipynb](01_export_yearly_turbine_scada.ipynb).
3. Set `YEARS` to your available years (current training uses 2016, 2017, 2018, 2020, 2021), check `RAW_ROOT` and `OUTPUT_DIR`, then Run All.
4. Output: `Data/Interim/turbine_exports/turbines_<year>.xlsx` and export reports.
5. Next: [Stage 2](../02_parquet_setup/README.md).

The [optional ten-minute Excel export](02_optional_export_10min_alarm_scada.ipynb) is for inspection and is not a prerequisite for Stage 2. Reruns may replace configured exports; raw CSVs are preserved.
