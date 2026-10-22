# Multi-year modelling workflowFor the new **single-turbine, one-command 2018–2025 experiment**, see[this guide](SINGLE_TURBINE.md). The workflow below is the original sharedpipeline using the main project protocol.

First follow [general setup](../general/USAGE.md), prepare raw CSVs, and install dependencies.
Run these commands from the repository root in order, waiting for each to finish:

```powershell
.\.venv\Scripts\python.exe scripts/build_alarm_table.py
.\.venv\Scripts\python.exe scripts/build_scada_parquet.py --workers 4
.\.venv\Scripts\python.exe scripts/build_clean_data.py
.\.venv\Scripts\python.exe scripts/build_labels.py
.\.venv\Scripts\python.exe scripts/build_features.py
.\.venv\Scripts\python.exe scripts/check_feature_leakage.py
.\.venv\Scripts\python.exe scripts/run_dummy_baselines.py
```

| Script | What it produces or checks |
|---|---|
| `build_alarm_table.py` | `data/interim/alarm_events.parquet`, containing configured in-scope alarm codes. Use `--keep-other` to also save out-of-scope events separately. |
| `build_scada_parquet.py` | Selected SCADA columns, one Parquet per table and year in `data/interim/scada/`. |
| `build_clean_data.py` | Joined and cleaned SCADA and daily summaries in `data/interim/clean/`, plus cleaning reports. |
| `build_labels.py` | Fault/stop episodes and future-event labels in `data/interim/labels/`. |
| `build_features.py` | Daily model inputs in `data/interim/features/daily_features.parquet`. |
| `check_feature_leakage.py` | Checks sampled turbine-days for accidental use of future information. |
| `run_dummy_baselines.py` | Constant/random reference scores in `reports/tables/baselines_dummy.csv`. This does not train the main predictive model. |

The build scripts scan available years from 2016 through 2026 independently of
the Excel export's input selection. Cleaning currently skips years
without all three SCADA table outputs. In particular, 2019 lacks temperature and
pressure data and is excluded from the configured modelling splits.

Parquet preserves data types and supports efficient compressed storage and
column selection. Raw CSV files remain the original source. These scripts do
not use the cleaned Excel workbook as their input.


## Evaluation scope

`run_dummy_baselines.py` selects alert thresholds using the configured validation
years and freezes them for test years. The current protocol uses 2016–2018 and
2020 for training, 2021–2022 for validation, and 2023–2026 for testing.
The dummy scorers themselves do not fit models. No validation/test rows means
the main baseline experiment cannot run. For an exploratory alternative using
2018 only, see [single-year instructions](../single_year/README.md).

The metric formulas are shared with the [single-year metric reference](../single_year/EVALUATION.md#metrics).
The chronological folds described in that reference apply only to the single-year runner.

## Rebuilding

SCADA conversion skips existing output files; remove only the generated table/year
outputs you intend to rebuild. Subsequent builders overwrite named outputs.
Rebuild downstream labels/features/results when their inputs change.
See [general usage](../general/USAGE.md) for troubleshooting and tests.
