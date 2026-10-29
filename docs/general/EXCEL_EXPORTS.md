# Excel export and cleaning

The current Excel entry point takes an input folder, filename pattern, and output
path. For example:

```powershell
.\.venv\Scripts\python.exe scripts/data_preparation/export_turbines.py --input-dir data/raw/2018 --pattern "*_2018_*.csv" --output data/interim/turbine_exports/turbines_2018.xlsx
```

The interim workbook is written to
`data/interim/turbine_exports/turbines_2018.xlsx`. It preserves exact alarm start
timestamps, including events between the 10-minute sensor readings. Each turbine
has its own sheet; `AlarmEvents` contains event start/end details.

To clean existing interim workbooks, run:

```powershell
.\.venv\Scripts\python.exe scripts/data_preparation/clean_interim.py
```

The processed copies go under `data/processed/` with matching relative paths.
Cleaning removes unavailable sensor columns and rows missing any remaining
required sensor value. **Alarm-only rows between sensor readings are removed
from processed Excel copies.** Use the interim workbook for complete exported
alarm timing. Raw files and interim workbooks are unchanged by cleaning.

The current exporter writes only the interim workbook; run `clean_interim.py`
separately to create the processed copy. To see available export options:

```powershell
.\.venv\Scripts\python.exe scripts/data_preparation/export_turbines.py --help
```

See [the data preparation guide](../../scripts/data_preparation/README.md) for more
Excel options.


Run these commands from the repository root after [general setup](USAGE.md).
This workflow is independent of the Parquet modelling pipeline.
