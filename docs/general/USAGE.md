# How to use this repository

This project prepares Hill of Towie wind-turbine data for predictive maintenance.
There are two independent workflows: Excel exports for inspection and Parquet
tables for feature building and evaluation. You do not need to build Excel
workbooks before running the modelling pipeline.

## 1. Install the dependencies

Install Python 3.10 or newer. Open a terminal in the repository root (the folder
containing `pyproject.toml`, `scripts/`, and `docs/`). All commands below assume
you are in that folder.

For Windows PowerShell, create an isolated Python environment and install:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r docs/general/requirements.txt
```

In VS Code, run **Python: Select Interpreter** from the command palette and
choose `.venv\Scripts\python.exe`. This makes the Run button use the installed
dependencies. The commands below use that interpreter explicitly, so activating
the environment is optional.

If you already have a suitable environment, install into it instead:

```powershell
python -m pip install -r docs/general/requirements.txt
```

Then replace `.\.venv\Scripts\python.exe` below with your environment's Python.
On macOS/Linux, the equivalent virtual-environment interpreter is
`./.venv/bin/python`.

The requirements file installs this repository in editable mode along with the
runtime and development packages declared in `pyproject.toml`. Standard-library
modules such as `csv`, `pathlib`, and `sqlite3` need no separate installation.

## 2. Put the raw data in place

Keep the original monthly CSV files under `data/raw/<year>/`, for example:

```text
data/raw/2018/tblSCTurbine_2018_01.csv
data/raw/2018/tblSCTurTemp_2018_01.csv
data/raw/2018/tblSCTurPress_2018_01.csv
data/raw/2018/tblAlarmLog_2018_01.csv
data/raw/2018/tblDailySummary_2018_01.csv
```

The SCADA modelling reader expects CSVs directly inside each year folder.
If extraction creates `data/raw/2018/2018/`, place the CSVs in the outer year
folder before running that pipeline. Preserve the original filenames.

If you need to download data, first list the available files, then select years:

```powershell
.\.venv\Scripts\python.exe scripts/download_data.py --list
.\.venv\Scripts\python.exe scripts/download_data.py --years 2018 2020
```

The downloader saves ZIPs and metadata to `data/raw/`, verifies available
checksums, and skips matching existing files. It does **not** extract ZIPs.
Extract the year archives for the SCADA, daily-summary, and Excel workflows.
The alarm reader can also read year ZIPs; avoid keeping both ZIPs and extracted
copies of the same alarm source in its search tree.

## 3. Choose a workflow

| Task | Instructions |
|---|---|
| One-command single-turbine experiment over 2018–2025 | [Multi-year single-turbine workflow](../multi_year/SINGLE_TURBINE.md) |
| Within-year classifier, plots, and baselines, default 2018 | [Single-year workflow](../single_year/README.md) |
| Data preparation and evaluation across multiple years | [Multi-year workflow](../multi_year/WORKFLOW.md) |
| Excel exports for human inspection | [Excel export and cleaning](EXCEL_EXPORTS.md) |

The single-year experiment still needs the shared data preparation steps through
`build_labels.py`. Neither dummy baseline runner trains a sensor-based model.
See the [script directory guide](../../scripts/README.md) for script scopes.

## 4. Reruns and checking results

- `build_scada_parquet.py` skips existing Parquet outputs. To rebuild a specific
  table/year after changing its source or reader, delete that generated file
  first, then rerun the downstream steps.
- The other build scripts replace their named outputs when rerun. Rebuild
  downstream tables whenever their inputs or relevant configuration change.
- Excel exports and cleaning replace outputs at the same paths. Close the
  destination workbook in Excel first so Windows can replace it.
- Inspect `reports/tables/` for pipeline summaries and the JSON sidecars beside
  exported/cleaned workbooks for counts and missing-feature information.

Run the automated tests from the repository root:

```powershell
.\.venv\Scripts\python.exe -m pytest
```

## Common problems

| Problem | What to check |
|---|---|
| `No module named wt_predmaint` | Install `docs/general/requirements.txt` using the same interpreter that runs the script. |
| Missing `pandas`, `yaml`, `pyarrow`, or another library | Check the IDE interpreter and rerun the installation command in that environment. |
| `python` opens the Microsoft Store or cannot run | Use the full path to an installed Python interpreter, or `py` if the Python launcher is configured. |
| Missing input Parquet files or no objects to concatenate | Run the earlier pipeline stages and check their skipped-year messages and raw-data paths. |
| Permission error replacing an Excel workbook | Close the workbook and check write access to its output folder. |

## Project reference files

- [requirements.txt](requirements.txt): installation command and library descriptions.
- [protocol.yaml](../../configs/protocol.yaml): shared cleaning, label, feature, and evaluation settings.
- [DECISIONS.md](../project/DECISIONS.md): reasoning behind project choices; record protocol changes here.
- [DATA_TRAPS.md](../project/DATA_TRAPS.md): known dataset issues and limitations.
- [roadmap.md](../project/roadmap.md): planned work and responsibilities.
