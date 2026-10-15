# Decision log

Add newest entries at the top. One entry per decision.

## Implemented changes — 2026-10-15

These entries record changes implemented by Codex at the user's request, including
implementation choices made while carrying out that work. They do not settle the
research-protocol TODOs listed below. Detailed dataset observations remain in
[DATA_TRAPS.md](DATA_TRAPS.md); commands are in
[the data preparation README](../scripts/data_preparation/README.md) and
[the test README](../tests/README.md).

### D-006: Select a year folder and generate annual workbooks
- Date: 2026-10-15
- Owner: Codex implementation, requested by the user.
- Context: Monthly files are now organised under `data/raw/<year>/`; the user wants
  to change one setting and run the annual formatting locally.
- Decision: Update `scripts/data_preparation/export_turbines.py` with
  `YEAR_FOLDER = "2018"`. Changing that string, or passing `--year`, selects the
  input folder and matching monthly filenames, and names the outputs automatically.
- Changes: Validate the four-digit year and input folder; support `--input-dir`,
  `--pattern`, `--output`, and `--interim-output` overrides. Combine the months into
  `data/interim/turbine_exports/turbines_<year>.xlsx`, then run the existing cleaner
  to produce `data/processed/turbine_exports/turbines_<year>.xlsx`. `--keep-missing`
  skips cleaning and exports directly to the output path. Existing annual outputs
  are replaced on rerun; monthly workbooks remain separate.
- Consequences: Both workbooks retain separate sheets per station, ordered by
  timestamp. The processed copy applies D-002; it is not a lossless alarm history.
- Execution status: A full-year export was started, then stopped when the user
  requested script changes only. No completed annual workbook was produced or
  validated in that run. An independent source audit was completed before the stop
  and saved as `data/interim/turbine_exports/turbines_2018.source_audit.json`.
  The latest year-folder paths were verified with mocked export/cleaning calls,
  without processing the datasets.

### D-005: Handle date formats and overlapping monthly boundaries
- Date: 2026-10-15
- Owner: Codex implementation while preparing the annual export.
- Context: The October pressure file uses day/month/year dates and rounded readings;
  adjacent monthly files overlap at midnight.
- Decision: Extend `timestamp()` in `src/wt_predmaint/data/turbine_export.py` to
  parse slash-separated dates explicitly as day-first, with optional seconds.
  ISO dates retain their existing handling; timestamps are not rounded or shifted.
- Changes: Merge duplicate sensor keys whose selected values match within
  `rel_tol=1e-9`, `abs_tol=1e-8`, retaining the longer numeric representation.
  Count these separately as `rounded_duplicates_merged`; identical duplicates are
  counted as `exact_duplicates_removed`. Larger conflicts still stop the export.
  Add tests for day-first dates, cross-month joins, and retained numeric precision.
- Consequences: Available monthly readings can share one chronological sheet per
  station without duplicate boundary rows. Source auditing found missing January
  and February pressure files, one alarm-only station `91`, and year-end midnight
  readings at `2019-01-01 00:00:00`. These findings were recorded in DATA_TRAPS;
  station 91 is not assumed to be an additional turbine, and boundary readings are
  retained. Missing source months are not fabricated.

### D-004: Add a batch Excel completeness checker
- Date: 2026-10-15
- Owner: Codex implementation, requested by the user.
- Decision: Add `tests/check_excel_missing_data.py` as a standalone read-only tool,
  with `CHECK_PATH` selecting a single workbook or a folder scanned recursively.
  Command-line `--path` and `--report-dir` overrides are also available.
- Changes: Check every sheet for missing cells and required columns, empty sheets,
  duplicate/unnamed headers, Excel error cells, and formulas in place of readings.
  Treat blank/whitespace values and common textual missing-value markers as missing.
  Allow blank alarm fields. The two unavailable June features may be absent, but
  their cells are checked if those columns are present.
- Outputs: `data/processed/missing_data_checks/summary.csv` and
  `feature_details.csv`, including counts, percentages, and example Excel row
  numbers. All rows are scanned; only the examples are capped. Unreadable files are
  reported without stopping checks of other workbooks. Exit codes distinguish
  success, data issues, and unreadable/no input files.
- Supporting files: Add `tests/test_missing_data_checker.py` and `tests/README.md`.
- Validation: The June processed workbook passed with 90,088 complete sensor rows
  across 21 sheets. This checks cells, not missing timestamps, sampling intervals,
  physical plausibility, or fault labels.

### D-003: Repair Windows workbook permissions and strengthen validation
- Date: 2026-10-15
- Owner: Codex implementation after the user reported an Excel opening error.
- Context: Moving a workbook out of a private Python 3.13 temporary directory
  retained restrictive Windows permissions. Earlier ZIP/XML checks did not detect
  the access problem.
- Decision: Restore inherited permissions on the affected processed workbook.
  Add `publish_workbook()` to copy completed bytes into a fresh sibling file in
  the destination folder before atomically replacing the output. Use it in both
  the exporter and cleaner so output permissions inherit from the destination.
- Changes: Add a Windows permission regression test and independent `openpyxl`
  read checks; add `openpyxl>=3.1` to the project's dev dependencies. Install
  openpyxl, pytest, and Ruff locally under `.venv/Lib/site-packages` and restore
  inherited permissions on that local package directory.
- Validation: Both June workbooks opened successfully in Microsoft Excel, read-only,
  with 21 sheets each. Tests and Ruff subsequently passed.

### D-002: Produce cleaned workbooks without changing interim data
- Date: 2026-10-15
- Owner: Codex implementation, requested by the user.
- Decision: Add `src/wt_predmaint/data/clean_workbook.py` and the entry point
  `scripts/data_preparation/clean_interim.py`. Process exported interim workbooks
  recursively into `data/processed/`, preserving relative paths and station sheets.
- Cleaning rules: Use the export's `.report.json` to remove unavailable feature
  columns. Remove rows missing a timestamp, station ID, or any remaining sensor
  reading. Blank `Alarmcode`/`AlarmEvents` fields are allowed. Remove alarm-only
  rows; keep empty station sheets if no rows survive. Do not interpolate, scale,
  remove outliers, replace undocumented sentinel values, or construct fault labels.
- Outputs: A processed workbook plus a `.cleaning.json` report with per-sheet
  input/kept/removed row counts and the number of alarm events in removed rows.
  Interim workbooks remain unchanged and retain the original alarm history.
- Validation: The June workbook retained 90,088 rows and removed 112,154 rows,
  mostly alarm-only records. All retained sensor cells were independently checked
  for completeness. Tests cover complete, partially missing, and alarm-only rows.
- Consequences: For annual data, months missing any required sensor table will
  lose incomplete rows in the processed copy. Use the interim workbook for alarm
  labelling and whenever those incomplete readings must be preserved.

### D-001: Extract requested features into station-specific Excel sheets
- Date: 2026-10-15
- Owner: Codex implementation, requested by the user.
- Context: The project initially contained package placeholders and a downloader,
  with no implemented feature-extraction pipeline.
- Decision: Create `scripts/data_preparation/` and its README. Implement reusable
  extraction in `src/wt_predmaint/data/turbine_export.py`, called through
  `scripts/data_preparation/export_turbines.py`. Use one workbook with a separate
  `Station_<ID>` sheet for each existing station ID, rather than assigning new IDs.
- Features: Select the user's pressure, temperature, wind-speed, generator/rotor
  speed, and power-setpoint columns. The exact list is the module's `FEATURES`
  mapping. Preserve the CSV's `wtc_` temperature prefix. For the broader requested
  temperature groups, select available mean A1/A3/A9/A21, DeltaTmp, brake, filter,
  and reactor signals. Exclude `tblDailySummary` because no features were requested.
- Alignment: Outer-join sensor records on exact `(StationId, TimeStamp)` keys.
  Map alarm `StationNr` to `StationId` and `TimeOn` to `TimeStamp`. Retain exact
  alarm times as additional rows when they fall between sensor readings. Export
  semicolon-separated `Alarmcode` values and `AlarmEvents` JSON preserving each
  event's code, start, and end; retain repeated events. Do not forward-fill alarms.
- Missing features: Leave `wtc_InvUpTmp_mean` and `wtc_TrafoTMa_mean` blank when
  absent from the source, and record missing columns in the export report.
- Safeguards: Validate IDs, timestamps, numeric values, duplicate keys, Excel row
  limits, and cell lengths. Export timestamps/IDs as text and sensor values as
  numbers. Write a `.report.json` with input paths, missing columns, duplicate
  counts, station row counts, and alarm-only station/row information. Original raw
  CSVs are read-only; generated workbooks remain under git-ignored data folders.
- Supporting changes: Add `tests/test_turbine_export.py` for alignment, station
  separation, missing features, duplicate handling, and subsequent regression tests.
- Validation: The June interim workbook contained 21 sheets and 202,242 rows;
  comparison against the source retained all timestamp keys and 113,637 alarm events.
  After the annual-format fixes, the suite passed 12 tests and Ruff checks. No
  model training or research-protocol settings were changed.

## Template
### D-000: <short title>
- Date: YYYY-MM-DD
- Owner: <name>
- Context: <why a decision is needed>
- Options considered: <a / b / c>
- Decision: <what was chosen>
- Consequences: <what changes, what is now fixed>

## Open decisions
- TODO: dataset version. Pinned to v2.1.0 (DOI 10.5281/zenodo.22662930); brief cited v1.0.0 DOI 10.5281/zenodo.14870023.
- TODO: train/val/test year ranges
- TODO: label windows, false-alarm budget, random seed (see `configs/protocol.yaml`)
- TODO: licence
