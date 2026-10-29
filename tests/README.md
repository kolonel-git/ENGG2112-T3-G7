# Check Excel files for missing data

Use `check_excel_missing_data.py` as a standalone, read-only checker. It never edits
your workbooks. Running `pytest` tests the checker itself; run the script to check data.

## Pick the files

Edit **`CHECK_PATH`** near the top of `check_excel_missing_data.py`:

```python
# All workbooks in processed, including subfolders (the default):
CHECK_PATH = ROOT / "data" / "processed"

# Or just one workbook:
CHECK_PATH = ROOT / "data/processed/turbine_exports/turbines_2018_06.xlsx"

# Or all interim workbooks:
CHECK_PATH = ROOT / "data" / "interim"
```

Keep only the assignment you want. Run from the project root:

```powershell
python tests/check_excel_missing_data.py
```

You can override the path without editing the script:

```powershell
python tests/check_excel_missing_data.py --path "data/processed/turbine_exports/turbines_2018_06.xlsx"
python tests/check_excel_missing_data.py --path data/interim --report-dir data/processed/interim_checks
```

On the current computer, the working interpreter is Blender's bundled Python, and
the Excel library is installed in this project's `.venv/Lib/site-packages`. Use:

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location) '.venv/Lib/site-packages')
& 'C:\Program Files\Blender Foundation\Blender 5.1\5.1\python\bin\python.exe' tests/check_excel_missing_data.py
```

For a regular project Python environment, install the dev dependencies with
`python -m pip install -e ".[dev]"` once.

## Results

The console prints PASS/FAIL for each sheet. By default, reports are written to
`data/processed/missing_data_checks/` (replaced on each run):

- `summary.csv`: per-sheet row counts, rows with issues, missing cells, missing
  required columns, and file errors.
- `feature_details.csv`: per-feature missing count/percentage, Excel errors,
  formulas, and up to ten example Excel row numbers to inspect.

Every row is checked; only the row-number examples are capped. Change
`MAX_EXAMPLE_ROWS` to show more examples. Unreadable workbooks are reported, and
the checker continues with the other files. Exit codes are 0 for complete data,
1 for data/schema issues, and 2 for file errors or no matching files.

## What counts as an issue

- Empty cells, whitespace, NaN, NA, N/A, NULL, None, NaT, <NA>, and infinite numbers.
- Excel error cells and formulas instead of literal readings (counted separately).
- Missing required headers, duplicate/unnamed headers, or empty sheets.

All present columns are checked except `Alarmcode` and `AlarmEvents`. Blank alarm
fields are expected when no alarm starts at that timestamp. Modify `IGNORE_COLUMNS`
if you deliberately want to require those fields too.

`REQUIRED_COLUMNS` checks the keys and requested sensor headers. The two features
unavailable in June (`wtc_InvUpTmp_mean`, `wtc_TrafoTMa_mean`) may be absent, but if
they are present, their blank cells are reported. Change `ALLOWED_ABSENT_COLUMNS`
when expectations change for another dataset. Interim files will correctly report
the alarm-only rows and the entirely blank unavailable feature columns.

This is a cell-completeness check. It does not detect entirely missing timestamps,
validate sampling intervals, or judge whether nonmissing values are physically plausible.
