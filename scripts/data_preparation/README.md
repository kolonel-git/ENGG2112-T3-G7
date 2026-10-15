# Data preparation scripts

Place the Python entry-point scripts for selecting and exporting turbine data here.
Put reusable loading, cleaning, and extraction functions in `src/wt_predmaint/data/`.

## Export selected features

`export_turbines.py` is implemented and needs only Python 3.10+ (no additional packages).
The selected column list is in `src/wt_predmaint/data/turbine_export.py`, in `FEATURES`.

From the repository root, run:

```powershell
python scripts/data_preparation/export_turbines.py --input-dir "path/to/your/CSVs" --pattern "*_2018_06.csv" --output data/interim/turbine_exports/turbines_2018_06.xlsx
```

On this machine, `python` is currently a Windows Store alias and `py` reports no
installed Python. The June 2018 export was run with the existing bundled interpreter:

```powershell
& 'C:\Program Files\Blender Foundation\Blender 5.1\5.1\python\bin\python.exe' scripts/data_preparation/export_turbines.py --input-dir 'C:\Users\parve\Desktop\University Stuff\ENG2112\Project\Data\2018_Only_Necessary' --pattern '*_2018_06.csv' --output data/interim/turbine_exports/turbines_2018_06.xlsx
```

The workbook contains one `Station_<ID>` sheet per turbine, sorted by timestamp.
Sensor values are numeric; timestamps and IDs are text to preserve their representation.
The accompanying `.report.json` records inputs, missing columns, duplicate counts,
and exported row counts. Re-running the command replaces the specified workbook.

## Alignment and cleaning rules

- Sensor tables are outer-joined on exact `StationId` and `TimeStamp` values.
- Alarm `StationNr` maps to `StationId`; `TimeOn` maps to `TimeStamp`.
  Alarms between sensor readings receive their own rows with blank sensor cells.
- `Alarmcode` contains the codes starting at that timestamp, separated by semicolons.
  `AlarmEvents` contains JSON preserving each code's original start and end times.
  A blank code means no alarm **start** recorded at that timestamp, not no active alarm.
- Identical selected sensor records are deduplicated and reported; conflicting
  duplicate keys cause an error. Every alarm event is retained, including repeats.
- Missing sensor values remain blank. No interpolation, scaling, outlier removal,
  alarm labels, or assumed missing-value sentinel replacements are applied.
- Temperature names retain their CSV `wtc_` prefix. The broadly requested temperature
  groups use mean signals: A1ExtTmp, A3LefTmp, A3RigTmp, A9IntTmp, A21IntTm,
  DeltaTmp, BrkTmpGn, GFilB1Tm/B2Tm/B3Tm, and ReacUTmp/VTmp/WTmp.
- `wtc_InvUpTmp_mean` and `wtc_TrafoTMa_mean` are absent from the June temperature CSV;
  their output columns are blank and the report lists them as missing.
- `tblDailySummary` is excluded because no features were requested from it.

This workbook preserves event timing for inspection. Alarm-only rows should be
handled explicitly when constructing a regular sensor series for machine learning.

## Workflow

1. Read original CSV files from `data/raw/`, preserving the originals.
2. Select the feature columns explicitly requested by the team, retaining `StationId`
   and the timestamp needed to identify each observation.
3. Apply agreed cleaning rules and record dataset issues in `docs/DATA_TRAPS.md`.
4. Group observations by `StationId`; preserve existing IDs rather than inventing
   new ones.
5. Export one Excel workbook at `data/interim/turbine_exports/turbines.xlsx`, with
   a separate sheet for each `StationId`. Include `StationId` in the exported rows
   and use it to name each sheet, respecting Excel's sheet-name restrictions and
   preventing name collisions.

The export directory is covered by the repository's existing data ignore rules.
Do not commit datasets or generated Excel files.

## Clean interim workbooks

```powershell
python scripts/data_preparation/clean_interim.py
```

This processes exported `.xlsx` files recursively from `data/interim/` into
`data/processed/`, preserving relative paths, workbook names, and turbine sheets.
Use the bundled interpreter shown above if `python` is unavailable.
Each input needs its accompanying `.report.json` from the exporter.

- Remove the unavailable sensor columns identified in the export report (for June,
  `wtc_InvUpTmp_mean` and `wtc_TrafoTMa_mean`).
- Remove rows with a blank/missing timestamp, station ID, or any remaining sensor value.
- Blank alarm fields are allowed: absence of an alarm start is not missing sensor data.
- Alarm-only rows are removed. Their events remain in the interim workbook; do not
  use the processed workbook alone to construct alarm labels.
- Keep sheets even if no rows survive. Record kept/removed rows and removed alarm
  event counts per sheet in the output's `.cleaning.json` sidecar.
- Leave interim files untouched. Re-running replaces the corresponding processed files.

## Further cleaning decisions

- Rules for missing-value sentinels and physically invalid readings.
- Whether to align alarms to sensor intervals for subsequent modelling, and the
  meaning of sensor timestamps (interval start versus interval end).

Excel sheets can hold
at most 1,048,576 rows (including the header); the export implementation must check
this limit before writing and must not silently truncate data.
