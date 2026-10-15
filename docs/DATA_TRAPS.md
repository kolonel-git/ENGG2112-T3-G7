# Data traps

Record dataset surprises here (missing sensors, unit changes, upgrades, timestamp issues, duplicate rows).

| Date | Table / field | What we found | How we handle it | Owner |
|------|---------------|---------------|------------------|-------|
| 2026-10-15 | Windows processed workbook access | Moving output out of Python 3.13's private temporary directory retained an ACL without the user's normal access permissions. | Restore inherited permissions on the affected workbook; publish future workbooks through a fresh file in the destination folder. Add a Windows ACL regression test and an independent Excel-library read test. | Data preparation |
| 2026-10-15 | June 2018 temperature headers | Requested temperature names have a `wtc_` prefix. `wtc_InvUpTmp_mean` and `wtc_TrafoTMa_mean` are absent. | Preserve source names; export absent columns as blanks and list them in the export report. | Data preparation |
| 2026-10-15 | June 2018 alarm keys/timestamps | Alarms use `StationNr`, `TimeOn`, `TimeOff`, and `Alarmcode`; starts often fall between sensor timestamps. | Map `StationNr` to `StationId`; preserve exact start times as separate rows and retain end times in `AlarmEvents`. No forward-filling or interval assumptions. | Data preparation |
