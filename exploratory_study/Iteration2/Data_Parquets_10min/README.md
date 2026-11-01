# Regular 10-minute turbine data

Create/update these files with `notebooks/03_preprocessing/01_preprocess_10min_parquets.ipynb`, after running notebook 02 for source Parquet.

Layout: `<year>/Station_<ID>/data.parquet`. The timestamp index is the start of a half-open 10-minute interval. Each year includes its full grid; `sensorObserved` distinguishes source readings from inserted missing intervals.

`alarmOccurred`, `triggeredAlarmCodes`, `triggeredAlarmCounts` and `alarmTriggerCount` describe NEW starts. `alarmOn` and `activeAlarmCodes` describe ANY active overlap. `alarmEvents` retains exact event timing inside JSON; it is not a sensor feature. Carry-in events from available prior years are retained in Parquet metadata for interval overlap calculations. Missing/zero alarm durations flag only their start interval.

Notebooks 04â€“08 and 09 read this format. Models select only `wtc_` columns as features. Rerun preprocessing after changing source data and retrain models afterwards. Original source Parquet files are preserved in Data_Parquets.

## Selected training alarms

`selectedAlarmOccurred` is true only for new starts of 3130, 8000, 8235 or 10105. The corresponding selected codes and counts are in `selectedTriggeredAlarmCodes`, `selectedTriggeredAlarmCounts` and `selectedAlarmTriggerCount`. These fields do not remove or replace the all-code `alarmOccurred`, `alarmEvents` or original alarm-count columns. Selected flags refer to starts, not ongoing activity. A model for a single code must still use that code's individual label; the combined flag means any selected code started.
