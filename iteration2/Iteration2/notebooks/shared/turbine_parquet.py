"""Persistent typed sensor and alarm caches for the feature notebooks."""
import json
import os
from pathlib import Path
from uuid import uuid4

import pandas as pd

CACHE_VERSION = 1


def load_turbine(workbook_path, turbine_id, cache_root, rebuild=False):
    """Read a current Parquet cache, or build it once from the Excel turbine sheet."""
    workbook_path = Path(workbook_path).resolve()
    stat = workbook_path.stat()
    fingerprint = {"version": CACHE_VERSION, "source": str(workbook_path),
                   "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    sheet = f"Station_{turbine_id}"
    if not str(turbine_id).isdigit():
        raise ValueError("TURBINE_ID must contain digits only.")
    year_folder = workbook_path.stem.removeprefix("turbines_")
    if len(year_folder) != 4 or not year_folder.isdigit():
        raise ValueError("Expected a yearly workbook named turbines_<year>.xlsx")
    folder = Path(cache_root) / year_folder / sheet
    sensors_path = folder / "sensors.parquet"
    alarms_path = folder / "alarms.parquet"
    manifest_path = folder / "source.json"
    try:
        current = json.loads(manifest_path.read_text(encoding="utf-8")) == fingerprint
    except (FileNotFoundError, ValueError):
        current = False
    if not rebuild and current and sensors_path.is_file() and alarms_path.is_file():
        print(f"Loading Parquet: {folder}")
        return pd.read_parquet(sensors_path), pd.read_parquet(alarms_path)

    print(f"Building Parquet cache from Excel ({sheet}); the first load can take minutes...")
    with pd.ExcelFile(workbook_path, engine="openpyxl") as workbook:
        if sheet not in workbook.sheet_names:
            raise ValueError(f"{sheet} is unavailable. Sheets: {workbook.sheet_names}")
        data = pd.read_excel(workbook, sheet_name=sheet)
    features = [c for c in data.columns if c.startswith("wtc_")]
    if not features:
        raise ValueError("No wtc_ sensor features found in the sheet.")
    sensors = data.loc[data[features].notna().any(axis=1), ["TimeStamp", *features]].copy()
    sensors["TimeStamp"] = pd.to_datetime(sensors["TimeStamp"], errors="coerce").astype("datetime64[ns]")
    sensors[features] = sensors[features].apply(pd.to_numeric, errors="coerce")
    sensors = sensors.dropna(subset=["TimeStamp"]).sort_values("TimeStamp").set_index("TimeStamp")
    events = []
    if "AlarmEvents" in data.columns:
        for payload in data["AlarmEvents"].dropna():
            if str(payload).strip():
                events.extend(json.loads(payload))
    else:
        print("AlarmEvents column unavailable: no alarm overlay will be cached.")
    alarms = pd.DataFrame(events, columns=["Alarmcode", "TimeOn", "TimeOff"])
    alarms["Alarmcode"] = alarms["Alarmcode"].astype(str)
    for column in ["TimeOn", "TimeOff"]:
        alarms[column] = pd.to_datetime(alarms[column], errors="coerce").astype("datetime64[ns]")
    alarms = alarms.dropna(subset=["TimeOn"])
    if workbook_path.stat().st_mtime_ns != stat.st_mtime_ns or workbook_path.stat().st_size != stat.st_size:
        raise RuntimeError("The workbook changed during conversion. Rerun after the export completes.")
    folder.mkdir(parents=True, exist_ok=True)
    # Publish the manifest last so interrupted conversions are rebuilt next time.
    manifest_path.unlink(missing_ok=True)
    token = uuid4().hex
    temporary = [folder / f".{token}-{name}" for name in ["sensors.parquet", "alarms.parquet", "source.json"]]
    try:
        sensors.to_parquet(temporary[0], engine="pyarrow", compression="snappy")
        alarms.to_parquet(temporary[1], engine="pyarrow", compression="snappy", index=False)
        temporary[2].write_text(json.dumps(fingerprint, indent=2), encoding="utf-8")
        for staged, target in zip(temporary, [sensors_path, alarms_path, manifest_path]):
            os.replace(staged, target)
    finally:
        for staged in temporary:
            staged.unlink(missing_ok=True)
    print(f"Saved Parquet cache: {folder}")
    return sensors, alarms
