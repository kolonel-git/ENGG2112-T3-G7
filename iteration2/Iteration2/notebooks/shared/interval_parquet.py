"""Build and read regular ten-minute turbine datasets, without altering source files."""

import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

SCHEMA_VERSION = 2
DEFAULT_SELECTED_ALARM_CODES = ("3130", "8000", "8235", "10105")
STEP = pd.Timedelta(minutes=10)


def year_folder(root, year):
    root = Path(root)
    for name in [str(year), f"turbines_{year}"]:
        if (root / name).is_dir():
            return root / name
    raise FileNotFoundError(
        f"No source Parquet folder for {year} in {root}. Run notebook 02 first."
    )


def add_selected_alarm_labels(frame, selected_alarm_codes=DEFAULT_SELECTED_ALARM_CODES):
    """Add selected-start labels without removing any original alarm records."""
    codes = {str(code).strip() for code in selected_alarm_codes}
    if not codes or any(not code for code in codes):
        raise ValueError("Choose at least one nonblank selected alarm code.")
    counts = frame["triggeredAlarmCounts"].map(
        lambda payload: {
            code: count for code, count in json.loads(payload).items() if code in codes
        }
    )
    frame["selectedTriggeredAlarmCounts"] = counts.map(json.dumps)
    frame["selectedTriggeredAlarmCodes"] = counts.map(lambda values: "; ".join(sorted(values)))
    frame["selectedAlarmTriggerCount"] = counts.map(lambda values: sum(values.values())).astype(
        "int32"
    )
    frame["selectedAlarmOccurred"] = frame["selectedAlarmTriggerCount"].gt(0)
    frame.attrs["selected_alarm_codes"] = sorted(codes)
    return frame


def align_year(sensors, alarms, year, turbine, selected_alarm_codes=DEFAULT_SELECTED_ALARM_CODES):
    """Full-year half-open grid. Sensor values stay at their original grid time."""
    begin, finish = pd.Timestamp(year, 1, 1), pd.Timestamp(year + 1, 1, 1)
    grid = pd.date_range(begin, finish, freq=STEP, inclusive="left")
    sensors = sensors.copy()
    if not isinstance(sensors.index, pd.DatetimeIndex) or sensors.index.tz is not None:
        raise ValueError("Expected a timezone-naive sensor timestamp index.")
    sensors = sensors.loc[(sensors.index >= begin) & (sensors.index < finish)]
    if (sensors.index != sensors.index.floor("10min")).any():
        raise ValueError(
            "Off-grid sensor timestamps found; choose an explicit aggregation "
            "policy before preprocessing."
        )
    sensors = sensors.loc[~sensors.index.duplicated(keep="last")].sort_index()
    columns = [c for c in sensors.columns if c.startswith("wtc_")]
    observed = sensors[columns].notna().any(axis=1)
    result = sensors[columns].apply(pd.to_numeric, errors="coerce").reindex(grid)
    result.index.name = "TimeStamp"
    result.insert(0, "StationId", str(turbine))
    result["sensorObserved"] = observed.reindex(grid, fill_value=False).astype(bool)
    starts = defaultdict(Counter)
    changes = defaultdict(Counter)
    events_by_bin = defaultdict(list)
    alarms = alarms.copy()
    for c in ["TimeOn", "TimeOff"]:
        alarms[c] = pd.to_datetime(alarms[c], errors="coerce")
        if alarms[c].dt.tz is not None:
            raise ValueError("Timezone-aware alarms require an explicit policy.")
    if alarms.TimeOn.isna().any():
        raise ValueError("An alarm has a missing/invalid start timestamp.")
    for event in alarms.itertuples(index=False):
        on, off = event.TimeOn, event.TimeOff
        if pd.isna(event.Alarmcode) or not str(event.Alarmcode).strip():
            raise ValueError("Blank alarm code found.")
        code = str(event.Alarmcode).strip()
        if pd.notna(off) and off < on:
            raise ValueError(f"Alarm {code} ends before it starts: {on} -> {off}")
        if begin <= on < finish:
            i = int((on - begin) // STEP)
            starts[i][code] += 1
            events_by_bin[i].append(
                {
                    "Alarmcode": code,
                    "TimeOn": on.isoformat(sep=" "),
                    "TimeOff": None if pd.isna(off) else off.isoformat(sep=" "),
                }
            )
        if pd.notna(off) and off > on:
            if on >= finish or off <= begin:
                continue
            first = int((max(on, begin) - begin) // STEP)
            last = int((min(off, finish).ceil("10min") - begin) // STEP)
        elif begin <= on < finish:
            first = int((on - begin) // STEP)
            last = first + 1  # Unknown/zero duration: start-bin evidence only.
        else:
            continue
        changes[first][code] += 1
        changes[last][code] -= 1
    active = Counter()
    active_text, triggered_text, count_text, event_text = [], [], [], []
    totals, flags = [], []
    for i in range(len(grid)):
        active.update(changes.get(i, {}))
        active_codes = sorted(c for c, n in active.items() if n > 0)
        counts = dict(sorted(starts.get(i, {}).items()))
        active_text.append("; ".join(active_codes))
        triggered_text.append("; ".join(counts))
        count_text.append(json.dumps(counts))
        totals.append(sum(counts.values()))
        flags.append(bool(active_codes))
        event_text.append(json.dumps(events_by_bin.get(i, [])))
    result["alarmOn"] = np.where(flags, "yes", "no")
    result["alarmOccurred"] = np.asarray(totals) > 0
    result["activeAlarmCodes"] = active_text
    result["triggeredAlarmCodes"] = triggered_text
    result["triggeredAlarmCounts"] = count_text
    result["alarmTriggerCount"] = np.asarray(totals, dtype="int32")
    # Original event timing retained inside JSON, never as extra timestamp rows.
    result["alarmEvents"] = event_text
    result.attrs["carry_in_alarm_events"] = [
        {
            "Alarmcode": str(e.Alarmcode).strip(),
            "TimeOn": e.TimeOn.isoformat(sep=" "),
            "TimeOff": e.TimeOff.isoformat(sep=" "),
        }
        for e in alarms.itertuples(index=False)
        if e.TimeOn < begin and pd.notna(e.TimeOff) and e.TimeOff > begin
    ]
    return add_selected_alarm_labels(result, selected_alarm_codes)


def preprocess_year(
    source_root,
    output_root,
    year,
    turbines=None,
    rebuild=False,
    selected_alarm_codes=DEFAULT_SELECTED_ALARM_CODES,
):
    source = year_folder(source_root, year)
    ids = (
        [p.name.removeprefix("Station_") for p in source.glob("Station_*") if p.is_dir()]
        if turbines is None
        else [str(t) for t in turbines]
    )
    if not ids:
        raise ValueError(f"No turbines found in {source}")
    summary = []
    for turbine in sorted(ids):
        if not turbine.isdigit():
            raise ValueError("Turbine IDs must contain digits only.")
        station = source / f"Station_{turbine}"
        sensors_path, alarms_path = station / "sensors.parquet", station / "alarms.parquet"
        # Include earlier-year alarm files to account for carry-in events if available.
        alarm_paths = []
        for directory in Path(source_root).iterdir():
            text = directory.name.removeprefix("turbines_")
            if text.isdigit() and int(text) <= year:
                candidate = year_folder(source_root, int(text)) / station.name / "alarms.parquet"
                if candidate.is_file() and candidate not in alarm_paths:
                    alarm_paths.append(candidate)
        if not sensors_path.is_file() or not alarms_path.is_file():
            raise FileNotFoundError(
                f"Missing sensor/alarm Parquet for {year}/{station.name}. Run notebook 02."
            )
        inputs = [sensors_path, *sorted(alarm_paths)]
        signature = {
            "schema": SCHEMA_VERSION,
            "year": year,
            "turbine": turbine,
            "selected_alarm_codes": sorted({str(c).strip() for c in selected_alarm_codes}),
            "inputs": [
                {
                    "path": str(p.resolve()),
                    "size": p.stat().st_size,
                    "mtime_ns": p.stat().st_mtime_ns,
                }
                for p in inputs
            ],
        }
        destination = Path(output_root) / str(year) / station.name
        target, manifest = destination / "data.parquet", destination / "source.json"
        try:
            current = json.loads(manifest.read_text()) == signature
        except (FileNotFoundError, ValueError):
            current = False
        if not rebuild and current and target.is_file():
            summary.append(
                {"Year": year, "Turbine": turbine, "Status": "unchanged", "Path": str(target)}
            )
            continue
        sensor_data = pd.read_parquet(sensors_path)
        alarm_parts = []
        begin, finish = pd.Timestamp(year, 1, 1), pd.Timestamp(year + 1, 1, 1)
        for path in sorted(alarm_paths):
            part = pd.read_parquet(path)
            on, off = pd.to_datetime(part.TimeOn), pd.to_datetime(part.TimeOff)
            # Keep starts this year and carry-in intervals crossing January 1.
            part = part.loc[((on >= begin) & (on < finish)) | ((on < begin) & (off > begin))]
            alarm_parts.append(part)
        events = pd.concat(alarm_parts, ignore_index=True)
        frame = align_year(sensor_data, events, year, turbine, selected_alarm_codes)
        # Detect changed inputs before publishing a paired data/manifest update.
        for entry in signature["inputs"]:
            st = Path(entry["path"]).stat()
            if st.st_size != entry["size"] or st.st_mtime_ns != entry["mtime_ns"]:
                raise RuntimeError("Source files changed during preprocessing; rerun afterwards.")
        destination.mkdir(parents=True, exist_ok=True)
        temporary = destination / f".{uuid4().hex}.parquet"
        try:
            frame.to_parquet(temporary, engine="pyarrow", compression="snappy")
            os.replace(temporary, target)
            manifest.write_text(json.dumps(signature, indent=2), encoding="utf-8")
        finally:
            temporary.unlink(missing_ok=True)
        summary.append(
            {
                "Year": year,
                "Turbine": turbine,
                "Status": "created",
                "Rows": len(frame),
                "Triggers": int(frame.alarmTriggerCount.sum()),
                "Path": str(target),
            }
        )
        print(year, turbine, len(frame), "rows", flush=True)
    return summary


def load_aligned(root, year, turbine, bin_alarm_times=False):
    path = Path(root) / str(year) / f"Station_{turbine}" / "data.parquet"
    if not path.is_file():
        raise FileNotFoundError(
            f"Run 03_preprocessing/01_preprocess_10min_parquets.ipynb first. Missing: {path}"
        )
    frame = pd.read_parquet(path)
    if (
        not isinstance(frame.index, pd.DatetimeIndex)
        or (frame.index != frame.index.floor("10min")).any()
    ):
        raise ValueError("Invalid 10-minute data; rerun notebook 03.")
    events = list(frame.attrs.get("carry_in_alarm_events", []))
    for payload in frame.loc[frame.alarmOccurred, "alarmEvents"]:
        events.extend(json.loads(payload))
    alarms = pd.DataFrame(events, columns=["Alarmcode", "TimeOn", "TimeOff"])
    for c in ["TimeOn", "TimeOff"]:
        alarms[c] = pd.to_datetime(alarms[c])
    if bin_alarm_times:
        alarms["TimeOn"] = alarms.TimeOn.dt.floor("10min")
        alarms["TimeOff"] = alarms.TimeOff.dt.ceil("10min")
    return frame, alarms
