"""Inventory monthly raw exports and prepare explicitly provisional coverage for Target C.

Calendar export presence is an assumption of complete logging, NOT independent certification.
No coverage interval is inferred from the first/last event. Only training/validation years
and prior alarm history are read; test years remain untouched.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from wt_predmaint.data.alarms import load_alarm_log


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    inventory, frames, intervals = [], [], []
    for year in range(2016, 2023):
        for month in range(1, 13):
            path = args.raw / str(year) / f"tblAlarmLog_{year}_{month:02d}.csv"
            if not path.exists():
                inventory.append({"year": year, "month": month, "present": False})
                continue
            frame = load_alarm_log([(path.name, path.read_bytes())])
            start = pd.Timestamp(year=year, month=month, day=1)
            end = start + pd.offsets.MonthBegin(1)
            inside = frame.loc[(frame.TimeOn >= start) & (frame.TimeOn < end)]
            stations = sorted(inside.StationId.unique())
            inventory.append(
                {
                    "year": year,
                    "month": month,
                    "present": True,
                    "rows": len(frame),
                    "stations": len(stations),
                    "first_alarm": str(frame.TimeOn.min()),
                    "last_alarm": str(frame.TimeOn.max()),
                    "fleet_days_with_alarm": inside.TimeOn.dt.normalize().nunique(),
                    "calendar_days": start.days_in_month,
                    "path": str(path.resolve()),
                }
            )
            # Use only stations actually represented in this export. Completely absent
            # stations are unknown, rather than treated as quiet turbines.
            active_days = set(inside.TimeOn.dt.normalize())
            for day in pd.date_range(start, end, inclusive="left", freq="D"):
                if day in active_days:
                    intervals.extend(
                        {"StationId": station, "Start": day, "End": day + pd.Timedelta(days=1)}
                        for station in stations
                    )
            frames.append(frame)
            print(f"{path.name}: {len(frame):,} alarms, {len(stations)} turbines", flush=True)
    if not frames:
        raise ValueError("no monthly raw alarm files found")
    pd.DataFrame(inventory).to_csv(args.out / "source_inventory.csv", index=False)
    pd.DataFrame(intervals).to_csv(args.out / "provisional_alarm_coverage.csv", index=False)
    alarms = pd.concat(frames, ignore_index=True).drop_duplicates()
    alarms.to_parquet(args.out / "raw_alarm_events.parquet", index=False)
    note = {
        "status": "PROVISIONAL: source monthly exports assumed complete",
        "assumption": "Calendar-month exports record all starts for represented turbines. "
        "File presence and fleet activity are not independent proof of completeness.",
        "years": list(range(2016, 2023)),
        "raw_alarm_rows": len(alarms),
        "files_present": sum(row["present"] for row in inventory),
        "files_expected": 84,
        "test_years_read": False,
        "conservative_exclusion": "Exclude entire days with no fleet alarm records, and any "
        "future or historical window overlapping those days.",
    }
    (args.out / "coverage_assumption.json").write_text(json.dumps(note, indent=2), encoding="utf-8")
    print(json.dumps(note, indent=2))


if __name__ == "__main__":
    main()
