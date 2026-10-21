"""Build the in-scope alarm event table from the raw tblAlarmLog files.

Alarms must come from the raw tblAlarmLog, not from the Alarmcode/AlarmEvents columns of the
processed turbine workbooks: those keep only ~0.2% of events (see docs/DATA_TRAPS.md).
"""

from __future__ import annotations

import io
import zipfile
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import pandas as pd

RAW_COLUMNS = ("TimeOn", "TimeOff", "StationNr", "Alarmcode")
SITE_STATION = 91  # site-level station, not a turbine
INTERVAL = "10min"


def alarm_code_table(cfg: dict[str, Any]) -> pd.DataFrame:
    """In-scope codes with description, stopping flag and tier, from protocol.yaml."""
    table = pd.DataFrame(cfg["labels"]["alarm_codes"]).rename(
        columns={
            "code": "Alarmcode",
            "description": "Description",
            "tier": "Tier",
            "stopping": "Stopping",
        }
    )
    table["Alarmcode"] = table["Alarmcode"].astype(int)
    return table[["Alarmcode", "Description", "Stopping", "Tier"]]


def iter_alarm_log_files(root: Path) -> Iterator[tuple[str, bytes]]:
    """Yield (name, bytes) for every tblAlarmLog CSV under root, including inside zips."""
    for path in sorted(root.rglob("*")):
        suffix = path.suffix.lower()
        if suffix == ".csv" and "tblalarmlog" in path.name.lower():
            yield path.name, path.read_bytes()
        elif suffix == ".zip":
            with zipfile.ZipFile(path) as zf:
                for member in sorted(zf.namelist()):
                    low = member.lower()
                    if low.endswith(".csv") and "tblalarmlog" in low:
                        yield f"{path.name}:{member}", zf.read(member)


def parse_times(values: pd.Series, name: str) -> pd.Series:
    """Parse one file's timestamps: ISO (date-only allowed) or month-first 'M/D/YYYY[ H:MM]'.

    The month-first layout was verified on the re-saved 2025-01 and 2026-01 files (day values
    above 12 only occur in the second field), including the date-only daily-summary file.
    Anything else raises rather than being guessed.
    """
    for fmt in ("ISO8601", "%m/%d/%Y %H:%M", "%m/%d/%Y"):
        try:
            return pd.to_datetime(values, format=fmt)
        except ValueError:
            continue
    raise ValueError(f"{name}: unrecognised timestamp format, e.g. {values.dropna().iloc[0]!r}")


def read_alarm_log(name: str, data: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(data), dtype={"TimeOn": str, "TimeOff": str})
    df.columns = [c.strip() for c in df.columns]
    missing = set(RAW_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"{name}: missing columns {sorted(missing)}; found {list(df.columns)}")
    df = df[list(RAW_COLUMNS)].copy()
    df["TimeOn"] = parse_times(df["TimeOn"], name)
    df["TimeOff"] = parse_times(df["TimeOff"], name) if df["TimeOff"].notna().any() else pd.NaT
    return df


def load_alarm_log(sources: Iterable[tuple[str, bytes]]) -> pd.DataFrame:
    """Concatenate raw alarm logs; drop the site-level station and exact duplicates."""
    frames = [read_alarm_log(name, data) for name, data in sources]
    if not frames:
        raise FileNotFoundError("no tblAlarmLog CSV files found")
    raw = pd.concat(frames, ignore_index=True)
    raw["TimeOff"] = pd.to_datetime(raw["TimeOff"])
    raw["Alarmcode"] = pd.to_numeric(raw["Alarmcode"], errors="raise").astype(int)
    raw = raw.rename(columns={"StationNr": "StationId"})
    raw["StationId"] = raw["StationId"].astype(int)
    return raw[raw["StationId"] != SITE_STATION].drop_duplicates().reset_index(drop=True)


def build_alarm_events(raw: pd.DataFrame, codes: pd.DataFrame) -> pd.DataFrame:
    """Keep in-scope codes; add Tier, Stopping, the 10-min interval and the duration.

    SCADA TimeStamp marks the END of its 10-min interval (wtc_CurTime_endvalue sits 1-2 s before
    it), so an event belongs to the first interval end at or after TimeOn: ceil, not floor.
    """
    events = raw.merge(codes, on="Alarmcode", how="inner")
    events["TimeStamp"] = events["TimeOn"].dt.ceil(INTERVAL)
    events["DurationMin"] = (events["TimeOff"] - events["TimeOn"]).dt.total_seconds() / 60
    cols = [
        "StationId",
        "Alarmcode",
        "Tier",
        "Stopping",
        "TimeOn",
        "TimeOff",
        "TimeStamp",
        "DurationMin",
    ]
    return events[cols].sort_values(["StationId", "TimeOn", "Alarmcode"]).reset_index(drop=True)


def other_alarm_events(raw: pd.DataFrame, codes: pd.DataFrame) -> pd.DataFrame:
    """Out-of-scope events, kept only to mask rows when fitting normal-behaviour models."""
    other = raw[~raw["Alarmcode"].isin(codes["Alarmcode"])]
    return other.sort_values(["StationId", "TimeOn"]).reset_index(drop=True)


def count_summary(events: pd.DataFrame) -> pd.DataFrame:
    """Events per tier, code and year, with turbine count; basis for feasibility decisions."""
    year = events["TimeOn"].dt.year.rename("Year")
    per_year = events.groupby(["Tier", "Alarmcode", year]).size().unstack("Year", fill_value=0)
    per_year["Total"] = per_year.sum(axis=1)
    turbines = events.groupby(["Tier", "Alarmcode"])["StationId"].nunique().rename("Turbines")
    return per_year.join(turbines)


def group_episodes(events: pd.DataFrame, gap: pd.Timedelta) -> pd.DataFrame:
    """Merge events of the same turbine and code into episodes.

    A new episode starts when an event begins more than `gap` after the previous episode
    ended (TimeOff if present, else TimeOn). Returns one row per episode.
    """
    ev = events.sort_values(["StationId", "Alarmcode", "TimeOn"]).reset_index(drop=True)
    keys = [ev["StationId"], ev["Alarmcode"]]
    ev["EndTime"] = ev["TimeOff"].fillna(ev["TimeOn"])
    prev_end = ev["EndTime"].groupby(keys, sort=False).cummax().groupby(keys, sort=False).shift()
    new_episode = prev_end.isna() | ((ev["TimeOn"] - prev_end) > gap)
    ev["Episode"] = new_episode.groupby(keys, sort=False).cumsum()
    out = (
        ev.groupby(["StationId", "Alarmcode", "Episode"], sort=False)
        .agg(
            Tier=("Tier", "first"),
            Stopping=("Stopping", "first"),
            Start=("TimeOn", "min"),
            End=("EndTime", "max"),
            Events=("TimeOn", "size"),
        )
        .reset_index()
    )
    return out.sort_values(["StationId", "Start"]).reset_index(drop=True)
