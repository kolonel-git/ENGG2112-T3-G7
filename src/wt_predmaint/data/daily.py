"""Loader for tblDailySummary (one row per turbine and day).

A row stamped D covers the day starting at D 00:00 UTC, i.e. [D, D + 1 day). Checked against the
10-min data: daily NacelleWind vs the mean of wtc_PrWindSp over those 144 intervals has
correlation 0.9996 (the end-labelled 10-min rows with TimeStamp in (D, D + 1 day]).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from wt_predmaint.data.alarms import SITE_STATION, parse_times
from wt_predmaint.data.clean import KEYS, dedupe_keys

DAILY_COLUMNS = (
    "Energy",
    "MeanPower",
    "CapacityFactor",
    "NacelleWind",
    "Availabillity",
    "TimeOk",
    "TimeReady",
    "TimeErr",
    "TimeExternalErr",
    "TimeGenerator",
    "Samples",
    "SamplePeriode",
)
HOUR_COLUMNS = ("TimeOk", "TimeReady", "TimeErr", "TimeExternalErr", "TimeGenerator")


def read_daily_file(path: Path) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0).columns.tolist()
    cols = ["StationId", "TimeStamp"] + [c for c in DAILY_COLUMNS if c in header]
    df = pd.read_csv(path, usecols=cols, low_memory=False)
    df["TimeStamp"] = parse_times(df["TimeStamp"].astype(str), path.name)
    return df


def add_daily_flags(df: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Flag impossible or incomplete days; rows are kept, never dropped."""
    d = cfg["cleaning"]["daily"]
    out = df.copy()
    out["partial_day"] = (out["Samples"] < d["min_samples"]) | (
        out["SamplePeriode"] != d["expected_period_minutes"]
    )
    out["bad_hours"] = (out[list(HOUR_COLUMNS)] > d["max_hours"]).any(axis=1)
    out["negative_energy"] = out["Energy"] < 0
    return out


def load_daily_summary(raw_dir: Path, cfg: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, int]]:
    files = sorted(raw_dir.glob("*/tblDailySummary_*.csv"))
    if not files:
        raise FileNotFoundError(f"no tblDailySummary files under {raw_dir}")
    raw = pd.concat([read_daily_file(f) for f in files], ignore_index=True)
    raw = raw[raw["StationId"].astype(int) != SITE_STATION]
    raw["StationId"] = raw["StationId"].astype(int)
    df, n_exact, n_conflict = dedupe_keys(raw)
    df = add_daily_flags(df, cfg).sort_values(KEYS).reset_index(drop=True)
    stats = {
        "rows": len(df),
        "exact_duplicates": n_exact,
        "conflicting_duplicates": n_conflict,
        "partial_day": int(df["partial_day"].sum()),
        "bad_hours": int(df["bad_hours"].sum()),
        "negative_energy": int(df["negative_energy"].sum()),
    }
    return df, stats
