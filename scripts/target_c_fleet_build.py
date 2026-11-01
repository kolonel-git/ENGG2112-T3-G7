"""Target C (fleet), step 1: raw CSVs -> cleaned 10-min parquet -> hourly sensor features (D-071).

Reads the raw tblSCTurbine, tblSCTurTemp, tblSCTurPress and tblAlarmLog CSVs under a raw folder
(any nesting, year folders such as `2017(1)/2017/` are fine; zips are ignored), cleans them as in
D-013 to D-016 (de-duplicate on turbine and timestamp, mask the per-table sentinels) and writes:

  <out>/alarms.parquet                       raw alarm events (site station 91 and exact duplicates dropped)
  <out>/scada10/<table>_<YYYY>_<MM>.parquet  cleaned 10-min mean columns
  <out>/features/<StationId>.parquet         hourly anchors with trailing 1-h and 24-h statistics

SCADA timestamps are interval ends, so a feature at anchor t uses intervals ending at or before t.

    python scripts/target_c_fleet_build.py --raw data/raw --out data/interim/target_c_fleet --step all
"""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq
import yaml

SCADA_TABLES = ("tblSCTurbine", "tblSCTurTemp", "tblSCTurPress")
NAME_RE = re.compile(r"^(tbl[A-Za-z]+)_(\d{4})_(\d{2})\.csv$", re.IGNORECASE)
SITE_STATION = 91


def load_cfg(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def find_raw_files(raw: Path) -> dict[tuple[str, int, int], Path]:
    """Map (table, year, month) to one CSV; the first path in sorted order wins."""
    found: dict[tuple[str, int, int], Path] = {}
    for path in sorted(raw.rglob("*.csv")):
        m = NAME_RE.match(path.name)
        if m is None:
            continue
        key = (m.group(1), int(m.group(2)), int(m.group(3)))
        found.setdefault(key, path)
    return found


def parse_times(values: pd.Series, name: str) -> pd.Series:
    """ISO timestamps, or the month-first layout of the re-saved 2025-01 and 2026-01 files."""
    for fmt in ("ISO8601", "%m/%d/%Y %H:%M", "%m/%d/%Y"):
        try:
            return pd.to_datetime(values, format=fmt)
        except ValueError:
            continue
    raise ValueError(f"{name}: unrecognised timestamp format, e.g. {values.dropna().iloc[0]!r}")


def wanted_column(col: str, cfg: dict[str, Any]) -> bool:
    tc = cfg["target_c_fleet"]
    if col in tc["extra_columns"]:
        return True
    return col.startswith("wtc_") and any(col.endswith(s) for s in tc["sensor_suffixes"])


def clean_scada_file(path: Path, table: str, cfg: dict[str, Any]) -> pd.DataFrame:
    header = path.open(encoding="utf-8-sig").readline().strip().split(",")
    header = [h.strip() for h in header]
    keep = ["TimeStamp", "StationId"] + [c for c in header if wanted_column(c, cfg)]
    types = {c: pa.float32() for c in keep[2:]}
    types.update({"TimeStamp": pa.string(), "StationId": pa.float64()})
    tbl = pacsv.read_csv(
        path,
        read_options=pacsv.ReadOptions(block_size=1 << 26),
        convert_options=pacsv.ConvertOptions(include_columns=keep, column_types=types),
    )
    df = tbl.to_pandas()
    df = df.dropna(subset=["TimeStamp", "StationId"])
    df["TimeStamp"] = parse_times(df["TimeStamp"], path.name)
    df = df[df["StationId"] != SITE_STATION]
    df = df.drop_duplicates(subset=["StationId", "TimeStamp"], keep="last")
    sentinels = [float(v) for v in cfg["cleaning"]["mask_sentinels"].get(table, [])]
    cols = keep[2:]
    if sentinels and cols:
        vals = df[cols].to_numpy()
        vals[np.isin(vals, np.array(sentinels, dtype=np.float32))] = np.nan
        df[cols] = vals
    df["StationId"] = df["StationId"].astype("int32")
    return df


def step_scada(files: dict[tuple[str, int, int], Path], out: Path, cfg: dict[str, Any]) -> None:
    dest = out / "scada10"
    dest.mkdir(parents=True, exist_ok=True)
    for (table, year, month), path in sorted(files.items()):
        if table not in SCADA_TABLES:
            continue
        target = dest / f"{table}_{year}_{month:02d}.parquet"
        if target.exists():
            continue
        began = time.perf_counter()
        df = clean_scada_file(path, table, cfg)
        tmp = dest / (target.name + ".tmp")
        df.to_parquet(tmp, index=False)
        tmp.rename(target)
        print(f"{target.name}: {len(df):,} rows, {df.shape[1] - 2} cols, "
              f"{time.perf_counter() - began:.1f}s", flush=True)


def step_alarms(files: dict[tuple[str, int, int], Path], out: Path) -> None:
    frames = []
    for (table, year, month), path in sorted(files.items()):
        if table.lower() != "tblalarmlog":
            continue
        df = pd.read_csv(path, dtype={"TimeOn": str, "TimeOff": str})
        df.columns = [c.strip() for c in df.columns]
        df = df[["TimeOn", "TimeOff", "StationNr", "Alarmcode"]].copy()
        df = df.dropna(subset=["TimeOn", "StationNr", "Alarmcode"])
        df["TimeOn"] = parse_times(df["TimeOn"], path.name)
        off = df["TimeOff"]
        parsed = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
        if off.notna().any():
            parsed.loc[off.notna()] = parse_times(off[off.notna()], path.name)
        df["TimeOff"] = parsed
        df["source"] = f"{year}_{month:02d}"
        frames.append(df)
    alarms = pd.concat(frames, ignore_index=True)
    alarms = alarms[alarms["StationNr"] != SITE_STATION]
    alarms = alarms.drop_duplicates(subset=["TimeOn", "TimeOff", "StationNr", "Alarmcode"])
    alarms["StationNr"] = alarms["StationNr"].astype("int32")
    alarms["Alarmcode"] = alarms["Alarmcode"].astype("int32")
    out.mkdir(parents=True, exist_ok=True)
    alarms.sort_values(["StationNr", "TimeOn"]).to_parquet(out / "alarms.parquet", index=False)
    print(f"alarms.parquet: {len(alarms):,} events from {len(frames)} files", flush=True)


def turbine_10min(out: Path, station: int) -> pd.DataFrame:
    """All years of one turbine, the three tables outer-joined on the 10-min interval end."""
    parts: dict[str, list[pd.DataFrame]] = {t: [] for t in SCADA_TABLES}
    for path in sorted((out / "scada10").glob("*.parquet")):
        table = path.name.split("_")[0]
        df = pq.read_table(path, filters=[("StationId", "=", station)]).to_pandas()
        if len(df):
            parts[table].append(df.drop(columns="StationId").set_index("TimeStamp"))
    joined: pd.DataFrame | None = None
    for table in SCADA_TABLES:
        if not parts[table]:
            continue
        frame = pd.concat(parts[table]).sort_index()
        frame = frame[~frame.index.duplicated(keep="last")]
        frame[f"present_{table}"] = np.float32(1.0)
        joined = frame if joined is None else joined.join(frame, how="outer")
    if joined is None:
        raise ValueError(f"no SCADA rows for turbine {station}")
    return joined


def hourly_features(df: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    tc = cfg["target_c_fleet"]
    grid = pd.date_range(df.index.min().ceil("h"), df.index.max().floor("h"), freq="10min")
    df = df.reindex(grid)
    present = df[[c for c in df if c.startswith("present_")]].notna().any(axis=1).astype("float32")
    sensors = [c for c in df if c.startswith("wtc_")]
    x = df[sensors].astype("float32")
    anchors = grid[grid.minute == 0]
    w_short, w_long = (int(h * 6) for h in tc["feature_windows_hours"])
    n_obs = present.rolling(w_short, min_periods=1).sum().reindex(anchors)
    mean_s = x.rolling(w_short, min_periods=tc["min_anchor_intervals"]).mean().reindex(anchors)
    roll_l = x.rolling(w_long, min_periods=w_long // 4)
    mean_l = roll_l.mean().reindex(anchors)
    std_l = roll_l.std().reindex(anchors)
    feats = pd.concat(
        [n_obs.rename("n_obs_1h"), mean_s.add_suffix("__1h"), mean_l.add_suffix("__24h"),
         std_l.add_suffix("__24h_std")],
        axis=1,
    )
    feats = feats[feats["n_obs_1h"] >= tc["min_anchor_intervals"]].astype("float32")
    feats.index.name = "anchor"
    return feats


def step_features(out: Path, cfg: dict[str, Any]) -> None:
    dest = out / "features"
    dest.mkdir(parents=True, exist_ok=True)
    first = sorted((out / "scada10").glob("tblSCTurbine_*.parquet"))
    stations = sorted(set(pq.read_table(first[-1], columns=["StationId"]).column(0).to_pylist()))
    for station in stations:
        target = dest / f"{station}.parquet"
        if target.exists():
            continue
        began = time.perf_counter()
        feats = hourly_features(turbine_10min(out, station), cfg)
        tmp = dest / (target.name + ".tmp")
        feats.reset_index().to_parquet(tmp, index=False)
        tmp.rename(target)
        print(f"{station}: {len(feats):,} anchors x {feats.shape[1]} cols, "
              f"{time.perf_counter() - began:.1f}s", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", type=Path, default=Path("data/raw"))
    ap.add_argument("--out", type=Path, default=Path("data/interim/target_c_fleet"))
    ap.add_argument("--config", type=Path, default=Path("configs/target_c_fleet.yaml"))
    ap.add_argument("--step", choices=["alarms", "scada", "features", "all"], default="all")
    args = ap.parse_args()
    cfg = load_cfg(args.config)
    files = find_raw_files(args.raw)
    print(f"{len(files)} raw CSV files found", flush=True)
    if args.step in ("alarms", "all"):
        step_alarms(files, args.out)
    if args.step in ("scada", "all"):
        step_scada(files, args.out, cfg)
    if args.step in ("features", "all"):
        step_features(args.out, cfg)


if __name__ == "__main__":
    main()
