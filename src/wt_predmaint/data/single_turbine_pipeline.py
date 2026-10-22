"""Chunked, station-isolated raw-data preparation for a multi-year experiment."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pandas as pd

from wt_predmaint.data.alarms import alarm_code_table, build_alarm_events, parse_times
from wt_predmaint.data.clean import clean_year, dedupe_keys
from wt_predmaint.data.daily import DAILY_COLUMNS, add_daily_flags
from wt_predmaint.data.scada import TABLES, key_columns
from wt_predmaint.features.daily import assemble_features, daily_aggregates
from wt_predmaint.labels.episodes import fault_episodes_a, stop_episodes_b
from wt_predmaint.labels.horizon import build_horizon_labels, summarise_labels

PREPARATION_VERSION = 1


def experiment_config(cfg: dict, start: int, train_end: int, calibration: int, end: int) -> dict:
    """Local split override: never modify the shared protocol or caller's dictionary."""
    if not 1900 <= start <= train_end < calibration < end <= 9998:
        raise ValueError("Require start <= train-end < calibration-year < end")
    if calibration != train_end + 1:
        raise ValueError("Calibration year must immediately follow the last training year")
    out = copy.deepcopy(cfg)
    out["splits"] = {"train_years": list(range(start, train_end + 1)),
                     "val_years": [calibration], "test_years": list(range(calibration + 1, end + 1))}
    out["cleaning"]["expected_turbines"] = 1
    return out


def read_station(path: Path, station: int, kind: str, chunksize: int = 100_000) -> pd.DataFrame:
    """Filter a CSV in chunks before timestamp parsing and concatenation."""
    header = pd.read_csv(path, nrows=0).columns.tolist()
    id_column = "StationNr" if kind == "alarm" else "StationId"
    if kind == "scada":
        columns = key_columns(header)
    elif kind == "daily":
        columns = ["StationId", "TimeStamp", *[c for c in DAILY_COLUMNS if c in header]]
    else:
        columns = ["StationNr", "TimeOn", "TimeOff", "Alarmcode"]
    pieces = []
    for chunk in pd.read_csv(path, usecols=columns, chunksize=chunksize, low_memory=False):
        selected = chunk.loc[pd.to_numeric(chunk[id_column], errors="raise") == station].copy()
        if selected.empty:
            continue
        if kind == "alarm":
            for field in ("TimeOn", "TimeOff"):
                selected[field] = parse_times(selected[field].astype("string"), path.name)
            selected = selected.rename(columns={"StationNr": "StationId"})
            selected["Alarmcode"] = pd.to_numeric(selected["Alarmcode"], errors="raise").astype(int)
        else:
            selected["TimeStamp"] = parse_times(selected["TimeStamp"].astype("string"), path.name)
            if kind == "scada":
                values = [c for c in columns if c not in ("StationId", "TimeStamp")]
                selected[values] = selected[values].astype("float32")
        selected["StationId"] = station
        pieces.append(selected)
    if pieces:
        return pd.concat(pieces, ignore_index=True)
    # Preserve datetime schema for an empty station selection.
    result = pd.DataFrame(columns=["StationId" if c == "StationNr" else c for c in columns])
    for c in ("TimeStamp", "TimeOn", "TimeOff"):
        if c in result:
            result[c] = pd.to_datetime(result[c])
    return result


def prepare(raw: Path, output: Path, station: int, cfg: dict, rebuild: bool = False,
            targets: tuple[str, ...] = ("a",)) -> None:
    """Prepare only this station, reusing per-year caches when source metadata matches."""
    if not targets or set(targets) - {"a", "b"}:
        raise ValueError("Choose targets a and/or b")
    years = sorted(sum(cfg["splits"].values(), []))
    tables = [*TABLES, "tblDailySummary", "tblAlarmLog"]
    inventory = {(year, table): sorted((raw / str(year)).rglob(f"{table}_{year}_*.csv"))
                 for year in years for table in tables}
    missing = [f"{year}/{table}" for (year, table), paths in inventory.items() if not paths]
    if missing:
        raise FileNotFoundError("Missing extracted CSV tables: " + ", ".join(missing))
    output.mkdir(parents=True, exist_ok=True)
    aggregates, daily_frames, event_frames, stop_frames, reports = [], [], [], [], {}
    for year in years:
        cache = output / "cache" / str(year)
        paths = [p for table in tables for p in inventory[year, table]]
        signature = {"version": PREPARATION_VERSION, "station": station,
                     "cleaning": cfg["cleaning"], "features": cfg["features"],
                     "labels": cfg["labels"],
                     "sources": [{"path": str(p.resolve()), "size": p.stat().st_size,
                                  "mtime_ns": p.stat().st_mtime_ns} for p in paths]}
        products = ("daily_aggregates", "daily_summary", "alarm_events", "stop_intervals")
        manifest = cache / "source.json"
        cached = (not rebuild and manifest.exists()
                  and all((cache / f"{name}.parquet").exists() for name in products)
                  and json.loads(manifest.read_text(encoding="utf-8"))["signature"] == signature)
        if cached:
            print(f"{year}: reuse verified station cache", flush=True)
            frames = [pd.read_parquet(cache / f"{name}.parquet") for name in products]
            stats = json.loads(manifest.read_text(encoding="utf-8"))["statistics"]
        else:
            print(f"{year}: read raw CSVs for turbine {station}", flush=True)
            sensor_tables = {}
            for table in TABLES:
                print(f"  {table}", flush=True)
                sensor_tables[table] = pd.concat(
                    [read_station(p, station, "scada") for p in inventory[year, table]],
                    ignore_index=True)
                if sensor_tables[table].empty:
                    raise ValueError(f"{year}/{table}: no rows for turbine {station}")
            cleaned, stats = clean_year(sensor_tables, year, cfg)
            del sensor_tables
            if cleaned.empty:
                raise ValueError(f"{year}: no joined sensor rows for turbine {station}")
            daily = pd.concat([read_station(p, station, "daily")
                               for p in inventory[year, "tblDailySummary"]], ignore_index=True)
            daily, exact, conflict = dedupe_keys(daily)
            daily = add_daily_flags(daily, cfg)
            daily = daily.loc[daily["TimeStamp"].dt.year == year].copy()
            if daily.empty:
                raise ValueError(f"{year}: no daily summary for turbine {station}")
            stats["daily_duplicates"] = {"exact": exact, "conflicting": conflict}
            alarms = pd.concat([read_station(p, station, "alarm")
                                for p in inventory[year, "tblAlarmLog"]], ignore_index=True)
            events = build_alarm_events(alarms.drop_duplicates(), alarm_code_table(cfg))
            frames = [daily_aggregates(cleaned, cfg), daily, events,
                      cleaned[["StationId", "TimeStamp", "wtc_GenRpm_mean",
                               "wtc_PrWindSp_mean", "outage_month"]]]
            cache.mkdir(parents=True, exist_ok=True)
            for name, frame in zip(products, frames, strict=True):
                frame.to_parquet(cache / f"{name}.parquet", index=False)
            manifest.write_text(json.dumps({"signature": signature, "statistics": stats}, indent=2),
                                encoding="utf-8")
        aggregates.append(frames[0])
        daily_frames.append(frames[1])
        event_frames.append(frames[2])
        stop_frames.append(frames[3])
        reports[str(year)] = stats
    daily = pd.concat(daily_frames, ignore_index=True).sort_values(["StationId", "TimeStamp"])
    events = pd.concat(event_frames, ignore_index=True).drop_duplicates()
    stop = pd.concat(stop_frames, ignore_index=True)
    features = assemble_features(pd.concat(aggregates, ignore_index=True), events, daily, cfg)
    features.to_parquet(output / "features.parquet", index=False)
    summaries = []
    builders = {"a": lambda: fault_episodes_a(daily, cfg),
                "b": lambda: stop_episodes_b(stop, cfg)}
    for target in targets:
        episodes, days = builders[target]()
        labels = build_horizon_labels(days, episodes, cfg)
        episodes.to_parquet(output / f"{target}_episodes.parquet", index=False)
        labels.to_parquet(output / f"{target}_labels.parquet", index=False)
        summaries.append(summarise_labels(labels, episodes, cfg).assign(target=target.upper()))
    pd.concat(summaries, ignore_index=True).to_csv(output / "label_summary.csv", index=False)
    (output / "preparation.json").write_text(json.dumps({
        "station": station, "protocol": cfg, "years": years, "targets": list(targets), "cleaning": reports,
        "feature_rows": len(features), "raw": str(raw.resolve()),
    }, indent=2), encoding="utf-8")
