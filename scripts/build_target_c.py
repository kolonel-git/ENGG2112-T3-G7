"""Build experimental Target C independently of the frozen A/B daily pipeline.

Requires cleaned scada_<year>.parquet, raw tblAlarmLog files and a verified coverage CSV.
See docs/TARGET_C.md. Processes one SCADA year at a time with seven days of prior context.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from wt_predmaint.config import load_protocol
from wt_predmaint.data.alarms import iter_alarm_log_files, load_alarm_log
from wt_predmaint.features.short_term import WINDOWS, short_term_features
from wt_predmaint.labels.alarm_forecast import alarm_labels, assign_splits


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--interim", type=Path, default=ROOT / "data/interim")
    p.add_argument("--raw", type=Path, default=ROOT / "data/raw")
    p.add_argument("--coverage", type=Path, required=True)
    p.add_argument("--clean", type=Path, help="read cleaned SCADA from another local directory")
    p.add_argument("--out", type=Path, help="output partitions without changing source data")
    p.add_argument("--events", type=Path, help="pre-audited raw alarm events parquet")
    p.add_argument("--years", type=int, nargs="+", help="SCADA years to build")
    p.add_argument("--prediction-minutes", type=int, choices=[10, 20, 30, 60], default=10)
    args = p.parse_args()
    coverage = pd.read_csv(args.coverage, parse_dates=["Start", "End"])
    events = (
        pd.read_parquet(args.events)
        if args.events
        else load_alarm_log(iter_alarm_log_files(args.raw))
    )
    paths = sorted((args.clean or args.interim / "clean").glob("scada_*.parquet"))
    if args.years:
        paths = [path for path in paths if int(path.stem.split("_")[-1]) in args.years]
    if not paths:
        p.error("no cleaned SCADA files found; run build_clean_data.py first")
    out = args.out or args.interim / "target_c"
    out.mkdir(parents=True, exist_ok=True)
    tail = pd.DataFrame()
    for path in paths:
        current = pd.read_parquet(path)
        current = current[
            [c for c in current if c in {"StationId", "TimeStamp"} or c.endswith(("_mean", "_max"))]
        ]
        if current.empty:
            continue
        # Carry only context from earlier files, never future sensor rows.
        context = pd.concat([tail, current], ignore_index=True)
        for station, g in current.groupby("StationId"):
            history = context.loc[context.StationId == station]
            features = short_term_features(history, events, args.prediction_minutes)
            features = features.loc[features.TimeStamp.isin(g.TimeStamp)]
            # Missing past alarm logs must not masquerade as a quiet history.
            for name, window in WINDOWS.items():
                past = features[["StationId", "TimeStamp"]].copy()
                past["TimeStamp"] -= window
                observed = alarm_labels(past, events.iloc[:0], coverage, window)
                missing = ~observed.horizon_observed.to_numpy()
                features.loc[missing, f"{name}_alarm_count"] = float("nan")
                if name == "7d":
                    features.loc[missing, "minutes_since_alarm"] = float("nan")
            labels = assign_splits(
                alarm_labels(features, events, coverage), load_protocol()["splits"]
            )
            table = features.merge(labels, on=["StationId", "TimeStamp"], validate="one_to_one")
            floats = table.select_dtypes("floating").columns
            table[floats] = table[floats].astype("float32")
            table.to_parquet(out / f"{path.stem}_{station}.parquet", index=False)
            print(f"{path.stem}, station {station}: {len(table):,} prediction points", flush=True)
        cutoff = current.TimeStamp.max() - pd.Timedelta(days=7)
        tail = context.loc[context.TimeStamp > cutoff].copy()


if __name__ == "__main__":
    main()
