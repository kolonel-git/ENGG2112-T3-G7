"""Run the look-ahead test on real data: remove the future from every source and recompute.

Picks random turbine-days (seeded from the config) and checks that each feature row is unchanged
when everything after the end of its day is deleted. Exits with status 1 on any difference.

    python scripts/check_feature_leakage.py --stations 4 --days 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from wt_predmaint.config import load_protocol
from wt_predmaint.features.leakage import check_no_lookahead

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--interim", type=Path, default=ROOT / "data" / "interim")
    p.add_argument("--stations", type=int, default=4)
    p.add_argument("--days", type=int, default=5, help="random days per station")
    p.add_argument("--b3", action="store_true", help="check B3 with a fixed train-fitted curve")
    p.add_argument("--nbm", action="store_true", help="check the NBM residual columns (D-054)")
    p.add_argument("--target", choices=["a", "b"], default="a")
    args = p.parse_args()
    cfg = load_protocol()
    rng = np.random.default_rng(int(cfg["seed"]))
    clean = args.interim / "clean"

    table = pd.read_parquet(args.interim / "features" / "daily_features.parquet")
    if args.nbm:
        from wt_predmaint.models.nbm import check_nbm_no_lookahead, fit_nbm

        labels = {t: pd.read_parquet(args.interim / f"labels/{t}_labels.parquet") for t in "ab"}
        model, _ = fit_nbm(table, labels, cfg)
        days = np.sort(table["Day"].unique())
        stations = rng.choice(np.sort(table["StationId"].unique()), args.stations, replace=False)
        picks = [
            (int(s), pd.Timestamp(d))
            for s in stations
            for d in rng.choice(days[len(days) // 10 :], args.days, replace=False)
        ]
        problems = check_nbm_no_lookahead(model, table, cfg, picks)
        print(f"checked {len(picks)} turbine-days on turbines {sorted(int(s) for s in stations)}")
        for item in problems:
            print(f"LOOK-AHEAD: {item['StationId']} {item['Day'].date()} {item['columns'][:8]}")
        if problems:
            sys.exit(1)
        print("no look-ahead found in the NBM columns")
        return
    stations = rng.choice(np.sort(table["StationId"].unique()), args.stations, replace=False)
    events = pd.read_parquet(args.interim / "alarm_events.parquet")
    summary = pd.read_parquet(clean / "daily_summary.parquet")

    cases: list[tuple[int, pd.Timestamp]] = []
    scada_parts = []
    for s in stations:
        days = table.loc[table["StationId"] == s, "Day"].to_numpy()
        days = days[len(days) // 10 :]  # leave room for the 7-day and 30-day look-backs
        for d in rng.choice(days, args.days, replace=False):
            cases.append((int(s), pd.Timestamp(d)))
        for f in sorted(clean.glob("scada_*.parquet")):
            scada_parts.append(pd.read_parquet(f, filters=[("StationId", "==", int(s))]))
    scada = pd.concat(scada_parts, ignore_index=True)

    if args.b3:
        from wt_predmaint.models.base import attach_labels, labelled
        from wt_predmaint.models.power_curve import PowerCurveProxy, check_b3_no_lookahead

        labels = pd.read_parquet(args.interim / f"labels/{args.target}_labels.parquet")
        train = labelled(attach_labels(table, labels, cfg), "train")
        model = PowerCurveProxy(args.target, interim=args.interim).fit(train, cfg)
        problems = check_b3_no_lookahead(model, scada, cfg, cases)
    else:
        problems = check_no_lookahead(scada, events, summary, cfg, cases)
    print(f"checked {len(cases)} turbine-days on turbines {sorted(int(s) for s in stations)}")
    if problems:
        for item in problems:
            print(f"LOOK-AHEAD: {item['StationId']} {item['Day'].date()} {item['columns'][:8]}")
        sys.exit(1)
    print("no look-ahead found: every row is unchanged when the future is removed")


if __name__ == "__main__":
    main()
