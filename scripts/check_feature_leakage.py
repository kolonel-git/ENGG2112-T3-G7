"""Run the look-ahead test on real data: remove the future from every source and recompute.

Picks random turbine-days (seeded from the config) and checks that each feature row is unchanged
when everything after the end of its day is deleted. Exits with status 1 on any difference.

    python scripts/check_feature_leakage.py --stations 4 --days 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Support running this file directly from an IDE without installing the package.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import numpy as np
import pandas as pd

from wt_predmaint.config import load_protocol
from wt_predmaint.features.leakage import check_no_lookahead




def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--interim", type=Path, default=ROOT / "data" / "interim")
    p.add_argument("--stations", type=int, default=4)
    p.add_argument("--days", type=int, default=5, help="random days per station")
    args = p.parse_args()
    cfg = load_protocol()
    rng = np.random.default_rng(int(cfg["seed"]))
    clean = args.interim / "clean"

    table = pd.read_parquet(args.interim / "features" / "daily_features.parquet")
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

    problems = check_no_lookahead(scada, events, summary, cfg, cases)
    print(f"checked {len(cases)} turbine-days on turbines {sorted(int(s) for s in stations)}")
    if problems:
        for item in problems:
            print(f"LOOK-AHEAD: {item['StationId']} {item['Day'].date()} {item['columns'][:8]}")
        sys.exit(1)
    print("no look-ahead found: every row is unchanged when the future is removed")


if __name__ == "__main__":
    main()
