"""Build the O2 episodes and horizon labels for targets A and B.

Needs the cleaned tables from scripts/build_clean_data.py. Writes, under data/interim/labels/:
  a_episodes.parquet, a_labels.parquet   (target A: tblDailySummary TimeErr)
  b_episodes.parquet, b_labels.parquet   (target B: SCADA stop runs)
and reports/tables/label_summary.csv (rows, positives and episodes per target and split).

    python scripts/build_labels.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Support running this file directly from an IDE without installing the package.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import pandas as pd

from wt_predmaint.config import load_protocol
from wt_predmaint.labels.episodes import fault_episodes_a, stop_episodes_b
from wt_predmaint.labels.horizon import build_horizon_labels, summarise_labels

SCADA_COLUMNS = ["StationId", "TimeStamp", "wtc_GenRpm_mean", "wtc_PrWindSp_mean", "outage_month"]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--interim", type=Path, default=ROOT / "data" / "interim")
    args = p.parse_args()
    cfg = load_protocol()
    clean = args.interim / "clean"
    out = args.interim / "labels"
    out.mkdir(parents=True, exist_ok=True)

    daily = pd.read_parquet(clean / "daily_summary.parquet")
    scada = pd.concat(
        [pd.read_parquet(f, columns=SCADA_COLUMNS) for f in sorted(clean.glob("scada_*.parquet"))],
        ignore_index=True,
    )

    summaries = []
    for name, (episodes, days) in {
        "a": fault_episodes_a(daily, cfg),
        "b": stop_episodes_b(scada, cfg),
    }.items():
        labels = build_horizon_labels(days, episodes, cfg)
        episodes.to_parquet(out / f"{name}_episodes.parquet")
        labels.to_parquet(out / f"{name}_labels.parquet")
        summary = summarise_labels(labels, episodes, cfg)
        summary.insert(0, "target", name.upper())
        summaries.append(summary)
        print(f"target {name.upper()}: {len(episodes):,} episodes, {len(labels):,} candidate days")
        print(summary.drop(columns="target").to_string(index=False))

    table = pd.concat(summaries, ignore_index=True)
    tables_dir = ROOT / "reports" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(tables_dir / "label_summary.csv", index=False)
    print(f"wrote {out} and {tables_dir / 'label_summary.csv'}")


if __name__ == "__main__":
    main()
