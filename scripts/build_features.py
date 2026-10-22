"""Build the daily feature table (D-021).

Needs scripts/build_clean_data.py, scripts/build_alarm_table.py and scripts/build_labels.py.
Writes data/interim/features/daily_features.parquet (one row per turbine-day) and
reports/tables/feature_summary.csv (coverage of the labelled rows and missing-value shares).

    python scripts/build_features.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from wt_predmaint.config import load_protocol
from wt_predmaint.features.daily import (
    KEYS,
    assemble_features,
    daily_aggregates,
    feature_columns,
)

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--interim", type=Path, default=ROOT / "data" / "interim")
    args = p.parse_args()
    cfg = load_protocol()
    clean = args.interim / "clean"

    pieces = []
    for f in sorted(clean.glob("scada_*.parquet")):
        pieces.append(daily_aggregates(pd.read_parquet(f), cfg))
        print(f"{f.name}: {len(pieces[-1]):,} turbine-days")
    daily = pd.concat(pieces, ignore_index=True)

    events = pd.read_parquet(args.interim / "alarm_events.parquet")
    summary = pd.read_parquet(clean / "daily_summary.parquet")
    table = assemble_features(daily, events, summary, cfg)

    out = args.interim / "features"
    out.mkdir(parents=True, exist_ok=True)
    table.to_parquet(out / "daily_features.parquet")
    cols = feature_columns(table)
    print(f"feature table: {len(table):,} turbine-days, {len(cols)} model columns")

    rows = []
    for target in ("a", "b"):
        labels = pd.read_parquet(args.interim / "labels" / f"{target}_labels.parquet")
        used = labels.loc[labels["label"].notna(), [*KEYS, "split"]]
        joined = used.merge(table, on=KEYS, how="left", indicator=True)
        for split in ("train", "val", "test"):
            j = joined.loc[joined["split"] == split]
            have = j["_merge"] == "both"
            rows.append(
                {
                    "target": target.upper(),
                    "split": split,
                    "labelled_rows": len(j),
                    "with_features": int(have.sum()),
                    "share_missing_values": round(float(j.loc[have, cols].isna().mean().mean()), 4),
                    "rows_all_op_missing": int(
                        j.loc[have].filter(like="op_").isna().all(axis=1).sum()
                    ),
                }
            )
    summary_table = pd.DataFrame(rows)
    tables = ROOT / "reports" / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    summary_table.to_csv(tables / "feature_summary.csv", index=False)
    print(summary_table.to_string(index=False))


if __name__ == "__main__":
    main()
