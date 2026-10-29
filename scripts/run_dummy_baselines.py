"""Floor results for the evaluation harness: a constant score (prevalence) and a random score.

Any real model has to beat these. Thresholds are chosen on validation and frozen for test (D-009).
Writes reports/tables/baselines_dummy.csv.

    python scripts/run_dummy_baselines.py
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
from wt_predmaint.evaluation.metrics import (
    evaluation_row,
    threshold_for_budget,
)
from wt_predmaint.evaluation.splits import labelled_split, split_years


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--interim", type=Path, default=ROOT / "data" / "interim")
    args = p.parse_args()
    cfg = load_protocol()
    ev = cfg["evaluation"]
    rng = np.random.default_rng(int(cfg["seed"]))

    rows = []
    for target in ("a", "b"):
        labels = pd.read_parquet(args.interim / "labels" / f"{target}_labels.parquet")
        missing_splits = [split for split in ("val", "test")
                          if labelled_split(labels, split).empty]
        if missing_splits:
            details = "; ".join(
                f"{split}: configured years {split_years(cfg, split)}"
                for split in missing_splits
            )
            p.error(
                f"Target {target.upper()} has no labelled rows for {details}. "
                "Build SCADA, cleaned data, and labels for those years before running "
                "baselines. See reports/tables/label_summary.csv. "
                "No baseline output has been overwritten."
            )
        episodes = pd.read_parquet(args.interim / "labels" / f"{target}_episodes.parquet")
        scorers = {
            "constant (prevalence)": lambda n: np.zeros(n),
            "random uniform": lambda n: rng.uniform(size=n),
        }
        for name, make in scorers.items():
            val = labelled_split(labels, "val").assign(score=lambda d, m=make: m(len(d)))
            thr = threshold_for_budget(
                val["label"].to_numpy(),
                val["score"].to_numpy(),
                float(ev["false_alarm_budget_per_turbine_month"]),
                float(ev["days_per_month"]),
            )
            for split in ("val", "test"):
                frame = val if split == "val" else labelled_split(labels, split)
                if split != "val":
                    frame = frame.assign(score=make(len(frame)))
                years = split_years(cfg, split)
                eps = episodes.loc[episodes["Start"].dt.year.isin(years)]
                row = evaluation_row(name, frame, eps, thr, cfg)
                rows.append({"target": target.upper(), "split": split, **row})

    table = pd.DataFrame(rows)
    out = ROOT / "reports" / "tables"
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "baselines_dummy.csv", index=False)
    show = ["target", "split", "model", "prevalence", "pr_auc", "recall", "median_lead_days",
            "false_alarms_per_turbine_month", "n_episodes"]  # fmt: skip
    print(table[show].round(3).to_string(index=False))
    print(f"wrote {out / 'baselines_dummy.csv'}")


if __name__ == "__main__":
    main()
