"""Reproduce B1 validation results and missing-history counts using the shared runner.

    python scripts/run_b1_recent_fault.py
    python scripts/run_b1_recent_fault.py --exclude 2304515

No test-scoring option: B1 development uses validation only (D-050).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from wt_predmaint.config import load_protocol
from wt_predmaint.evaluation.runner import run_baselines
from wt_predmaint.models.base import attach_labels, labelled, scoring_rows
from wt_predmaint.models.registry import build
from wt_predmaint.models.rules import NAMES

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interim", type=Path, default=ROOT / "data" / "interim")
    parser.add_argument("--exclude", type=int, nargs="*", default=[])
    args = parser.parse_args()
    cfg = load_protocol()
    features = pd.read_parquet(args.interim / "features" / "daily_features.parquet")
    tables_dir, scores_dir = ROOT / "reports" / "tables", args.interim / "scores"
    tables_dir.mkdir(parents=True, exist_ok=True)
    scores_dir.mkdir(parents=True, exist_ok=True)
    coverage = []
    for target in ("a", "b"):
        labels = pd.read_parquet(args.interim / "labels" / f"{target}_labels.parquet")
        episodes = pd.read_parquet(args.interim / "labels" / f"{target}_episodes.parquet")
        table = attach_labels(features, labels, cfg)
        result = run_baselines(
            build(list(NAMES.values()), target), table, episodes, cfg, target,
            exclude_stations=args.exclude,
        )
        result.table["history_target"] = "A"
        variant = result.table["variant"].iloc[0]
        suffix = "" if variant == "all" else f"_{variant}"
        out = tables_dir / f"b1_{target}{suffix}.csv"
        result.table.to_csv(out, index=False)
        for (var, name, split), frame in result.scores.items():
            frame.to_parquet(scores_dir / f"{target}_{var}_{name}_{split}.parquet")
        selected = table.loc[~table["StationId"].isin(args.exclude)]
        for scope, frame in (
            ("all_feature_days", scoring_rows(selected, "val")),
            ("labelled_days", labelled(selected, "val")),
        ):
            coverage.append({
                "target": target.upper(), "variant": variant, "split": "val", "scope": scope,
                "n_rows": len(frame),
                "missing_count": int(frame["fault_days_30"].isna().sum()),
                "missing_recency": int(frame["days_since_fault_day"].isna().sum()),
            })
        show = [
            "target", "model", "pr_auc", "recall", "random_floor_recall",
            "median_lead_days", "false_alarms_per_turbine_month", "n_episodes",
        ]
        print(result.table[show].round(3).to_string(index=False), flush=True)
        print(f"wrote {out}", flush=True)
    pd.DataFrame(coverage).to_csv(tables_dir / f"b1_history_coverage{suffix}.csv", index=False)


if __name__ == "__main__":
    main()
