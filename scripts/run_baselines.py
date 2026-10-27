"""Run the registered baselines (and the two floors) through the evaluation harness.

Fits on train years, chooses thresholds on validation, and reports validation. Test years are scored
only with --frozen, once per frozen configuration. Writes reports/tables/baselines_<target>.csv (and
baselines_<target>_<variant>.csv for a sensitivity run) and per-day scores to
data/interim/scores/ for plotting with scripts/plot_risk_timeline.py --scores.

    python scripts/run_baselines.py --target both
    python scripts/run_baselines.py --target a --exclude 2304515      # sensitivity variant
    python scripts/run_baselines.py --target a --baselines B1_recent_fault
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from wt_predmaint.config import load_protocol
from wt_predmaint.evaluation.runner import run_baselines
from wt_predmaint.models.base import attach_labels
from wt_predmaint.models.registry import REGISTRY, build

ROOT = Path(__file__).resolve().parents[1]
SHOW = [
    "target", "split", "model", "prevalence", "pr_auc", "recall", "random_floor_recall",
    "median_lead_days", "false_alarms_per_turbine_month", "n_episodes",
]  # fmt: skip


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--interim", type=Path, default=ROOT / "data" / "interim")
    p.add_argument("--target", choices=["a", "b", "both"], default="both")
    p.add_argument("--baselines", nargs="*", help=f"ids to run; registered: {sorted(REGISTRY)}")
    p.add_argument("--exclude", type=int, nargs="*", default=[], help="turbine ids to leave out")
    p.add_argument("--frozen", action="store_true", help="also score test (frozen configs only)")
    p.add_argument("--resamples", type=int, help="bootstrap resamples (default: protocol.yaml)")
    args = p.parse_args()

    cfg = load_protocol()
    features = pd.read_parquet(args.interim / "features" / "daily_features.parquet")
    nbm_path = args.interim / "features" / "nbm_residuals.parquet"
    if nbm_path.exists():  # NBM columns (D-054, D-055); other baselines ignore them
        features = features.merge(pd.read_parquet(nbm_path), on=["StationId", "Day"], how="left")
    splits = ("val", "test") if args.frozen else ("val",)
    if args.frozen:
        print("WARNING: scoring the test years. Use only for a frozen configuration.")

    tables_dir = ROOT / "reports" / "tables"
    scores_dir = args.interim / "scores"
    tables_dir.mkdir(parents=True, exist_ok=True)
    scores_dir.mkdir(parents=True, exist_ok=True)

    for target in ("a", "b") if args.target == "both" else (args.target,):
        labels = pd.read_parquet(args.interim / "labels" / f"{target}_labels.parquet")
        episodes = pd.read_parquet(args.interim / "labels" / f"{target}_episodes.parquet")
        table = attach_labels(features, labels, cfg)
        result = run_baselines(
            build(args.baselines, target, interim=args.interim), table, episodes, cfg, target,
            splits=splits, frozen=args.frozen, exclude_stations=args.exclude,
            resamples=args.resamples,
        )  # fmt: skip
        variant = result.table["variant"].iloc[0]
        suffix = "" if variant == "all" else f"_{variant}"
        out = tables_dir / f"baselines_{target}{suffix}.csv"
        result.table.to_csv(out, index=False)
        for (var, name, split), frame in result.scores.items():
            frame.to_parquet(scores_dir / f"{target}_{var}_{name}_{split}.parquet")
        print(f"\ntarget {target.upper()} ({variant})")
        print(result.table[SHOW].round(3).to_string(index=False))
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
