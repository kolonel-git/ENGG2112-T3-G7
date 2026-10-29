"""Chronological within-year CV for constant and random dummy baselines.

Uses existing labels/episodes from build_labels.py; no features or fitted models
are needed. Each quarter after Q1 is a test block. Earlier days calibrate the
threshold, with the full label horizon purged before the test block. Test rows
whose full horizon reaches the next block are also excluded. This is exploratory
within-year evaluation, separate from the project's held-out-year results.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from wt_predmaint.config import load_protocol
from wt_predmaint.evaluation.metrics import evaluation_row, threshold_for_budget

YEAR = 2018  # Change this or pass --year.


def chronological_folds(labels: pd.DataFrame, year: int, horizon: int):
    """Yield expanding calibration and disjoint test blocks shared by all turbines."""
    start = pd.Timestamp(year=year, month=1, day=1)
    end = pd.Timestamp(year=year + 1, month=1, day=1)
    data = labels.loc[
        labels["Day"].between(start, end, inclusive="left") & labels["label"].notna()
    ].sort_values(["StationId", "Day"]).copy()
    data["label"] = data["label"].astype(int)
    horizon_end = data["Day"] + pd.Timedelta(days=horizon)
    for fold, month in enumerate((4, 7, 10), 1):
        test_start = pd.Timestamp(year=year, month=month, day=1)
        test_end = test_start + pd.DateOffset(months=3)
        calibration = data.loc[horizon_end < test_start].copy()
        test = data.loc[(data["Day"] >= test_start) & (horizon_end < test_end)].copy()
        yield fold, test_start, test_end, calibration, test


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=YEAR)
    parser.add_argument("--interim", type=Path, default=ROOT / "data/interim")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resamples", type=int, help="Override bootstrap resamples")
    args = parser.parse_args()
    if not 1900 <= args.year <= 9998:
        parser.error("--year must be between 1900 and 9998")
    if args.resamples is not None and args.resamples < 1:
        parser.error("--resamples must be positive")
    cfg = load_protocol()
    ev = cfg["evaluation"]
    horizon = int(cfg["labels"]["exclusion_end_days"])
    rows = []
    for target_index, target in enumerate(("a", "b")):
        folder = args.interim / "labels"
        labels = pd.read_parquet(folder / f"{target}_labels.parquet")
        episodes = pd.read_parquet(folder / f"{target}_episodes.parquet")
        for fold, start, end, calibration, test in chronological_folds(
            labels, args.year, horizon
        ):
            if calibration.empty or test.empty:
                print(f"Target {target.upper()}, fold {fold}: skipped; "
                      f"{len(calibration)} calibration and {len(test)} test rows")
                continue
            # Independent, reproducible streams for each target and fold.
            rng = np.random.default_rng([int(cfg["seed"]), args.year, target_index, fold])
            eps = episodes.loc[episodes["Start"].between(start, end, inclusive="left")]
            for name in ("constant (prevalence)", "random uniform"):
                val = calibration.copy()
                frame = test.copy()
                val["score"] = (np.zeros(len(val)) if name.startswith("constant")
                                else rng.uniform(size=len(val)))
                frame["score"] = (np.zeros(len(frame)) if name.startswith("constant")
                                  else rng.uniform(size=len(frame)))
                threshold = threshold_for_budget(
                    val["label"].to_numpy(), val["score"].to_numpy(),
                    float(ev["false_alarm_budget_per_turbine_month"]),
                    float(ev["days_per_month"]),
                )
                result = evaluation_row(name, frame, eps, threshold, cfg, args.resamples)
                rows.append({
                    "year": args.year, "target": target.upper(), "fold": fold,
                    "test_block_start": start.date(), "test_block_end_exclusive": end.date(),
                    "calibration_rows": len(val), "purge_days": horizon, **result,
                })
    if not rows:
        parser.error("No evaluable folds. Build labels for the selected year and check coverage.")
    output = args.output or ROOT / f"reports/tables/baselines_dummy_cv_{args.year}.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    table.to_csv(output, index=False)
    print(table[["target", "fold", "model", "n_rows", "pr_auc", "recall",
                 "false_alarms_per_turbine_month"]].round(3).to_string(index=False))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
