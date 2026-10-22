"""Tune Revael's B4 regularisation on validation only, separately for targets A and B.

Each candidate is fitted on labelled train rows through Kanav's shared runner. The script writes
the comparison, selected validation row and train-fitted coefficient attribution table. It never
reads or scores test years and does not edit ``protocol.yaml`` automatically.

    python scripts/tune_b4_logistic.py --target both
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from wt_predmaint.config import load_protocol
from wt_predmaint.evaluation.runner import run_baselines
from wt_predmaint.models.base import attach_labels
from wt_predmaint.models.logistic import LogisticBaseline

ROOT = Path(__file__).resolve().parents[1]


def candidate_values(cfg: dict[str, Any]) -> list[float]:
    """Validated, deterministic C grid from the protocol."""
    values = sorted({float(value) for value in cfg["baseline_b4"]["c_grid"]})
    if not values or not all(np.isfinite(value) and value > 0 for value in values):
        raise ValueError("baseline_b4.c_grid must contain finite positive values")
    return values


def select_row(rows: list[dict[str, Any]], cfg: dict[str, Any]) -> dict[str, Any]:
    """Select highest validation metric, breaking an exact tie toward smaller C."""
    metric = str(cfg["baseline_b4"]["selection_metric"])
    tie_break = str(cfg["baseline_b4"]["selection_tie_break"])
    if metric != "pr_auc" or tie_break != "smallest_c":
        raise ValueError("B4 tuning currently supports pr_auc with smallest_c tie-breaking")
    finite = [row for row in rows if np.isfinite(float(row[metric]))]
    if not finite:
        raise ValueError(f"no finite validation {metric} values were produced")
    return max(finite, key=lambda row: (float(row[metric]), -float(row["C"])))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interim", type=Path, default=ROOT / "data" / "interim")
    parser.add_argument("--target", choices=["a", "b", "both"], default="both")
    parser.add_argument(
        "--resamples", type=int, help="bootstrap resamples; protocol default if omitted"
    )
    args = parser.parse_args()

    cfg = load_protocol()
    features = pd.read_parquet(args.interim / "features" / "daily_features.parquet")
    targets = ("a", "b") if args.target == "both" else (args.target,)
    tuning_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    coefficient_frames: list[pd.DataFrame] = []
    selected_values: dict[str, float] = {}

    for target in targets:
        labels = pd.read_parquet(args.interim / "labels" / f"{target}_labels.parquet")
        episodes = pd.read_parquet(args.interim / "labels" / f"{target}_episodes.parquet")
        table = attach_labels(features, labels, cfg)
        target_rows: list[dict[str, Any]] = []
        fitted: dict[float, LogisticBaseline] = {}

        for value in candidate_values(cfg):
            candidate_cfg = copy.deepcopy(cfg)
            candidate_cfg["baseline_b4"]["selected_c"][target] = value
            model = LogisticBaseline(target)
            result = run_baselines(
                [model],
                table,
                episodes,
                candidate_cfg,
                target,
                splits=("val",),
                frozen=False,
                resamples=args.resamples,
            )
            row = result.table.loc[result.table["model"] == model.name].iloc[0].to_dict()
            row["C"] = value
            target_rows.append(row)
            fitted[value] = model

        selected = select_row(target_rows, cfg)
        selected_c = float(selected["C"])
        selected_values[target] = selected_c
        for row in target_rows:
            row["selected"] = float(row["C"]) == selected_c
        tuning_rows.extend(target_rows)
        selected_rows.append(selected)
        top_n = int(cfg["baseline_b4"]["coefficient_top_n"])
        coefficient_frames.append(fitted[selected_c].coefficient_table(top_n))

    output = ROOT / "reports" / "tables"
    output.mkdir(parents=True, exist_ok=True)
    tuning = pd.DataFrame(tuning_rows)
    validation = pd.DataFrame(selected_rows)
    coefficients = pd.concat(coefficient_frames, ignore_index=True)
    tuning.to_csv(output / "b4_logistic_tuning.csv", index=False)
    validation.to_csv(output / "b4_logistic_validation.csv", index=False)
    coefficients.to_csv(output / "b4_logistic_coefficients.csv", index=False)

    show = [
        "target",
        "C",
        "selected",
        "pr_auc",
        "recall",
        "random_floor_pr_auc",
        "random_floor_recall",
        "false_alarms_per_turbine_month",
    ]
    print(tuning[show].sort_values(["target", "C"]).round(4).to_string(index=False))
    print("\nFreeze these values in configs/protocol.yaml after reviewing the table:")
    for target, value in selected_values.items():
        print(f"  {target}: {value:g}")
    print(f"wrote {output / 'b4_logistic_tuning.csv'}")
    print(f"wrote {output / 'b4_logistic_validation.csv'}")
    print(f"wrote {output / 'b4_logistic_coefficients.csv'}")


if __name__ == "__main__":
    main()
