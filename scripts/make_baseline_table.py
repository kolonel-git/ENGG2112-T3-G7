"""Validation-only B2 tuning and T0 integration, including turbine sensitivity."""

import argparse
from pathlib import Path

import pandas as pd

from wt_predmaint.config import load_protocol
from wt_predmaint.evaluation.runner import run_baselines
from wt_predmaint.models.alarm import alarm_variant
from wt_predmaint.models.base import attach_labels
from wt_predmaint.models.registry import build

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tune-b2", action="store_true")
    args = parser.parse_args()
    cfg = load_protocol()
    interim = ROOT / "data/interim"
    out = ROOT / "reports/tables"
    features = pd.read_parquet(interim / "features/daily_features.parquet")
    results = []
    parameters = []
    seasonal = []
    for target in ("a", "b"):
        labels = pd.read_parquet(interim / f"labels/{target}_labels.parquet")
        episodes = pd.read_parquet(interim / f"labels/{target}_episodes.parquet")
        table = attach_labels(features, labels, cfg)
        for excluded in [[]] if args.tune_b2 else [[], cfg["baseline_table"]["exclude_stations"]]:
            models = (
                [alarm_variant(target, v) for v in cfg["b2_alarm_count"]["candidates"]]
                if args.tune_b2
                else build(None, target)
            )
            run = run_baselines(models, table, episodes, cfg, target, exclude_stations=excluded)
            results.append(run.table)
            for (variant, name, split), scores in run.scores.items():
                scores.to_parquet(interim / f"scores/{target}_{variant}_{name}_{split}.parquet")
            for model in models:
                if hasattr(model, "parameters"):
                    parameters.extend(
                        dict(variant=run.table.variant.iloc[0], **p) for p in model.parameters
                    )
            print(
                run.table[
                    ["target", "variant", "model", "pr_auc", "recall", "random_floor_recall"]
                ].to_string(index=False),
                flush=True,
            )
        if args.tune_b2:
            rows = results[-1].loc[results[-1].model.str.startswith("B2_")]
            best = rows.sort_values("pr_auc", ascending=False, kind="stable").iloc[0]
            print(f"SELECT {target}: {best.model.split('__')[1]}", flush=True)
            val = table.loc[table.split.eq("val")].copy()
            val["month"] = val.Day.dt.to_period("M").astype(str)
            cols = [c for c in val if c.startswith("alarm_") and c.endswith("_w7")]
            monthly = val.groupby("month")[cols].mean().reset_index()
            monthly["target"] = target
            seasonal.append(monthly)
    pd.concat(results, ignore_index=True).to_csv(
        out / ("b2_tuning.csv" if args.tune_b2 else "t0_validation.csv"), index=False
    )
    if not args.tune_b2:
        combined = pd.concat(results, ignore_index=True)
        for variant, frame in combined.groupby("variant"):
            frame.to_csv(out / f"t0_{variant}.csv", index=False)
    if parameters:
        pd.DataFrame(parameters).to_csv(out / "r1_parameters.csv", index=False)
    if seasonal:
        pd.concat(seasonal).to_csv(out / "b2_monthly_counts.csv", index=False)


if __name__ == "__main__":
    main()
