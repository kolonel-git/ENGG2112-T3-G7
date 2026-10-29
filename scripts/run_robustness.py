"""O3 robustness on validation years: sensor-group dropout and leave-one-turbine-out (D-062).

Uses the frozen LightGBM settings (D-060), the primary model ``L1_lgbm`` and validation rows only;
the test years are not read. Writes ``reports/tables/robustness_dropout.csv``,
``robustness_loto.csv`` (one row per left-out turbine) and ``robustness_loto_summary.csv``.

    python scripts/run_robustness.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from wt_predmaint.config import load_protocol
from wt_predmaint.evaluation.metrics import episodes_in_split, threshold_for_budget
from wt_predmaint.evaluation.robustness import dropout_table, leave_one_turbine_out
from wt_predmaint.evaluation.splits import split_years
from wt_predmaint.models.base import attach_labels, labelled
from wt_predmaint.models.lgbm import LightGBMBaseline

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interim", type=Path, default=ROOT / "data" / "interim")
    args = parser.parse_args()
    cfg = load_protocol()
    ev = cfg["evaluation"]
    features = pd.read_parquet(args.interim / "features" / "daily_features.parquet")

    dropout, loto, summary = [], [], []
    for target in ("a", "b"):
        labels = pd.read_parquet(args.interim / "labels" / f"{target}_labels.parquet")
        episodes = pd.read_parquet(args.interim / "labels" / f"{target}_episodes.parquet")
        table = attach_labels(features, labels, cfg)
        train, val = labelled(table, "train"), labelled(table, "val")
        model = LightGBMBaseline(target).fit(train, cfg)
        scores = model.score(table.loc[table["split"] == "val"])
        val = val.assign(score=scores.reindex(val.index).to_numpy())
        threshold = threshold_for_budget(
            val["label"].to_numpy(),
            val["score"].to_numpy(),
            float(ev["false_alarm_budget_per_turbine_month"]),
            float(ev["days_per_month"]),
        )
        eps = episodes_in_split(episodes, split_years(cfg, "val"))
        d = dropout_table(model, model.columns_, val, eps, threshold, cfg)
        dropout.append(d.assign(target=target.upper()))

        per, pooled_left, pooled_full = leave_one_turbine_out(
            lambda t=target: LightGBMBaseline(t), table, cfg
        )
        loto.append(per.assign(target=target.upper()))
        valid = per["pr_auc_left_out"].notna()
        summary.append(
            {
                "target": target.upper(),
                "pooled_pr_auc_left_out": pooled_left,
                "pooled_pr_auc_full_fit": pooled_full,
                "pooled_prevalence": float(val["label"].mean()),
                "turbines": len(per),
                "turbines_with_both_classes": int(valid.sum()),
                "median_pr_auc_left_out": float(per.loc[valid, "pr_auc_left_out"].median()),
                "median_pr_auc_full_fit": float(per.loc[valid, "pr_auc_full_fit"].median()),
                "min_positive_rows_per_turbine": int(per["n_positive"].min()),
                "median_change_left_out_minus_full": float(
                    (per["pr_auc_left_out"] - per["pr_auc_full_fit"]).median()
                ),
            }
        )

    out = ROOT / "reports" / "tables"
    out.mkdir(parents=True, exist_ok=True)
    dropout_all = pd.concat(dropout, ignore_index=True)
    dropout_all.to_csv(out / "robustness_dropout.csv", index=False)
    pd.concat(loto, ignore_index=True).to_csv(out / "robustness_loto.csv", index=False)
    pd.DataFrame(summary).to_csv(out / "robustness_loto_summary.csv", index=False)
    show = [
        "target",
        "group",
        "n_columns_dropped",
        "pr_auc",
        "pr_auc_change",
        "recall",
        "recall_change",
    ]
    print(dropout_all[show].round(3).to_string(index=False))
    print(pd.DataFrame(summary).round(3).to_string(index=False))
    assert np.isfinite(dropout_all["pr_auc"]).all()


if __name__ == "__main__":
    main()
