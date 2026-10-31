"""Turbine-ranking metric for the main models on the validation years (D-065).

Reads the validation scores saved by ``scripts/run_baselines.py`` (only the ``val`` rows of the full
fleet are used; the test years are not ranked, which would be a new look at them), ranks the
labelled turbines each day, and writes ``reports/tables/ranking_validation.csv``.

    python scripts/run_ranking.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from wt_predmaint.config import load_protocol
from wt_predmaint.evaluation.ranking import daily_precision_at_k, ranking_summary
from wt_predmaint.models.base import attach_labels, labelled

ROOT = Path(__file__).resolve().parents[1]
MODELS = [
    "B0b_random",
    "B1_recent_fault",
    "B2_alarm_count",
    "R1_zscore",
    "B3_power_curve",
    "B4_logistic",
    "R2_nbm_indicator",
    "L1_lgbm",
    "L1_lgbm_nbm",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interim", type=Path, default=ROOT / "data" / "interim")
    args = parser.parse_args()
    cfg = load_protocol()
    rk = cfg["exploratory_round2"]["ranking"]
    ks = [int(k) for k in rk["top_k"]]
    features = pd.read_parquet(args.interim / "features" / "daily_features.parquet")

    rows = []
    for target in ("a", "b"):
        labels = pd.read_parquet(args.interim / "labels" / f"{target}_labels.parquet")
        val = labelled(attach_labels(features[["StationId", "Day"]], labels, cfg), "val")
        val = val[["StationId", "Day", "label"]]
        for model in MODELS:
            path = args.interim / "scores" / f"{target}_all_{model}_val.parquet"
            scored = val.merge(pd.read_parquet(path), on=["StationId", "Day"], how="inner")
            daily = daily_precision_at_k(scored, ks, int(rk["min_turbines"]), int(cfg["seed"]))
            summary = ranking_summary(daily, ks, int(rk["resamples"]), int(cfg["seed"]))
            for r in summary.itertuples(index=False):
                rows.append({"target": target.upper(), "model": model, **r._asdict()})
    table = pd.DataFrame(rows)
    out = ROOT / "reports" / "tables"
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "ranking_validation.csv", index=False)
    show = ["target", "model", "k", "mean_precision", "mean_chance", "difference"]
    print(table[show].round(3).to_string(index=False))
    print(f"wrote {out / 'ranking_validation.csv'}")


if __name__ == "__main__":
    main()
