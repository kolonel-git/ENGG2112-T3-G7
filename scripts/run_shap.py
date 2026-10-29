"""SHAP for the frozen LightGBM models and its stability across turbine folds (D-062).

Explains the primary model ``L1_lgbm`` on validation rows only (the test years are not read) and
writes ``reports/tables/shap_importance_{a,b}.csv`` (all features, largest first),
``shap_stability.csv`` (one row per left-out turbine) and ``nbm_attribution.csv`` (which NBM sensor
sets the indicator, on positive and negative labelled validation days). SHAP is model attribution,
not causation.

    python scripts/run_shap.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from wt_predmaint.config import load_protocol
from wt_predmaint.explain.shap_lgbm import global_importance, importance_stability, sample_rows
from wt_predmaint.models.base import attach_labels, labelled
from wt_predmaint.models.lgbm import LightGBMBaseline

ROOT = Path(__file__).resolve().parents[1]


def nbm_attribution(interim: Path, labels: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Share of validation days on which each NBM sensor has the largest smoothed residual."""
    nbm = pd.read_parquet(interim / "features" / "nbm_residuals.parquet")
    smooth = [c for c in nbm.columns if c.startswith("nbm_w")]
    nbm = nbm.dropna(subset=["nbm_indicator"])
    top = nbm[smooth].abs().idxmax(axis=1).str.replace(r"^nbm_w\d+_", "", regex=True)
    nbm = nbm.assign(sensor=top)
    val = attach_labels(nbm[["StationId", "Day", "sensor"]], labels, cfg)
    val = labelled(val, "val")
    rows = []
    for name, part in (("all labelled", val), ("label 1", val[val["label"] == 1]),
                       ("label 0", val[val["label"] == 0])):  # fmt: skip
        share = part["sensor"].value_counts(normalize=True)
        rows += [
            {"days": name, "n_days": len(part), "sensor": s, "share": float(v)}
            for s, v in share.items()
        ]
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interim", type=Path, default=ROOT / "data" / "interim")
    args = parser.parse_args()
    cfg = load_protocol()
    ex = cfg["explain"]
    features = pd.read_parquet(args.interim / "features" / "daily_features.parquet")
    out = ROOT / "reports" / "tables"
    out.mkdir(parents=True, exist_ok=True)

    stability = []
    for target in ("a", "b"):
        labels = pd.read_parquet(args.interim / "labels" / f"{target}_labels.parquet")
        table = attach_labels(features, labels, cfg)
        model = LightGBMBaseline(target).fit(labelled(table, "train"), cfg)
        val = labelled(table, "val")
        explained = sample_rows(val, int(ex["shap_sample_rows"]), int(cfg["seed"]))
        importance = global_importance(model, explained)
        importance.to_csv(out / f"shap_importance_{target}.csv", index=False)
        print(f"target {target.upper()}: explained {len(explained)} validation rows")
        print(importance.head(int(ex["top_n"])).round(4).to_string(index=False))

        folds = sample_rows(val, int(ex["stability_sample_rows"]), int(cfg["seed"]))
        full = global_importance(model, folds)
        stab = importance_stability(
            lambda t=target: LightGBMBaseline(t), table, cfg, full, folds, int(ex["top_n"])
        )
        stability.append(stab.assign(target=target.upper()))
        if target == "a":
            nbm_attribution(args.interim, labels, cfg).to_csv(
                out / "nbm_attribution.csv", index=False
            )
    result = pd.concat(stability, ignore_index=True)
    result.to_csv(out / "shap_stability.csv", index=False)
    print(result.groupby("target")[["spearman_vs_full", "top_n_overlap"]].describe().round(3))


if __name__ == "__main__":
    main()
