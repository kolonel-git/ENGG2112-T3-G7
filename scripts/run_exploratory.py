"""Exploratory validation-only runs: drift-robust features and low-capacity models (D-064).

Every combination of feature set and model in the ``exploratory`` block of the protocol is run;
none is selected and nothing is tuned. Each is scored (a) on the validation years through the shared
runner and (b) with a year-shift check on train years only (fit on days up to ``fit_up_to``, score
``score_year``). The test years are never scored here: the runner is called without ``frozen``.
Writes ``reports/tables/exploratory_validation.csv``.

``--round 2`` runs the second round (D-065): the rolling self-baseline feature sets, and adds the
turbine-ranking metric on validation scores; it writes ``exploratory_self_baseline.csv``.

    python scripts/run_exploratory.py
    python scripts/run_exploratory.py --round 2
"""

from __future__ import annotations

import argparse
import copy
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd

from wt_predmaint.config import load_protocol
from wt_predmaint.evaluation.metrics import pr_auc
from wt_predmaint.evaluation.ranking import daily_precision_at_k, ranking_summary
from wt_predmaint.evaluation.runner import run_baselines
from wt_predmaint.features.relative import build_feature_set
from wt_predmaint.models.base import Baseline, attach_labels, labelled
from wt_predmaint.models.lgbm import LightGBMBaseline
from wt_predmaint.models.logistic import LogisticBaseline
from wt_predmaint.models.sparse import SparseLogisticBaseline

ROOT = Path(__file__).resolve().parents[1]


def model_grid(cfg: dict[str, Any], target: str) -> list[tuple[str, str, Callable[[], Baseline]]]:
    """(name, setting, factory) for every model in the exploratory grid."""
    ex = cfg["exploratory"]
    shallow = ex["shallow_trees"]
    grid: list[tuple[str, str, Callable[[], Baseline]]] = [
        ("L1 LightGBM (frozen)", "D-060", lambda: LightGBMBaseline(target, name="X_lgbm_frozen")),
        ("B4 logistic (frozen)", "D-041", lambda: LogisticBaseline(target)),
    ]
    for trees in shallow["n_estimators"]:
        overrides = {k: v for k, v in shallow.items() if k != "n_estimators"}
        overrides["n_estimators"] = trees
        grid.append(
            (
                "Shallow trees",
                f"{trees} trees, depth {shallow['max_depth']}",
                lambda o=overrides, t=trees: LightGBMBaseline(
                    target, overrides=o, name=f"X_shallow_{t}"
                ),
            )
        )
    for c in ex["sparse_logistic"]["c_grid"]:
        grid.append(
            ("Sparse logistic (L1)", f"C={c:g}", lambda c=c: SparseLogisticBaseline(target, c))
        )
    return grid


def year_shift(factory: Callable[[], Baseline], table: pd.DataFrame, cfg: dict[str, Any]) -> dict:
    """Fit on train days up to ``fit_up_to`` and score ``score_year`` (both are train years)."""
    ys = cfg["exploratory"]["year_shift"]
    train = labelled(table, "train")
    year = train["Day"].dt.year
    fit_rows = train.loc[year <= int(ys["fit_up_to"])]
    score_rows = train.loc[year == int(ys["score_year"])]
    model = factory().fit(fit_rows, cfg)
    scores = model.score(score_rows)
    y = score_rows["label"].to_numpy()
    return {
        "year_shift_pr_auc": pr_auc(y, scores.to_numpy()),
        "year_shift_prevalence": float(y.mean()),
        "year_shift_rows": len(y),
    }


def ranking_columns(
    scores: dict, model_name: str, table: pd.DataFrame, cfg: dict[str, Any]
) -> dict[str, float]:
    """Precision at the top k turbines per day against chance, on validation (D-065)."""
    rk = cfg["exploratory_round2"]["ranking"]
    frame = scores[("all", model_name, "val")]
    labels = labelled(table, "val")[["StationId", "Day", "label"]]
    joined = labels.merge(frame, on=["StationId", "Day"], how="inner")
    ks = [int(k) for k in rk["top_k"]]
    daily = daily_precision_at_k(joined, ks, int(rk["min_turbines"]), int(cfg["seed"]))
    summary = ranking_summary(daily, ks, int(rk["resamples"]), int(cfg["seed"]))
    out: dict[str, float] = {}
    for r in summary.itertuples(index=False):
        out[f"rank_precision_{r.k}"] = r.mean_precision
        out[f"rank_chance_{r.k}"] = r.mean_chance
        out[f"rank_diff_{r.k}"] = r.difference
        out[f"rank_diff_{r.k}_lo"] = r.difference_lo
        out[f"rank_diff_{r.k}_hi"] = r.difference_hi
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interim", type=Path, default=ROOT / "data" / "interim")
    parser.add_argument("--round", type=int, choices=[1, 2], default=1)
    args = parser.parse_args()
    cfg = load_protocol()
    ex = cfg["exploratory"]
    r2 = cfg["exploratory_round2"]
    feature_sets = ex["feature_sets"] if args.round == 1 else r2["feature_sets"]
    out_name = "exploratory_validation.csv" if args.round == 1 else "exploratory_self_baseline.csv"
    features = pd.read_parquet(args.interim / "features" / "daily_features.parquet")

    rows: list[dict[str, Any]] = []
    for target in ("a", "b"):
        labels = pd.read_parquet(args.interim / "labels" / f"{target}_labels.parquet")
        episodes = pd.read_parquet(args.interim / "labels" / f"{target}_episodes.parquet")
        for feature_set in feature_sets:
            table = attach_labels(build_feature_set(features, cfg, feature_set), labels, cfg)
            for name, setting, factory in model_grid(cfg, target):
                model = factory()
                result = run_baselines(
                    [model], table, episodes, copy.deepcopy(cfg), target,
                    splits=("val",), frozen=False, resamples=int(ex["resamples"]),
                )  # fmt: skip
                row = result.table.loc[result.table["model"] == model.name].iloc[0].to_dict()
                extra = {}
                if args.round == 2:
                    extra = ranking_columns(result.scores, model.name, table, cfg)
                rows.append(
                    {
                        "target": target.upper(),
                        "feature_set": feature_set,
                        "model": name,
                        "setting": setting,
                        **{
                            k: row[k]
                            for k in (
                                "n_rows",
                                "n_positive_rows",
                                "pr_auc",
                                "pr_auc_lo",
                                "pr_auc_hi",
                                "recall",
                                "false_alarms_per_turbine_month",
                                "random_floor_pr_auc",
                                "random_floor_recall",
                            )
                        },  # fmt: skip
                        **year_shift(factory, table, cfg),
                        **extra,
                    }
                )
                print(
                    f"{target.upper()} {feature_set:12s} {name:22s} {setting:18s} "
                    f"val {row['pr_auc']:.3f}  shift {rows[-1]['year_shift_pr_auc']:.3f}",
                    flush=True,
                )
    out = ROOT / "reports" / "tables"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / out_name, index=False)
    print(f"wrote {out / out_name} ({len(rows)} configurations)")


if __name__ == "__main__":
    main()
