"""Exploratory validation-only runs: drift-robust features and low-capacity models (D-064).

Every combination of feature set and model in the ``exploratory`` block of the protocol is run;
none is selected and nothing is tuned. Each is scored (a) on the validation years through the shared
runner and (b) with a year-shift check on train years only (fit on days up to ``fit_up_to``, score
``score_year``). The test years are never scored here: the runner is called without ``frozen``.
Writes ``reports/tables/exploratory_validation.csv``.

``--round 2`` runs the second round (D-065): the rolling self-baseline feature sets, and adds the
turbine-ranking metric on validation scores; it writes ``exploratory_self_baseline.csv``.

``--round 3`` runs the third round (D-066): train-only feature ranking and pruning, and refits with
one feature group removed; it writes ``exploratory_selection.csv`` and the train-only feature
rankings ``feature_ranking_a.csv`` and ``feature_ranking_b.csv``.

    python scripts/run_exploratory.py
    python scripts/run_exploratory.py --round 2
    python scripts/run_exploratory.py --round 3
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
from wt_predmaint.models.selected import (
    SelectedBaseline,
    family_columns,
    gain_cv_ranking,
    ranked_columns,
    year_stable_auc_ranking,
)
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


def selection_grid(
    cfg: dict[str, Any], target: str, feature_set: str
) -> list[tuple[str, str, dict[str, Any], Callable[[], Baseline]]]:
    """(model, setting, extra columns, factory) for every pruning and group-removal run (D-066)."""
    r3 = cfg["exploratory_round3"]
    bases: dict[str, tuple[str, Callable[[], Baseline]]] = {
        "lgbm_frozen": ("L1 LightGBM (frozen)", lambda: LightGBMBaseline(target, name="X_lgbm")),
        "b4_logistic": ("B4 logistic (frozen)", lambda: LogisticBaseline(target)),
    }
    grid: list[tuple[str, str, dict[str, Any], Callable[[], Baseline]]] = []
    for key in r3["models"]:
        label, base = bases[key]
        grid.append((label, "all columns", {"method": "none", "keep": 0, "dropped": ""}, base))
        for method in r3["methods"]:
            for keep in r3["keep"]:
                extra = {"method": method, "keep": int(keep), "dropped": ""}
                grid.append(
                    (
                        label,
                        f"{method}, top {keep}",
                        extra,
                        lambda b=base, m=method, k=int(keep): SelectedBaseline(
                            b, target, method=m, keep=k, name=f"X_{m}_{k}"
                        ),
                    )
                )
        if feature_set == "raw":
            for group in r3["drop_groups"]:
                extra = {"method": "drop_group", "keep": 0, "dropped": group}
                grid.append(
                    (
                        label,
                        f"without {group}",
                        extra,
                        lambda b=base, g=group: SelectedBaseline(
                            b, target, drop_group=g, name=f"X_drop_{g}"
                        ),
                    )
                )
    return grid


def ranking_table(table: pd.DataFrame, target: str, cfg: dict[str, Any]) -> pd.DataFrame:
    """Train-only ranking of every candidate column, with the top-k membership of each method."""
    train = labelled(table, "train")
    gain = gain_cv_ranking(train, target, cfg)
    usable = list(gain.index)
    auc = year_stable_auc_ranking(train, usable)
    out = pd.DataFrame({"feature": usable, "gain_cv_share": gain.to_numpy()})
    out["gain_cv_rank"] = range(1, len(out) + 1)
    out["auc_weakest_year"] = out["feature"].map(auc)
    out["year_stable"] = out["auc_weakest_year"].notna()
    families = family_columns(usable, cfg)
    owner = {c: g for g, cols in families.items() for c in reversed(cols)}
    out["group"] = out["feature"].map(owner).fillna("other")
    for method in cfg["exploratory_round3"]["methods"]:
        order = ranked_columns(train, target, method, cfg)
        for k in (10, 25, 50):
            out[f"{method}_top{k}"] = out["feature"].isin(order[:k])
    return out


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
    parser.add_argument("--round", type=int, choices=[1, 2, 3], default=1)
    args = parser.parse_args()
    cfg = load_protocol()
    ex = cfg["exploratory"]
    r2 = cfg["exploratory_round2"]
    feature_sets = {1: ex["feature_sets"], 2: r2["feature_sets"]}.get(args.round) or cfg[
        "exploratory_round3"
    ]["feature_sets"]
    out_name = {
        1: "exploratory_validation.csv",
        2: "exploratory_self_baseline.csv",
        3: "exploratory_selection.csv",
    }[args.round]
    features = pd.read_parquet(args.interim / "features" / "daily_features.parquet")

    rows: list[dict[str, Any]] = []
    ranks: dict[str, list[pd.DataFrame]] = {}
    for target in ("a", "b"):
        labels = pd.read_parquet(args.interim / "labels" / f"{target}_labels.parquet")
        episodes = pd.read_parquet(args.interim / "labels" / f"{target}_episodes.parquet")
        for feature_set in feature_sets:
            table = attach_labels(build_feature_set(features, cfg, feature_set), labels, cfg)
            if args.round == 3:
                grid = selection_grid(cfg, target, feature_set)
                rank = ranking_table(table, target, cfg)
                rank.insert(0, "feature_set", feature_set)
                ranks.setdefault(target, []).append(rank)
            else:
                grid = [(n, st, {}, f) for n, st, f in model_grid(cfg, target)]
            for name, setting, labels_extra, factory in grid:
                model = factory()
                result = run_baselines(
                    [model], table, episodes, copy.deepcopy(cfg), target,
                    splits=("val",), frozen=False, resamples=int(ex["resamples"]),
                )  # fmt: skip
                row = result.table.loc[result.table["model"] == model.name].iloc[0].to_dict()
                extra = dict(labels_extra)
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
    for target, frames in ranks.items():
        pd.concat(frames, ignore_index=True).to_csv(
            out / f"feature_ranking_{target}.csv", index=False
        )
    print(f"wrote {out / out_name} ({len(rows)} configurations)")


if __name__ == "__main__":
    main()
