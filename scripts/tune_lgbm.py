"""Choose the LightGBM settings on validation only, separately for targets A and B (D-056).

The whole search is the grid in the ``lgbm`` block of ``configs/protocol.yaml``. Each setting is
fitted on labelled train rows through the shared runner and scored on validation; the selection
uses ``L1_lgbm`` (without NBM columns) so the primary model is not chosen by the extra columns, and
``L1_lgbm_nbm`` is run with the same settings for comparison. The script writes
``reports/tables/lgbm_tuning.csv``, never reads or scores test years and does not edit
``protocol.yaml``: freeze the printed values there, with a decision entry.

    python scripts/tune_lgbm.py --target both --resamples 100
"""

from __future__ import annotations

import argparse
import copy
import itertools
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from wt_predmaint.config import load_protocol
from wt_predmaint.evaluation.runner import run_baselines
from wt_predmaint.models.base import attach_labels
from wt_predmaint.models.lgbm import TUNABLE, LightGBMBaseline
from wt_predmaint.models.nbm import attach_nbm

ROOT = Path(__file__).resolve().parents[1]
SELECT_ON = "L1_lgbm"


def candidates(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Every combination in ``lgbm.grid``, in a fixed order."""
    grid = cfg["lgbm"]["grid"]
    if set(grid) != set(TUNABLE):
        raise ValueError(f"lgbm.grid must define exactly {sorted(TUNABLE)}")
    names = list(TUNABLE)
    values = [sorted({v for v in grid[name]}) for name in names]
    if not all(values):
        raise ValueError("every lgbm.grid entry needs at least one value")
    return [dict(zip(names, combo, strict=True)) for combo in itertools.product(*values)]


def select_row(rows: list[dict[str, Any]], cfg: dict[str, Any]) -> dict[str, Any]:
    """Highest validation PR-AUC; an exact tie goes to fewer leaves, then fewer trees."""
    block = cfg["lgbm"]
    if block["selection_metric"] != "pr_auc" or block["selection_tie_break"] != "simplest":
        raise ValueError("tuning supports pr_auc with the 'simplest' tie-break only")
    finite = [r for r in rows if np.isfinite(float(r["pr_auc"]))]
    if not finite:
        raise ValueError("no finite validation PR-AUC values were produced")
    return max(finite, key=lambda r: (float(r["pr_auc"]), -r["num_leaves"], -r["n_estimators"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interim", type=Path, default=ROOT / "data" / "interim")
    parser.add_argument("--target", choices=["a", "b", "both"], default="both")
    parser.add_argument("--resamples", type=int, help="bootstrap resamples (protocol default)")
    args = parser.parse_args()

    cfg = load_protocol()
    features = attach_nbm(
        pd.read_parquet(args.interim / "features" / "daily_features.parquet"), args.interim
    )
    all_rows: list[dict[str, Any]] = []
    chosen: dict[str, dict[str, Any]] = {}
    for target in ("a", "b") if args.target == "both" else (args.target,):
        labels = pd.read_parquet(args.interim / "labels" / f"{target}_labels.parquet")
        episodes = pd.read_parquet(args.interim / "labels" / f"{target}_episodes.parquet")
        table = attach_labels(features, labels, cfg)
        rows: list[dict[str, Any]] = []
        for params in candidates(cfg):
            trial = copy.deepcopy(cfg)
            trial["lgbm"]["selected"][target] = dict(params)
            for use_nbm in (False, True):
                model = LightGBMBaseline(target, use_nbm=use_nbm)
                result = run_baselines(
                    [model], table, episodes, trial, target,
                    splits=("val",), frozen=False, resamples=args.resamples,
                )  # fmt: skip
                row = result.table.loc[result.table["model"] == model.name].iloc[0].to_dict()
                rows.append({**row, **params})
        pick = select_row([r for r in rows if r["model"] == SELECT_ON], cfg)
        chosen[target] = {k: pick[k] for k in TUNABLE}
        for r in rows:
            r["selected"] = all(r[k] == chosen[target][k] for k in TUNABLE)
        all_rows.extend(rows)

    out = ROOT / "reports" / "tables"
    out.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(all_rows)
    table.to_csv(out / "lgbm_tuning.csv", index=False)
    show = ["target", "model", *TUNABLE, "selected", "pr_auc", "recall", "random_floor_pr_auc"]
    print(table[show].sort_values(["target", "model", "num_leaves", "n_estimators"]).round(4)
          .to_string(index=False))  # fmt: skip
    print("\nFreeze these in configs/protocol.yaml (lgbm.selected) after reviewing the table:")
    for target, params in chosen.items():
        print(f"  {target}: {params}")
    print(f"wrote {out / 'lgbm_tuning.csv'}")


if __name__ == "__main__":
    main()
