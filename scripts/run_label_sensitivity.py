"""Label sensitivity on the validation years (D-068).

Rebuilds the labels in memory with other positive windows, fault-day thresholds, stop lengths and a
fault-only version of target B, reruns the frozen models on each and writes
``reports/tables/label_sensitivity.csv``. The frozen label files are only read, to check that the
reference variant reproduces them. The test years are never scored (the runner is not frozen).

    python scripts/run_label_sensitivity.py
"""

from __future__ import annotations

import argparse
import copy
import itertools
from pathlib import Path
from typing import Any

import pandas as pd

from wt_predmaint.config import load_protocol
from wt_predmaint.evaluation.ranking import daily_precision_at_k, ranking_summary
from wt_predmaint.evaluation.runner import run_baselines
from wt_predmaint.labels.variants import build_variant, variant_cfg
from wt_predmaint.models.base import attach_labels, labelled
from wt_predmaint.models.registry import build

ROOT = Path(__file__).resolve().parents[1]
SCADA_COLUMNS = ["StationId", "TimeStamp", "wtc_GenRpm_mean", "wtc_PrWindSp_mean", "outage_month"]
KEEP = (
    "n_rows", "n_positive_rows", "pr_auc", "pr_auc_lo", "pr_auc_hi", "recall",
    "false_alarms_per_turbine_month", "random_floor_pr_auc", "random_floor_recall",
)  # fmt: skip


def variants(cfg: dict[str, Any], target: str) -> list[dict[str, Any]]:
    """Every (horizon, severity, weather) combination for one target, reference first."""
    s = cfg["label_sensitivity"]
    if target == "a":
        grid = itertools.product(s["horizons"], s["a_fault_hours"], ["keep"])
    else:
        grid = itertools.product(s["horizons"], s["b_stop_intervals"], s["b_weather_stops"])
    out = [{"horizon": h, "severity": v, "weather": w} for h, v, w in grid]
    ref = {"horizon": 30, "severity": 4 if target == "a" else 18, "weather": "keep"}
    out.sort(key=lambda d: d != ref)
    if out[0] != ref:
        raise ValueError(
            "the reference variant (30 days, 4 hours or 18 intervals) is not in the grid"
        )
    return out


def check_reference(labels: pd.DataFrame, episodes: pd.DataFrame, frozen: Path, name: str) -> None:
    """Stop unless the reference variant reproduces the frozen label and episode files."""
    old = pd.read_parquet(frozen / f"{name}_labels.parquet")
    old_ep = pd.read_parquet(frozen / f"{name}_episodes.parquet")
    keys = ["StationId", "Day"]
    new = labels.sort_values(keys).reset_index(drop=True)
    old = old.sort_values(keys).reset_index(drop=True)
    same = (
        new[keys].equals(old[keys])
        and new["label"].astype("float").equals(old["label"].astype("float"))
        and episodes.reset_index(drop=True).equals(old_ep.reset_index(drop=True))
    )
    if not same:
        raise RuntimeError(f"the reference variant does not reproduce the frozen {name} labels")


def ranking_columns(scores: pd.DataFrame, table: pd.DataFrame, cfg: dict[str, Any]) -> dict:
    """Daily top-k precision minus chance on validation, for one scored model."""
    k = int(cfg["label_sensitivity"]["ranking_top_k"])
    rk = cfg["exploratory_round2"]["ranking"]
    val = labelled(table, "val")[["StationId", "Day", "label"]]
    joined = val.merge(scores, on=["StationId", "Day"], how="inner")
    daily = daily_precision_at_k(joined, [k], int(rk["min_turbines"]), int(cfg["seed"]))
    r = ranking_summary(daily, [k], int(cfg["label_sensitivity"]["resamples"]), int(cfg["seed"]))
    r = r.iloc[0]
    return {
        "rank_diff": r["difference"], "rank_diff_lo": r["difference_lo"],
        "rank_diff_hi": r["difference_hi"],
    }  # fmt: skip


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interim", type=Path, default=ROOT / "data" / "interim")
    args = parser.parse_args()
    cfg = load_protocol()
    sens = cfg["label_sensitivity"]
    clean = args.interim / "clean"
    daily = pd.read_parquet(clean / "daily_summary.parquet")
    scada = pd.concat(
        [pd.read_parquet(f, columns=SCADA_COLUMNS) for f in sorted(clean.glob("scada_*.parquet"))],
        ignore_index=True,
    )
    alarms = pd.read_parquet(args.interim / "alarm_events.parquet")
    features = pd.read_parquet(args.interim / "features" / "daily_features.parquet")

    rows: list[dict[str, Any]] = []
    for target in ("a", "b"):
        for v in variants(cfg, target):
            vcfg = variant_cfg(
                cfg, v["horizon"],
                a_hours=v["severity"] if target == "a" else None,
                b_intervals=v["severity"] if target == "b" else None,
            )  # fmt: skip
            episodes, labels = build_variant(
                target, vcfg, daily, scada, alarms, drop_weather=v["weather"] == "drop"
            )
            is_reference = v == variants(cfg, target)[0]
            if is_reference:
                check_reference(labels, episodes, args.interim / "labels", target)
            table = attach_labels(features, labels, vcfg)
            models = build(list(sens["models"]), target, interim=args.interim)
            result = run_baselines(
                models, table, episodes, copy.deepcopy(vcfg), target,
                splits=("val",), frozen=False, resamples=int(sens["resamples"]),
            )  # fmt: skip
            for _, r in result.table.iterrows():
                extra = ranking_columns(result.scores[("all", r["model"], "val")], table, vcfg)
                rows.append(
                    {
                        "target": target.upper(), **v, "model": r["model"],
                        "prevalence": r["n_positive_rows"] / r["n_rows"],
                        "excess_over_floor": r["pr_auc"] - r["random_floor_pr_auc"],
                        "excludes_floor": bool(
                            r["pr_auc_lo"] > r["random_floor_pr_auc"]
                            or r["pr_auc_hi"] < r["random_floor_pr_auc"]
                        ),
                        **{k: r[k] for k in KEEP}, **extra,
                    }
                )  # fmt: skip
            best = result.table.loc[~result.table["model"].isin(["B0a_prevalence", "B0b_random"])]
            print(
                f"{target.upper()} {v} prevalence "
                f"{best['n_positive_rows'].iloc[0] / best['n_rows'].iloc[0]:.3f} "
                f"floor {best['random_floor_pr_auc'].iloc[0]:.3f} "
                f"best {best['pr_auc'].max():.3f}",
                flush=True,
            )
    out = ROOT / "reports" / "tables"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "label_sensitivity.csv", index=False)
    print(f"wrote {out / 'label_sensitivity.csv'} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
