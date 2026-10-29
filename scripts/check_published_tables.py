"""Check published exports without predictions; this is not score reproduction (D-069)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from wt_predmaint.config import load_protocol
from wt_predmaint.evaluation.audit import RESULT_KEYS
from wt_predmaint.evaluation.runner import FLOOR_COLUMNS

ROOT = Path(__file__).resolve().parents[1]


def check_tables(tables: Path) -> pd.DataFrame:
    """Fail on export, frozen-threshold, accounting or floor inconsistencies."""
    frames = [pd.read_csv(tables / f"baselines_{t}.csv", float_precision="round_trip")
              for t in ("a", "b")]
    runner = pd.concat(frames, ignore_index=True)
    test = pd.read_csv(tables / "test_results.csv", float_precision="round_trip")
    val = pd.read_csv(tables / "t0_validation.csv", float_precision="round_trip")
    expected = runner.loc[runner.split.eq("test") & runner.variant.eq("all")]
    keys = ["target", "model"]
    shared = [c for c in test if c in runner]
    pd.testing.assert_frame_equal(
        test[shared].sort_values(keys).reset_index(drop=True),
        expected[shared].sort_values(keys).reset_index(drop=True),
        check_exact=False, rtol=0, atol=1e-12,
    )
    # The full-fleet validation export must match T0, including frozen thresholds.
    cols = list(runner.columns)
    pd.testing.assert_frame_equal(
        val.loc[val.variant.eq("all"), cols].sort_values(keys).reset_index(drop=True),
        runner.loc[runner.split.eq("val"), cols].sort_values(keys).reset_index(drop=True),
        check_exact=False, rtol=0, atol=1e-12,
    )
    if runner.duplicated(RESULT_KEYS).any():
        raise ValueError("duplicate runner results")
    for _, group in runner.groupby(["variant", "target", "split"]):
        floor = group.loc[group.model.eq("B0b_random")]
        if len(floor) != 1:
            raise ValueError("missing or duplicate random floor")
        for source, column in FLOOR_COLUMNS.items():
            np.testing.assert_allclose(group[column], floor.iloc[0][source], rtol=0, atol=1e-12)
    for metric in ["pr_auc", "recall", "false_alarms_per_turbine_month", "median_lead_days"]:
        finite = runner[metric].notna()
        if not ((runner.loc[finite, metric + "_lo"] <= runner.loc[finite, metric])
                & (runner.loc[finite, metric] <= runner.loc[finite, metric + "_hi"])).all():
            raise ValueError(f"{metric}: estimate outside interval")
    np.testing.assert_allclose(runner.prevalence, runner.n_positive_rows / runner.n_rows)
    np.testing.assert_allclose(runner.recall, runner.n_detected / runner.n_episodes)
    np.testing.assert_allclose(runner.recall_minus_random_floor,
                               runner.recall - runner.random_floor_recall)
    if not ((runner.n_positive_rows <= runner.n_rows)
            & (runner.n_detected <= runner.n_episodes)).all():
        raise ValueError("impossible counts")
    paired = runner.pivot(index=["variant", *keys], columns="split", values="threshold")
    if not paired.val.eq(paired.test).all():
        raise ValueError("test threshold differs from validation")
    excludes = (test.pr_auc_lo > test.random_floor_pr_auc) | (
        test.pr_auc_hi < test.random_floor_pr_auc)
    if not excludes.eq(test.pr_auc_interval_excludes_floor).all():
        raise ValueError("incorrect interval comparison")
    if not test.recall.ge(test.random_floor_recall).eq(test.recall_at_or_above_floor).all():
        raise ValueError("incorrect recall comparison")
    budget = load_protocol()["evaluation"]["false_alarm_budget_per_turbine_month"]
    if not runner.loc[runner.split.eq("val"), "false_alarms_per_turbine_month"].le(
        budget + 1e-10
    ).all():
        raise ValueError("validation exceeds budget")
    burden = pd.read_csv(tables / "alert_budget_validation.csv", float_precision="round_trip")
    official = burden.loc[burden.budget.eq(budget)]
    compare = ["variant", "target", "model", "n_rows", "n_positive_rows", "n_episodes",
               "n_detected", "pr_auc", "recall", "false_alarms_per_turbine_month"]
    order = ["variant", "target", "model"]
    pd.testing.assert_frame_equal(
        official[compare].sort_values(order).reset_index(drop=True),
        val[compare].sort_values(order).reset_index(drop=True),
        check_exact=False, rtol=0, atol=1e-12,
    )
    np.testing.assert_allclose(burden.missed_episodes, burden.n_episodes - burden.n_detected)
    days = load_protocol()["evaluation"]["days_per_month"]
    np.testing.assert_allclose(burden.false_alarms_per_turbine_month,
                               burden.false_alert_days / (burden.n_rows / days))
    return runner[RESULT_KEYS].assign(status="table_consistent_score_audit_pending")


def main() -> None:
    tables = ROOT / "reports/tables"
    audit = check_tables(tables)
    audit.to_csv(tables / "published_table_audit.csv", index=False)
    paths = [tables / n for n in ["baselines_a.csv", "baselines_b.csv", "test_results.csv",
                                  "t0_validation.csv", "alert_budget_validation.csv"]]
    paths.append(ROOT / "configs/protocol.yaml")
    manifest = {
        "scope": "Published table consistency only; scores unavailable. Metrics and intervals "
                 "are not reproduced. No model fitting, scoring or test recalibration.",
        "sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in paths},
    }
    (tables / "published_table_audit.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Checked {len(audit)} published runner rows; original-score audit pending")


if __name__ == "__main__":
    main()
