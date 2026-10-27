"""D-054: descriptive validation alert burden at a configured budget grid."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from wt_predmaint.evaluation.metrics import evaluate, threshold_for_budget


def budget_sweep(frame: pd.DataFrame, episodes: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Evaluate fixed scores, retaining strict ties and the shared episode definition.

    Inputs must be labelled validation rows. No model fitting or test evaluation is allowed.
    Alert-day counts are restricted to labelled rows and are not maintenance visits.
    """
    if frame.empty or "split" not in frame or not frame["split"].eq("val").all():
        raise ValueError("budget analysis requires nonempty validation rows only")
    if not frame.Day.dt.year.isin(cfg["splits"]["val_years"]).all():
        raise ValueError("budget analysis received dates outside validation years")
    if frame.duplicated(["StationId", "Day"]).any():
        raise ValueError("duplicate turbine-day scores")
    if not frame.label.isin([0, 1]).all() or not np.isfinite(frame.score).all():
        raise ValueError("budget analysis requires binary labels and finite scores")
    if not episodes.Start.dt.year.isin(cfg["splits"]["val_years"]).all():
        raise ValueError("episodes must be from validation years")
    budgets = np.asarray(cfg["budget_analysis"]["budgets"], dtype=float)
    if not len(budgets) or not np.isfinite(budgets).all() or (budgets < 0).any():
        raise ValueError("budgets must be finite, nonnegative and nonempty")
    if len(set(budgets)) != len(budgets):
        raise ValueError("budgets must be unique")
    rows = []
    for budget in sorted(budgets):
        threshold = threshold_for_budget(
            frame.label.to_numpy(),
            frame.score.to_numpy(),
            float(budget),
            float(cfg["evaluation"]["days_per_month"]),
        )
        metrics = evaluate(frame, episodes, threshold, cfg)
        alerts = frame.score.gt(threshold)
        rows.append(
            {
                "budget": float(budget),
                "threshold": threshold,
                **metrics,
                "false_alert_days": int((alerts & frame.label.eq(0)).sum()),
                "labelled_alert_days": int(alerts.sum()),
                "missed_episodes": int(metrics["n_episodes"] - metrics["n_detected"]),
            }
        )
    return pd.DataFrame(rows)


def add_random_comparison(table: pd.DataFrame) -> pd.DataFrame:
    """Require one random reference at each target/fleet/budget and join it explicitly."""
    keys = ["target", "variant", "budget"]
    floor = table.loc[table.model.eq("B0b_random"), keys + ["recall", "false_alert_days"]]
    floor = floor.rename(
        columns={"recall": "random_floor_recall", "false_alert_days": "random_false_alert_days"}
    )
    out = table.merge(floor, on=keys, how="left", validate="many_to_one", indicator=True)
    if not out["_merge"].eq("both").all():
        raise ValueError("every budget comparison needs its matching random floor")
    out["recall_minus_random_floor"] = out.recall - out.random_floor_recall
    return out.drop(columns="_merge")
