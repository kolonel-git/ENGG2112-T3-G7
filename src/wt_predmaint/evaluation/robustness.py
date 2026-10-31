"""O3 robustness on validation years: sensor-group dropout and leave-one-turbine-out (D-062).

Both tests use the frozen LightGBM settings (D-060) and validation rows only; the test years are
not read. Dropout sets the features of a sensor group to missing at scoring time and leaves the
fitted model unchanged, so it measures how much the model leans on that group. Leave-one-turbine-out
refits without one turbine's train rows and scores that turbine's validation rows, which shows
whether performance depends on having seen the turbine.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from wt_predmaint.evaluation.metrics import evaluate, pr_auc
from wt_predmaint.models.base import Baseline, labelled

STRIP = ("w7mean_", "w7std_", "op_", "max_")
STATE = {
    "wind_mean", "wind_std", "rpm_mean", "frac_producing", "frac_idle", "frac_stopped",
}  # fmt: skip
SENSOR = re.compile(r"^wtc_(.+)_(?:mean|max|min)$")


def _base(column: str) -> str:
    """Column name without its window and daily-aggregate prefixes."""
    changed = True
    while changed:
        changed = False
        for prefix in STRIP:
            if column.startswith(prefix):
                column, changed = column.removeprefix(prefix), True
    return column


def group_columns(columns: list[str], cfg: dict[str, Any]) -> dict[str, list[str]]:
    """Feature columns that belong to each sensor group in ``robustness.sensor_groups``.

    A column belongs to a group if its sensor token is listed, or, for a group with ``state``, if it
    is a wind, rpm or operating-state summary. A column in two groups is an error.
    """
    groups = cfg["robustness"]["sensor_groups"]
    found: dict[str, list[str]] = {name: [] for name in groups}
    owner: dict[str, str] = {}
    for column in columns:
        base = _base(column)
        match = SENSOR.match(base)
        for name, spec in groups.items():
            hit = (match is not None and match.group(1) in spec["tokens"]) or (
                bool(spec.get("state")) and base in STATE
            )
            if hit:
                if column in owner:
                    raise ValueError(f"{column} is in both {owner[column]} and {name}")
                owner[column] = name
                found[name].append(column)
    return found


def drop_columns(rows: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Copy of ``rows`` with ``columns`` set to missing."""
    out = rows.copy()
    for column in columns:
        out[column] = np.nan
    return out


def dropout_table(
    model: Baseline,
    columns: list[str],
    val: pd.DataFrame,
    episodes: pd.DataFrame,
    threshold: float,
    cfg: dict[str, Any],
) -> pd.DataFrame:
    """Validation metrics with each sensor group removed, at the intact model's threshold.

    ``val`` holds labelled validation rows with StationId, Day and label; ``episodes`` are the
    validation episodes. The first row, ``none``, is the intact model. The threshold is the one
    chosen on intact validation scores, so false alarms can rise or fall when scores shift.
    """
    groups = group_columns(columns, cfg)
    rows = []
    for name, dropped in [("none", []), *groups.items()]:
        scored = val[["StationId", "Day", "label"]].assign(
            score=model.score(drop_columns(val, dropped)).to_numpy()
        )
        result = evaluate(scored, episodes, threshold, cfg)
        rows.append({"group": name, "n_columns_dropped": len(dropped), **result})
    table = pd.DataFrame(rows)
    table["pr_auc_change"] = table["pr_auc"] - table.loc[0, "pr_auc"]
    table["recall_change"] = table["recall"] - table.loc[0, "recall"]
    return table


def leave_one_turbine_out(
    factory: Callable[[], Baseline],
    table: pd.DataFrame,
    cfg: dict[str, Any],
) -> tuple[pd.DataFrame, float, float]:
    """Per-turbine validation PR-AUC when the turbine is left out of training, against the full fit.

    ``table`` is the feature table with labels attached. Returns one row per turbine (counts,
    prevalence, PR-AUC left out and with the full fit), the pooled PR-AUC of the left-out scores and
    the pooled PR-AUC of the full fit on the same rows. A turbine without both classes in
    validation gets NaN.
    """
    train, val = labelled(table, "train"), labelled(table, "val")
    full = factory().fit(train, cfg)
    full_scores = pd.Series(full.score(val).to_numpy(), index=val.index)
    left_scores = pd.Series(np.nan, index=val.index)
    rows = []
    for station in sorted(val["StationId"].unique()):
        held = val["StationId"] == station
        fold = factory().fit(train.loc[train["StationId"] != station], cfg)
        left_scores.loc[held] = fold.score(val.loc[held]).to_numpy()
        y = val.loc[held, "label"].to_numpy()
        both = 0 < y.sum() < len(y)
        rows.append(
            {
                "StationId": int(station),
                "n_rows": int(held.sum()),
                "n_positive": int(y.sum()),
                "prevalence": float(y.mean()),
                "pr_auc_left_out": pr_auc(y, left_scores[held].to_numpy()) if both else np.nan,
                "pr_auc_full_fit": pr_auc(y, full_scores[held].to_numpy()) if both else np.nan,
            }
        )
    y_all = val["label"].to_numpy()
    return (
        pd.DataFrame(rows),
        pr_auc(y_all, left_scores.to_numpy()),
        pr_auc(y_all, full_scores.to_numpy()),
    )
