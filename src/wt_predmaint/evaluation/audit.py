"""Check saved results against the harness without fitting or scoring a model (D-069)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from wt_predmaint.evaluation.metrics import (
    _check,
    episodes_in_split,
    evaluation_row,
    threshold_for_budget,
)
from wt_predmaint.evaluation.runner import FLOOR_COLUMNS, RANDOM_FLOOR
from wt_predmaint.evaluation.splits import split_years

RESULT_KEYS = ["variant", "target", "model", "split"]
ScoreKey = tuple[str, str, str, str]


def audit_results(
    table: pd.DataFrame,
    frames: dict[ScoreKey, pd.DataFrame],
    episodes: dict[str, pd.DataFrame],
    cfg: dict[str, Any],
) -> pd.DataFrame:
    """Recompute point estimates, bootstrap intervals, validation thresholds and floors.

    ``frames`` contains labelled saved scores keyed by (variant, target, model, split).
    Validation scores are required even for a test-only table. Test rates may exceed the
    validation budget; they are checked against the frozen threshold, never recalibrated.
    This verifies numerical consistency, not that G3 was approved or test was scored once.
    """
    if table.empty or not set(RESULT_KEYS).issubset(table):
        raise ValueError("results must be nonempty and contain the result keys")
    if table[RESULT_KEYS].isna().any().any() or table.duplicated(RESULT_KEYS).any():
        raise ValueError("results must have unique, nonmissing result keys")
    if not table.target.isin(["A", "B"]).all() or not table.split.isin(["val", "test"]).all():
        raise ValueError("results require targets A/B and splits val/test")
    ev = cfg["evaluation"]
    records = []
    for row in table.to_dict("records"):
        key = tuple(row[k] for k in RESULT_KEYS)
        variant, target, model, split = key
        validation = frames[(variant, target, model, "val")]
        frame = frames[key]
        for name, data in (("val", validation), (split, frame)):
            _check(data)
            if not data.Day.dt.year.isin(split_years(cfg, name)).all():
                raise ValueError(f"{key}: scores outside {name} years")
            if "split" not in data or not data.split.eq(name).all():
                raise ValueError(f"{key}: incorrect score split")
        threshold = threshold_for_budget(
            validation.label.to_numpy(), validation.score.to_numpy(),
            float(ev["false_alarm_budget_per_turbine_month"]), float(ev["days_per_month"]),
        )
        eps = episodes_in_split(episodes[target], split_years(cfg, split))
        expected = evaluation_row(model, frame, eps, threshold, cfg)
        for metric, value in expected.items():
            if metric == "model":
                continue
            # A one-ULP threshold change can include a tied score under strict >.
            if metric == "threshold":
                if metric not in row or row[metric] != value:
                    raise ValueError(f"{key}: threshold disagrees with the harness")
                continue
            if metric not in row or not np.isclose(
                row[metric], value, rtol=0, atol=1e-10, equal_nan=True
            ):
                raise ValueError(f"{key}: {metric} disagrees with the harness")
        records.append({**{k: row[k] for k in RESULT_KEYS}, "status": "verified"})

    for _, group in table.groupby(["variant", "target", "split"]):
        floor = group.loc[group.model.eq(RANDOM_FLOOR)]
        if len(floor) != 1:
            raise ValueError("every result group requires exactly one random floor")
        for source, column in FLOOR_COLUMNS.items():
            if column not in group or not np.allclose(
                group[column], floor.iloc[0][source], rtol=0, atol=1e-10, equal_nan=True
            ):
                raise ValueError(f"{column} disagrees with the matching random floor")
        if "recall_minus_random_floor" not in group or not np.allclose(
            group.recall_minus_random_floor, group.recall - group.random_floor_recall,
            rtol=0, atol=1e-10, equal_nan=True,
        ):
            raise ValueError("recall_minus_random_floor disagrees with the matching random floor")
    return pd.DataFrame(records)
