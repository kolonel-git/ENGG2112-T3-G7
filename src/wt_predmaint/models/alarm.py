"""B2 alarm counts; missing inputs contribute zero, with no tie jitter."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from wt_predmaint.models.base import require_train_only


class AlarmCountRule:
    name = "B2_alarm_count"

    def __init__(self, target: str) -> None:
        if target not in {"a", "b"}:
            raise ValueError("target must be a or b")
        self.target = target
        self.variant: str | None = None

    def fit(self, train: pd.DataFrame, cfg: dict[str, Any]) -> AlarmCountRule:
        require_train_only(train)
        settings = cfg["b2_alarm_count"]
        variant = self.variant or settings["selected_variant"][self.target]
        if variant not in settings["candidates"]:
            raise ValueError("unknown alarm variant")
        suffix = f"_w{settings['window_days']}"
        stems = {"total": ["total"], "tier2": ["tier2"], "tier1_2": ["tier1", "tier2"]}
        self.columns = [
            "alarm_" + s + suffix for s in stems.get(variant, [variant.removeprefix("code_")])
        ]
        self.fallback = float(settings["missing_count"])
        return self

    def score(self, rows: pd.DataFrame) -> pd.Series:
        values = rows.reindex(columns=self.columns).apply(pd.to_numeric, errors="coerce")
        values = values.replace([np.inf, -np.inf], np.nan).fillna(self.fallback)
        return values.sum(axis=1).astype("float64")


def alarm_variant(target: str, variant: str) -> AlarmCountRule:
    model = AlarmCountRule(target)
    model.variant = variant
    model.name += "__" + variant
    return model
