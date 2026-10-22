"""B2 alarm counts; missing inputs contribute zero, with no tie jitter."""

import numpy as np
import pandas as pd

from wt_predmaint.models.base import require_train_only


class AlarmCountRule:
    name = "B2_alarm_count"

    def __init__(self, target):
        if target not in {"a", "b"}:
            raise ValueError("target must be a or b")
        self.target = target
        self.variant = None

    def fit(self, train, cfg):
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

    def score(self, rows):
        values = rows.reindex(columns=self.columns).apply(pd.to_numeric, errors="coerce")
        values = values.replace([np.inf, -np.inf], np.nan).fillna(self.fallback)
        return values.sum(axis=1).astype("float64")


def alarm_variant(target, variant):
    model = AlarmCountRule(target)
    model.variant = variant
    model.name += "__" + variant
    return model
