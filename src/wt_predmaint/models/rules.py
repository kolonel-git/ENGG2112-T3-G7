"""Recent-fault rules (B1, D-050). All inputs are past-only target-A history."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from wt_predmaint.models.base import require_train_only

NAMES = {
    "combined": "B1_recent_fault",
    "count": "B1_fault_count",
    "recency": "B1_fault_recency",
}


class RecentFaultRule:
    """Count recent fault days, with bounded recency to break count ties.

    Recency is 1 / (1 + days_since_fault_day), or zero before the first known
    fault. The primary score is count + tie_weight * recency. With a weight
    strictly between zero and one, recency cannot overturn a one-day count gap.
    Missing counts use the configured fallback; no statistics are fitted.
    Both targets deliberately use target A's history (BASELINE_SPLIT 6.1).
    """

    def __init__(self, target: str, variant: str = "combined") -> None:
        if target not in {"a", "b"}:
            raise ValueError("target must be 'a' or 'b'")
        if variant not in NAMES:
            raise ValueError(f"unknown B1 variant {variant!r}")
        self.target, self.variant, self.name = target, variant, NAMES[variant]
        self.settings: dict[str, float] | None = None

    def fit(self, train: pd.DataFrame, cfg: dict[str, Any]) -> RecentFaultRule:
        require_train_only(train)
        settings = cfg["b1_recent_fault"]
        weight = float(settings["tie_weight"])
        missing_count = float(settings["missing_count"])
        if not 0 < weight < 1:
            raise ValueError("tie_weight must be strictly between zero and one")
        if not np.isfinite(missing_count) or missing_count < 0:
            raise ValueError("missing_count must be finite and nonnegative")
        self.settings = {"tie_weight": weight, "missing_count": missing_count}
        return self

    def score(self, rows: pd.DataFrame) -> pd.Series:
        if self.settings is None:
            raise RuntimeError("call fit before score")
        result = pd.Series(0.0, index=rows.index)
        if self.variant != "recency":
            count = self._history(rows, "fault_days_30").fillna(self.settings["missing_count"])
            result = result + count
        if self.variant != "count":
            days = self._history(rows, "days_since_fault_day")
            recency = (1.0 / (1.0 + days)).fillna(0.0)
            weight = self.settings["tie_weight"] if self.variant == "combined" else 1.0
            result = result + weight * recency
        return result

    @staticmethod
    def _history(rows: pd.DataFrame, column: str) -> pd.Series:
        values = rows[column].astype(float)
        present = values.dropna()
        if not np.isfinite(present).all() or (present < 0).any():
            raise ValueError(f"{column} must contain nonnegative finite values or missing values")
        return values
