"""R2: the O1 normal-behaviour indicator used as an O2 risk score (D-058).

The score of a turbine-day is its ``nbm_indicator``: the largest absolute 7-day smoothed
standardised residual over the NBM sensors (``models/nbm.py``). Nothing is learned here. The NBM
itself is fitted on healthy train days by ``scripts/build_nbm.py`` and its columns are joined onto
the feature table by ``models.nbm.attach_nbm``; the fleet median in the fleet-relative variant uses
only turbines observed on the same day. A day with no indicator (too few observed days in the
window, or no input) gets the configured fallback so every scored day has a finite number (D-022).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from wt_predmaint.models.base import require_train_only

COLUMN = "nbm_indicator"


class NbmIndicatorBaseline:
    """R2: score = NBM indicator, with a fixed score for days without one."""

    name = "R2_nbm_indicator"

    def __init__(self, target: str) -> None:
        if target not in {"a", "b"}:
            raise ValueError("target must be 'a' or 'b'")
        self.target = target

    def fit(self, train: pd.DataFrame, cfg: dict[str, Any]) -> NbmIndicatorBaseline:
        require_train_only(train)
        if COLUMN not in train:
            raise ValueError("R2 needs the NBM columns: run scripts/build_nbm.py first")
        self.fallback_ = float(cfg["nbm"]["indicator_missing_score"])
        if not np.isfinite(self.fallback_):
            raise ValueError("nbm.indicator_missing_score must be finite")
        return self

    def score(self, rows: pd.DataFrame) -> pd.Series:
        """The indicator per row; one finite float per row, indexed like ``rows``."""
        if not hasattr(self, "fallback_"):
            raise RuntimeError("fit must be called before score")
        values = pd.to_numeric(rows[COLUMN], errors="coerce").astype("float64")
        values = values.where(np.isfinite(values), self.fallback_)
        return values.rename(None)
