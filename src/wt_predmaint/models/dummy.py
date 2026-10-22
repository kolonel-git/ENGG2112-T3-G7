"""The two floors as baselines (B0a, B0b). They also show the interface in use."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from wt_predmaint.models.base import KEYS, require_train_only


class PrevalenceBaseline:
    """B0a: the same score for every row. PR-AUC then equals the positive rate (D-020)."""

    def __init__(self, target: str) -> None:
        self.name = "B0a_constant"
        self.target = target

    def fit(self, train: pd.DataFrame, cfg: dict[str, Any]) -> PrevalenceBaseline:
        require_train_only(train)
        return self

    def score(self, rows: pd.DataFrame) -> pd.Series:
        return pd.Series(0.0, index=rows.index)


class RandomBaseline:
    """B0b: a uniform score fixed per (turbine, day) by hashing the keys with the seed.

    Hashing, not a shared random stream, keeps the score of a row independent of the other rows
    in the batch, as the interface requires.
    """

    def __init__(self, target: str) -> None:
        self.name = "B0b_random"
        self.target = target
        self.seed = 0

    def fit(self, train: pd.DataFrame, cfg: dict[str, Any]) -> RandomBaseline:
        require_train_only(train)
        self.seed = int(cfg["seed"])
        return self

    def score(self, rows: pd.DataFrame) -> pd.Series:
        keys = rows[KEYS].assign(seed=self.seed)
        hashed = pd.util.hash_pandas_object(keys, index=False).to_numpy(dtype=np.uint64)
        return pd.Series(hashed / np.float64(2**64), index=rows.index)
