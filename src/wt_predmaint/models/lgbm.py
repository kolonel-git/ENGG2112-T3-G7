"""L1: class-weighted LightGBM for targets A and B (D-056).

Inputs are the daily and 7-day window features B4 uses (``b4_feature_columns``), plus the NBM
residual columns when ``use_nbm`` is set. Everything is learned from labelled train rows only: the
columns kept, the positive-class weight (negatives over positives in train) and the trees. Missing
values go to LightGBM as they are. Hyper-parameters come from the ``lgbm`` block of the protocol,
chosen on validation by ``scripts/tune_lgbm.py`` and frozen per target before any test scoring.
"""

from __future__ import annotations

from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from wt_predmaint.config import require
from wt_predmaint.models.base import require_train_only
from wt_predmaint.models.logistic import b4_feature_columns

NBM_PREFIX = "nbm_"
TUNABLE = ("num_leaves", "n_estimators", "learning_rate", "min_child_samples")


def lgbm_feature_columns(table: pd.DataFrame, cfg: dict[str, Any], use_nbm: bool) -> list[str]:
    """B4's candidate columns, then the NBM columns (in table order) if requested."""
    columns = b4_feature_columns(table, cfg)
    if use_nbm:
        columns += [c for c in table.columns if c.startswith(NBM_PREFIX)]
    return columns


def lgbm_params(cfg: dict[str, Any], target: str) -> dict[str, Any]:
    """Frozen settings for one target; raises if the target's selection is still TODO."""
    block = cfg["lgbm"]
    chosen = require(cfg, f"lgbm.selected.{target}")
    params = {k: chosen[k] for k in TUNABLE}
    params.update(
        subsample=float(block["subsample"]),
        colsample_bytree=float(block["colsample_bytree"]),
        reg_lambda=float(block["reg_lambda"]),
    )
    return params


class LightGBMBaseline:
    """Class-weighted gradient-boosted trees with train-only column selection."""

    def __init__(self, target: str, use_nbm: bool = False) -> None:
        if target not in {"a", "b"}:
            raise ValueError("target must be 'a' or 'b'")
        self.target = target
        self.use_nbm = use_nbm
        self.name = "L1_lgbm_nbm" if use_nbm else "L1_lgbm"

    def fit(self, train: pd.DataFrame, cfg: dict[str, Any]) -> LightGBMBaseline:
        require_train_only(train)
        y = train["label"].astype(int)
        if set(y.unique()) != {0, 1}:
            raise ValueError("LightGBM training rows must contain both label classes 0 and 1")
        candidates = lgbm_feature_columns(train, cfg, self.use_nbm)
        if self.use_nbm and not any(c.startswith(NBM_PREFIX) for c in candidates):
            raise ValueError("L1_lgbm_nbm needs the NBM columns: run scripts/build_nbm.py first")
        share = float(cfg["preprocessing_p0"]["min_observed_share"])
        x = train[candidates].apply(pd.to_numeric, errors="coerce").astype("float32")
        x = x.replace([np.inf, -np.inf], np.nan)
        keep = x.columns[(x.notna().mean() >= share) & (x.nunique(dropna=True) > 1)]
        if not len(keep):
            raise ValueError("no LightGBM feature columns remain after train-only selection")
        self.columns_ = list(keep)
        self.params_ = lgbm_params(cfg, self.target)
        self.fallback_ = float(cfg["lgbm"]["nonfinite_score_fallback"])
        self.scale_pos_weight_ = float((y == 0).sum() / (y == 1).sum())
        self.model_ = lgb.LGBMClassifier(
            objective="binary",
            scale_pos_weight=self.scale_pos_weight_,
            subsample_freq=1,
            random_state=int(cfg["seed"]),
            deterministic=True,
            force_row_wise=True,
            n_jobs=1,
            verbose=-1,
            **self.params_,
        )
        self.model_.fit(x[self.columns_], y)
        return self

    def _matrix(self, rows: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in self.columns_ if c not in rows]
        if missing:
            raise KeyError(f"rows are missing fitted feature columns: {missing[:5]}")
        x = rows[self.columns_].apply(pd.to_numeric, errors="coerce").astype("float32")
        return x.replace([np.inf, -np.inf], np.nan)

    def feature_matrix(self, rows: pd.DataFrame) -> pd.DataFrame:
        """The fitted columns of ``rows`` as the model sees them (float32, non-finite missing)."""
        return self._matrix(rows)

    def score(self, rows: pd.DataFrame) -> pd.Series:
        """Positive-class probability per row; one finite value each, indexed like ``rows``."""
        if not hasattr(self, "model_"):
            raise RuntimeError("fit must be called before score")
        p = self.model_.predict_proba(self._matrix(rows))[:, 1].astype("float64")
        p[~np.isfinite(p)] = self.fallback_
        return pd.Series(p, index=rows.index, dtype="float64")

    def importance_table(self) -> pd.DataFrame:
        """Split-gain importance per kept column, largest first (attribution, not causation)."""
        if not hasattr(self, "model_"):
            raise RuntimeError("fit must be called before importance is available")
        gain = self.model_.booster_.feature_importance(importance_type="gain")
        table = pd.DataFrame({"feature": self.columns_, "gain": gain})
        table["share"] = table["gain"] / table["gain"].sum()
        return table.sort_values("gain", ascending=False, ignore_index=True)
