"""D-071 exploratory models, evaluated through the existing A/B runner on validation only.

These factories are deliberately separate from the frozen default registry. Parameter choices
come from the protocol; the notebook never overwrites frozen baseline results or tunes on test.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.utils.class_weight import compute_sample_weight
from threadpoolctl import threadpool_limits

from wt_predmaint.features.prepare import TrainOnlyPreprocessor
from wt_predmaint.models.base import Baseline, require_train_only
from wt_predmaint.models.lgbm import LightGBMBaseline
from wt_predmaint.models.logistic import LogisticBaseline, b4_feature_columns

KINDS = {
    "random_forest": "X_RF",
    "extra_trees": "X_ExtraTrees",
    "histogram_gradient_boosting": "X_HistGB",
}


class AdditionalTreeBaseline:
    """A train-only daily classifier using the same candidate feature families as B4."""

    def __init__(self, target: str, kind: str) -> None:
        if target not in {"a", "b"}:
            raise ValueError("target must be a or b")
        if kind not in KINDS:
            raise ValueError(f"Unknown model kind: {kind}")
        self.target = target
        self.kind = kind
        self.name = KINDS[kind]

    def fit(self, train: pd.DataFrame, cfg: dict[str, Any]) -> AdditionalTreeBaseline:
        require_train_only(train)
        y = train["label"].astype(int)
        if set(y.unique()) != {0, 1}:
            raise ValueError("Training must include both classes")
        columns = b4_feature_columns(train, cfg)
        if not columns:
            raise ValueError("No permitted daily/window model features found")
        preparation = cfg["preprocessing_p0"]
        clipping = preparation["clip_quantiles"]
        self.preprocessor_ = TrainOnlyPreprocessor(
            columns,
            min_observed_share=float(preparation["min_observed_share"]),
            add_missing_indicators=bool(preparation["add_missing_indicators"]),
            scale=False,
            clip_quantiles=None if clipping is None else tuple(clipping),
        )
        x = self.preprocessor_.fit_transform(train)
        settings = cfg["additional_validation_models"]
        self.parameters_ = dict(settings[self.kind])
        self.n_jobs_ = int(settings["n_jobs"])
        self.fallback_ = float(settings["nonfinite_score_fallback"])
        if self.n_jobs_ < 1 or not 0 <= self.fallback_ <= 1:
            raise ValueError("Use positive n_jobs and a fallback probability in [0, 1]")
        self.class_counts_ = y.value_counts().sort_index().to_dict()
        common = {"random_state": int(cfg["seed"]), **self.parameters_}
        if self.kind == "random_forest":
            self.model_ = RandomForestClassifier(n_jobs=self.n_jobs_, **common)
        elif self.kind == "extra_trees":
            self.model_ = ExtraTreesClassifier(n_jobs=self.n_jobs_, **common)
        else:
            self.model_ = HistGradientBoostingClassifier(**common)
        with threadpool_limits(limits=self.n_jobs_):
            if self.kind == "histogram_gradient_boosting":
                self.model_.fit(x, y, sample_weight=compute_sample_weight("balanced", y))
            else:
                self.model_.fit(x, y)
        return self

    def score(self, rows: pd.DataFrame) -> pd.Series:
        if not hasattr(self, "model_"):
            raise RuntimeError("Fit the model before scoring")
        if rows.empty:
            return pd.Series(index=rows.index, dtype="float64")
        x = self.preprocessor_.transform(rows)
        positive = np.flatnonzero(self.model_.classes_ == 1)
        if len(positive) != 1:
            raise RuntimeError("No unique positive class in the fitted estimator")
        with threadpool_limits(limits=self.n_jobs_):
            values = self.model_.predict_proba(x)[:, int(positive[0])].astype("float64")
        values[~np.isfinite(values)] = self.fallback_
        return pd.Series(values, index=rows.index, dtype="float64")


def build_additional_comparison(target: str, cfg: dict[str, Any]) -> list[Baseline]:
    """New models plus original B4/L1 controls; runner automatically adds constant/random floors."""
    models: list[Baseline] = []
    if cfg["additional_validation_models"]["include_frozen_controls"]:
        models.extend([LogisticBaseline(target), LightGBMBaseline(target)])
    models.extend(AdditionalTreeBaseline(target, kind) for kind in KINDS)
    return models
