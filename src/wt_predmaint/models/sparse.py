"""Low-capacity exploratory model: L1-penalised logistic regression (D-064).

Same inputs and train-only preparation as B4; the L1 penalty zeroes most coefficients, so the
model can only lean on a few features. ``C`` is a constructor argument because every value in
``exploratory.sparse_logistic.c_grid`` is run and reported; none is selected.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from wt_predmaint.features.prepare import TrainOnlyPreprocessor
from wt_predmaint.models.base import require_train_only
from wt_predmaint.models.logistic import b4_feature_columns


class SparseLogisticBaseline:
    """Class-weighted L1 logistic regression with train-only scaling and imputation."""

    def __init__(self, target: str, c: float) -> None:
        if target not in {"a", "b"}:
            raise ValueError("target must be 'a' or 'b'")
        if not np.isfinite(c) or c <= 0:
            raise ValueError("C must be finite and positive")
        self.target = target
        self.c = float(c)
        self.name = f"X_sparse_logistic_C{c:g}"

    def fit(self, train: pd.DataFrame, cfg: dict[str, Any]) -> SparseLogisticBaseline:
        require_train_only(train)
        y = train["label"].astype(int)
        if set(y.unique()) != {0, 1}:
            raise ValueError("training rows must contain both label classes 0 and 1")
        columns = b4_feature_columns(train, cfg)
        prep = cfg["preprocessing_p0"]
        self.preprocessor_ = TrainOnlyPreprocessor(
            columns,
            min_observed_share=float(prep["min_observed_share"]),
            add_missing_indicators=bool(prep["add_missing_indicators"]),
            scale=bool(prep["scale"]),
        )
        matrix = self.preprocessor_.fit_transform(train)
        self.model_ = LogisticRegression(
            C=self.c,
            l1_ratio=1.0,  # pure L1 penalty
            solver="liblinear",
            class_weight="balanced",
            max_iter=int(cfg["baseline_b4"]["max_iter"]),
            tol=float(cfg["baseline_b4"]["tolerance"]),
            random_state=int(cfg["seed"]),
        )
        self.model_.fit(matrix, y)
        self.n_nonzero_ = int(np.count_nonzero(self.model_.coef_))
        return self

    def score(self, rows: pd.DataFrame) -> pd.Series:
        """Positive-class probability per row, finite, indexed like ``rows``."""
        if not hasattr(self, "model_"):
            raise RuntimeError("fit must be called before score")
        p = self.model_.predict_proba(self.preprocessor_.transform(rows.copy()))[:, 1]
        p = np.where(np.isfinite(p), p, 0.5)
        return pd.Series(p.astype("float64"), index=rows.index)
