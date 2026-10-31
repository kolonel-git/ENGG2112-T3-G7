"""SHAP attribution for the frozen LightGBM models and its stability across turbine folds (D-062).

SHAP values describe how the fitted model uses its inputs (log-odds contribution of each feature to
the score of a day). They are model attribution, not causation: a feature with a large value
tells you what the model leans on, not what drives faults. Everything here runs on validation rows;
the test years are not read.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import shap

from wt_predmaint.models.base import labelled
from wt_predmaint.models.lgbm import LightGBMBaseline


def shap_matrix(model: LightGBMBaseline, rows: pd.DataFrame) -> tuple[np.ndarray, float]:
    """SHAP values (rows x features, log-odds of the positive class) and the base value."""
    x = model.feature_matrix(rows)
    explainer = shap.TreeExplainer(model.model_)
    values = explainer.shap_values(x)
    if isinstance(values, list):  # older shap returns one array per class
        values = values[1]
    values = np.asarray(values)
    if values.ndim == 3:  # newer shap: rows x features x classes
        values = values[:, :, 1]
    base = np.atleast_1d(np.asarray(explainer.expected_value, dtype=float))[-1]
    return values, float(base)


def global_importance(model: LightGBMBaseline, rows: pd.DataFrame) -> pd.DataFrame:
    """Mean absolute SHAP per feature, its share of the total, and the direction of the effect.

    ``direction`` is the rank correlation between a feature's value and its SHAP value on the
    explained rows (positive: higher values push the score up); it is NaN for a feature that is
    constant or almost always missing on these rows.
    """
    values, _ = shap_matrix(model, rows)
    x = model.feature_matrix(rows)
    mean_abs = np.abs(values).mean(axis=0)
    direction = []
    for j, column in enumerate(model.columns_):
        pair = pd.DataFrame({"x": x[column].to_numpy(), "s": values[:, j]}).dropna()
        ok = len(pair) > 10 and pair["x"].nunique() > 1 and pair["s"].nunique() > 1
        direction.append(float(pair["x"].corr(pair["s"], method="spearman")) if ok else np.nan)
    table = pd.DataFrame(
        {"feature": model.columns_, "mean_abs_shap": mean_abs, "direction": direction}
    )
    table["share"] = table["mean_abs_shap"] / table["mean_abs_shap"].sum()
    return table.sort_values("mean_abs_shap", ascending=False, ignore_index=True)


def sample_rows(rows: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """A seeded sample of at most ``n`` rows, in their original order."""
    if len(rows) <= n:
        return rows
    return rows.sample(n, random_state=seed).sort_index()


def importance_stability(
    factory: Callable[[], LightGBMBaseline],
    table: pd.DataFrame,
    cfg: dict[str, Any],
    full_importance: pd.DataFrame,
    rows: pd.DataFrame,
    top_n: int,
) -> pd.DataFrame:
    """Refit with each turbine's train rows removed and compare SHAP importance with the full fit.

    One row per left-out turbine: Spearman correlation of the fold's mean absolute SHAP with the
    full model's over all features, and how many of the full model's ``top_n`` features are still in
    the fold's top ``top_n``. The same explained ``rows`` are used for every fold.
    """
    train = labelled(table, "train")
    full = full_importance.set_index("feature")["mean_abs_shap"]
    top_full = set(full.sort_values(ascending=False).head(top_n).index)
    out = []
    for station in sorted(train["StationId"].unique()):
        fold = factory().fit(train.loc[train["StationId"] != station], cfg)
        imp = global_importance(fold, rows).set_index("feature")["mean_abs_shap"]
        shared = full.index.intersection(imp.index)
        corr = full.loc[shared].corr(imp.loc[shared], method="spearman")
        top_fold = set(imp.sort_values(ascending=False).head(top_n).index)
        out.append(
            {
                "StationId": int(station),
                "spearman_vs_full": float(corr),
                "top_n": top_n,
                "top_n_overlap": len(top_full & top_fold),
                "n_features_shared": len(shared),
            }
        )
    return pd.DataFrame(out)
