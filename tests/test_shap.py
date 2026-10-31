"""SHAP helpers on synthetic data; no real data is read."""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from wt_predmaint.config import load_protocol
from wt_predmaint.explain.shap_lgbm import (
    global_importance,
    importance_stability,
    sample_rows,
    shap_matrix,
)
from wt_predmaint.models.base import labelled
from wt_predmaint.models.lgbm import LightGBMBaseline

CFG = copy.deepcopy(load_protocol())
CHOSEN = {"num_leaves": 4, "n_estimators": 25, "learning_rate": 0.2, "min_child_samples": 5}
CFG["lgbm"]["selected"] = {"a": dict(CHOSEN), "b": dict(CHOSEN)}
CFG["labels"]["feature_window_days"] = 7


def make_table(seed: int = 2) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    parts = []
    for station in (1, 2, 3, 4):
        n = 90
        labels = rng.integers(0, 2, n)
        parts.append(
            pd.DataFrame(
                {
                    "StationId": station,
                    "Day": pd.date_range("2020-01-01", periods=n),
                    "label": pd.array(labels, dtype="Int8"),
                    "status": np.where(labels == 1, "positive", "negative"),
                    "split": ["train"] * 60 + ["val"] * 30,
                    "w7mean_op_wtc_HubPresA_mean": 3.0 * labels + rng.normal(0, 0.4, n),
                    "w7mean_op_wtc_GeOilTmp_mean": rng.normal(size=n),
                    "w7mean_op_wtc_ConvWTmp_mean": rng.normal(size=n),
                }
            )
        )
    return pd.concat(parts, ignore_index=True)


def fitted() -> tuple[LightGBMBaseline, pd.DataFrame, pd.DataFrame]:
    table = make_table()
    model = LightGBMBaseline("a").fit(labelled(table, "train"), CFG)
    return model, table, labelled(table, "val")


def test_shap_values_add_up_to_the_model_margin() -> None:
    model, _, val = fitted()
    values, base = shap_matrix(model, val)
    margin = model.model_.predict(model.feature_matrix(val), raw_score=True)
    assert values.shape == (len(val), len(model.columns_))
    assert np.allclose(values.sum(axis=1) + base, margin, atol=1e-4)


def test_the_informative_feature_ranks_first_and_pushes_the_score_up() -> None:
    model, _, val = fitted()
    imp = global_importance(model, val)
    assert imp.iloc[0]["feature"] == "w7mean_op_wtc_HubPresA_mean"
    assert imp.iloc[0]["direction"] > 0.5
    assert imp["share"].sum() == pytest.approx(1.0)
    assert imp["mean_abs_shap"].is_monotonic_decreasing


def test_sample_is_seeded_and_keeps_order() -> None:
    rows = make_table()
    a, b = sample_rows(rows, 50, seed=7), sample_rows(rows, 50, seed=7)
    assert a.index.equals(b.index) and a.index.is_monotonic_increasing and len(a) == 50
    assert sample_rows(rows, 10_000, seed=7) is rows


def test_stability_across_turbine_folds_is_high_for_a_clear_signal() -> None:
    model, table, val = fitted()
    full = global_importance(model, val)
    stab = importance_stability(lambda: LightGBMBaseline("a"), table, CFG, full, val, top_n=1)
    assert stab["StationId"].tolist() == [1, 2, 3, 4]
    assert (stab["top_n_overlap"] == 1).all()
    assert stab["spearman_vs_full"].min() > 0.4
