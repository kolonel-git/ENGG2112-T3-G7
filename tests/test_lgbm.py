"""LightGBM baseline on synthetic data; no real data is read."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import pandas as pd
import pytest

from wt_predmaint.config import load_protocol
from wt_predmaint.models.base import check_score_contract, labelled
from wt_predmaint.models.lgbm import LightGBMBaseline, lgbm_feature_columns
from wt_predmaint.models.registry import REGISTRY, build


def config() -> dict[str, Any]:
    cfg = copy.deepcopy(load_protocol())
    cfg["labels"]["feature_window_days"] = 3
    chosen = {"num_leaves": 4, "n_estimators": 20, "learning_rate": 0.2, "min_child_samples": 5}
    cfg["lgbm"]["selected"] = {"a": dict(chosen), "b": dict(chosen)}
    return cfg


CFG = config()


def make_table(seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_train, n_val, n_test = 120, 40, 40
    n = n_train + n_val + n_test
    labels = rng.integers(0, 2, n)
    signal = 2.0 * labels + rng.normal(0, 0.5, n)
    return pd.DataFrame(
        {
            "StationId": np.resize([1, 2, 3], n),
            "Day": pd.date_range("2020-01-01", periods=n, freq="D"),
            "label": pd.array(labels, dtype="Int8"),
            "status": np.where(labels == 1, "positive", "negative"),
            "split": np.repeat(["train", "val", "test"], [n_train, n_val, n_test]),
            "w3mean_signal": signal,
            "w3std_signal": rng.normal(size=n),
            "wind_mean": rng.normal(8, 2, n),
            "alarm_total_w3": rng.integers(0, 5, n),
            "constant_but_named": np.ones(n),
            "nbm_res_GeOilTmp": signal + rng.normal(0, 0.1, n),
            "nbm_indicator": np.abs(signal),
        }
    )


def test_feature_columns_add_nbm_only_when_asked() -> None:
    table = make_table()
    plain = lgbm_feature_columns(table, CFG, use_nbm=False)
    with_nbm = lgbm_feature_columns(table, CFG, use_nbm=True)
    assert not any(c.startswith("nbm_") for c in plain)
    assert {"nbm_res_GeOilTmp", "nbm_indicator"} <= set(with_nbm) - set(plain)
    assert "StationId" not in with_nbm and "label" not in with_nbm


@pytest.mark.parametrize("use_nbm", [False, True])
def test_score_contract_and_class_weight_from_train(use_nbm: bool) -> None:
    table = make_table()
    train = labelled(table, "train")
    model = LightGBMBaseline("a", use_nbm=use_nbm).fit(train, CFG)
    y = train["label"].astype(int)
    assert model.scale_pos_weight_ == pytest.approx((y == 0).sum() / (y == 1).sum())
    rows = table.loc[table["split"] == "val"]
    assert check_score_contract(model, rows) == []
    scores = model.score(rows)
    assert scores.between(0, 1).all()
    # the planted signal should rank positives above negatives on held-out rows
    auc_like = scores[rows["label"] == 1].mean() - scores[rows["label"] == 0].mean()
    assert auc_like > 0.2


def test_nbm_columns_are_used_only_by_the_nbm_variant() -> None:
    train = labelled(make_table(), "train")
    plain = LightGBMBaseline("a").fit(train, CFG)
    nbm = LightGBMBaseline("a", use_nbm=True).fit(train, CFG)
    assert not any(c.startswith("nbm_") for c in plain.columns_)
    assert any(c.startswith("nbm_") for c in nbm.columns_)
    assert "constant_but_named" not in nbm.columns_


def test_fit_is_reproducible_and_seeded() -> None:
    table = make_table()
    train = labelled(table, "train")
    first = LightGBMBaseline("b").fit(train, CFG).score(table)
    second = LightGBMBaseline("b").fit(train, CFG).score(table)
    assert np.allclose(first, second)


def test_fit_rejects_validation_rows_and_one_class() -> None:
    table = make_table()
    with pytest.raises(ValueError, match="outside the train split"):
        LightGBMBaseline("a").fit(labelled(table, "val"), CFG)
    train = labelled(table, "train")
    with pytest.raises(ValueError, match="both label classes"):
        LightGBMBaseline("a").fit(train.loc[train["label"] == 0], CFG)


def test_unselected_target_is_refused() -> None:
    cfg = copy.deepcopy(CFG)
    cfg["lgbm"]["selected"]["a"] = None
    with pytest.raises(ValueError, match="TODO"):
        LightGBMBaseline("a").fit(labelled(make_table(), "train"), cfg)


def test_missing_and_infinite_inputs_still_give_finite_scores() -> None:
    table = make_table()
    model = LightGBMBaseline("a", use_nbm=True).fit(labelled(table, "train"), CFG)
    rows = table.loc[table["split"] == "val"].copy()
    rows.iloc[0, rows.columns.get_loc("w3mean_signal")] = np.nan
    rows.iloc[1, rows.columns.get_loc("nbm_indicator")] = np.inf
    assert np.isfinite(model.score(rows)).all()


def test_importance_table_and_registry() -> None:
    model = LightGBMBaseline("a").fit(labelled(make_table(), "train"), CFG)
    imp = model.importance_table()
    assert imp["share"].sum() == pytest.approx(1.0)
    assert imp.iloc[0]["feature"] in {"w3mean_signal", "w3std_signal", "wind_mean"}
    assert {"L1_lgbm", "L1_lgbm_nbm"} <= set(REGISTRY)
    assert [b.name for b in build(["L1_lgbm_nbm"], "b")] == ["L1_lgbm_nbm"]
