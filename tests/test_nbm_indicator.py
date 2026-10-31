"""R2 (NBM indicator as a score) on synthetic rows; no real data is read."""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from wt_predmaint.config import load_protocol
from wt_predmaint.models.base import check_score_contract, labelled
from wt_predmaint.models.nbm_indicator import NbmIndicatorBaseline
from wt_predmaint.models.registry import REGISTRY, build

CFG = copy.deepcopy(load_protocol())


def make_table() -> pd.DataFrame:
    n = 30
    labels = np.arange(n) % 2
    indicator = np.where(labels == 1, 2.0, 0.5) + np.linspace(0, 0.1, n)
    indicator[3] = np.nan
    return pd.DataFrame(
        {
            "StationId": np.resize([1, 2], n),
            "Day": pd.date_range("2020-01-01", periods=n, freq="D"),
            "label": pd.array(labels, dtype="Int8"),
            "status": np.where(labels == 1, "positive", "negative"),
            "split": ["train"] * 20 + ["val"] * 10,
            "nbm_indicator": indicator.astype("float32"),
        }
    )


def test_score_is_the_indicator_with_a_fixed_fallback() -> None:
    table = make_table()
    model = NbmIndicatorBaseline("a").fit(labelled(table, "train"), CFG)
    scores = model.score(table)
    assert np.allclose(scores.drop(index=3), table["nbm_indicator"].drop(index=3).astype(float))
    assert scores.iloc[3] == CFG["nbm"]["indicator_missing_score"]
    assert scores.index.equals(table.index) and np.isfinite(scores).all()


def test_score_contract_holds_on_validation_rows() -> None:
    table = make_table()
    model = NbmIndicatorBaseline("b").fit(labelled(table, "train"), CFG)
    assert check_score_contract(model, table.loc[table["split"] == "val"]) == []


def test_infinite_values_get_the_fallback() -> None:
    table = make_table()
    table.loc[5, "nbm_indicator"] = np.inf
    scores = NbmIndicatorBaseline("a").fit(labelled(table, "train"), CFG).score(table)
    assert scores.iloc[5] == CFG["nbm"]["indicator_missing_score"]


def test_fit_needs_the_nbm_column_and_train_rows_only() -> None:
    table = make_table()
    with pytest.raises(ValueError, match="build_nbm"):
        NbmIndicatorBaseline("a").fit(labelled(table.drop(columns="nbm_indicator"), "train"), CFG)
    with pytest.raises(ValueError, match="outside the train split"):
        NbmIndicatorBaseline("a").fit(labelled(table, "val"), CFG)


def test_registered_for_both_targets() -> None:
    assert "R2_nbm_indicator" in REGISTRY
    assert [b.name for b in build(["R2_nbm_indicator"], "a")] == ["R2_nbm_indicator"]
    assert NbmIndicatorBaseline("b").target == "b"
