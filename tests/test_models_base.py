from typing import Any

import numpy as np
import pandas as pd
import pytest

from wt_predmaint.config import load_protocol
from wt_predmaint.models.base import (
    Baseline,
    attach_labels,
    check_score_contract,
    check_scores,
    labelled,
    require_train_only,
    scoring_rows,
)
from wt_predmaint.models.dummy import PrevalenceBaseline, RandomBaseline

# All data below is synthetic.
CFG = load_protocol()


def make_table() -> pd.DataFrame:
    days = pd.to_datetime(["2020-03-01", "2020-03-02", "2021-03-01", "2021-03-02", "2023-03-01"])
    features = pd.DataFrame(
        {
            "StationId": 1,
            "Day": days,
            "x": [1.0, 2.0, 3.0, 4.0, 5.0],
            "alarm_total_w7": [0, 1, 2, 3, 4],
        }
    )
    labels = pd.DataFrame(
        {
            "StationId": 1,
            "Day": days[:4],  # 2023-03-01 has features but no label row
            "status": pd.Categorical(["negative", "zone", "positive", "in_episode"]),
            "label": pd.array([0, pd.NA, 1, pd.NA], dtype="Int8"),
            "split": ["train", "train", "val", "val"],
        }
    )
    return attach_labels(features, labels, CFG)


def test_attach_labels_keeps_every_feature_day_and_fills_unlabelled_ones() -> None:
    table = make_table()
    assert len(table) == 5
    last = table.iloc[-1]
    assert last["status"] == "no_data" and pd.isna(last["label"]) and last["split"] == "test"
    assert table.iloc[1]["status"] == "zone" and pd.isna(table.iloc[1]["label"])


def test_labelled_and_scoring_rows() -> None:
    table = make_table()
    train = labelled(table, "train")
    assert len(train) == 1 and train["label"].dtype == np.int64  # the zone day has no label
    assert len(scoring_rows(table, "train")) == 2  # but it can still be scored
    assert len(scoring_rows(table, "test")) == 1 and len(labelled(table, "test")) == 0


def test_require_train_only_rejects_other_splits_and_unlabelled_rows() -> None:
    table = make_table()
    require_train_only(labelled(table, "train"))
    with pytest.raises(ValueError):
        require_train_only(labelled(table, "val"))
    with pytest.raises(ValueError):
        require_train_only(scoring_rows(table, "train"))  # includes the unlabelled zone day


def test_floors_follow_the_interface_and_the_score_contract() -> None:
    table = make_table()
    train = labelled(table, "train")
    for baseline in (PrevalenceBaseline("a"), RandomBaseline("a")):
        assert isinstance(baseline, Baseline)
        fitted = baseline.fit(train, CFG)
        assert check_score_contract(fitted, table) == []


def test_random_floor_is_reproducible_uniform_and_seeded() -> None:
    table = make_table()
    train = labelled(table, "train")
    a = RandomBaseline("a").fit(train, CFG).score(table)
    b = RandomBaseline("a").fit(train, CFG).score(table)
    assert (a == b).all() and ((a >= 0) & (a < 1)).all() and a.nunique() == len(a)
    other = dict(CFG, seed=CFG["seed"] + 1)
    assert not (RandomBaseline("a").fit(train, other).score(table) == a).all()


class PeeksAtLabels:
    name, target = "bad_label", "a"

    def fit(self, train: pd.DataFrame, cfg: dict[str, Any]) -> "PeeksAtLabels":
        return self

    def score(self, rows: pd.DataFrame) -> pd.Series:
        return rows["label"].fillna(0).astype(float)


class UsesBatchStatistics:
    name, target = "bad_batch", "a"

    def fit(self, train: pd.DataFrame, cfg: dict[str, Any]) -> "UsesBatchStatistics":
        return self

    def score(self, rows: pd.DataFrame) -> pd.Series:
        return (rows["x"] - rows["x"].mean()).astype(float)  # depends on the whole batch


class ReturnsNaN:
    name, target = "bad_nan", "a"

    def fit(self, train: pd.DataFrame, cfg: dict[str, Any]) -> "ReturnsNaN":
        return self

    def score(self, rows: pd.DataFrame) -> pd.Series:
        return pd.Series(np.nan, index=rows.index)


def test_contract_check_catches_label_use_batch_use_and_nan() -> None:
    table = make_table()
    assert "label or status" in " ".join(check_score_contract(PeeksAtLabels(), table))
    assert "other rows" in " ".join(check_score_contract(UsesBatchStatistics(), table))
    assert "finite" in " ".join(check_score_contract(ReturnsNaN(), table))


def test_check_scores_rejects_misaligned_index_and_non_series() -> None:
    table = make_table()
    with pytest.raises(ValueError):
        check_scores(pd.Series(0.0, index=table.index[:-1]), table)
    with pytest.raises(TypeError):
        check_scores(np.zeros(len(table)), table)  # type: ignore[arg-type]
