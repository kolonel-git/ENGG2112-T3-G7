import copy
from typing import Any

import numpy as np
import pandas as pd
import pytest

from wt_predmaint.config import load_protocol
from wt_predmaint.evaluation.runner import (
    RANDOM_FLOOR,
    ContractError,
    run_baselines,
    variant_name,
    with_floors,
)
from wt_predmaint.labels.horizon import build_horizon_labels
from wt_predmaint.models.base import attach_labels, require_train_only
from wt_predmaint.models.registry import REGISTRY, build

# All data below is synthetic.
T = pd.Timestamp


def cfg_three_years() -> dict[str, Any]:
    cfg = copy.deepcopy(load_protocol())
    cfg["splits"] = {"train_years": [2020], "val_years": [2021], "test_years": [2022]}
    return cfg


CFG = cfg_three_years()


def synthetic(stations: tuple[int, ...] = (1, 2, 3)) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Features with a signal that rises in the 20 days before every episode, plus labels."""
    days = pd.date_range("2020-01-01", "2022-12-31", freq="D")
    rng = np.random.default_rng(0)
    feats, eps = [], []
    for s in stations:
        starts = pd.to_datetime(
            ["2020-03-01", "2020-07-10", "2020-11-20", "2021-02-15", "2021-06-01", "2021-09-20",
             "2022-02-10", "2022-06-15", "2022-10-05"]
        ) + pd.Timedelta(days=int(s))  # fmt: skip
        eps.append(pd.DataFrame({"StationId": s, "Start": starts, "End": starts}))
        signal = np.zeros(len(days))
        for st in starts:
            signal[(days >= st - pd.Timedelta(days=20)) & (days < st)] = 1.0
        feats.append(
            pd.DataFrame({"StationId": s, "Day": days, "x": signal + rng.normal(0, 0.3, len(days))})
        )
    episodes = pd.concat(eps, ignore_index=True)
    features = pd.concat(feats, ignore_index=True)
    labels = build_horizon_labels(features[["StationId", "Day"]], episodes, CFG)
    return attach_labels(features, labels, CFG), episodes


class ColumnScore:
    """Scores a row by one feature column; needs no fitting."""

    def __init__(self, target: str, column: str = "x", name: str = "col_x") -> None:
        self.name, self.target, self.column = name, target, column

    def fit(self, train: pd.DataFrame, cfg: dict[str, Any]) -> "ColumnScore":
        require_train_only(train)
        return self

    def score(self, rows: pd.DataFrame) -> pd.Series:
        return rows[self.column].astype(float)


class UsesLabels:
    name, target = "cheat", "a"

    def fit(self, train: pd.DataFrame, cfg: dict[str, Any]) -> "UsesLabels":
        return self

    def score(self, rows: pd.DataFrame) -> pd.Series:
        return rows["label"].fillna(0).astype(float)


def run(baselines: list, **kw: Any):
    table, episodes = synthetic()
    return run_baselines(baselines, table, episodes, CFG, "a", resamples=30, **kw)


def test_signal_baseline_beats_the_floors_and_floor_columns_are_attached() -> None:
    res = run([ColumnScore("a")])
    t = res.table.set_index("model")
    assert {"B0a_constant", RANDOM_FLOOR, "col_x"} == set(t.index)
    assert t.loc["col_x", "pr_auc"] > t.loc[RANDOM_FLOOR, "pr_auc"] + 0.2
    assert t.loc["col_x", "recall"] > t.loc[RANDOM_FLOOR, "recall"]
    assert t["random_floor_recall"].nunique() == 1  # same floor beside every row
    assert t.loc["col_x", "recall_minus_random_floor"] == pytest.approx(
        t.loc["col_x", "recall"] - t.loc[RANDOM_FLOOR, "recall"]
    )
    assert set(t["split"]) == {"val"} and set(t["variant"]) == {"all"}


def test_threshold_comes_from_validation_and_respects_the_budget() -> None:
    row = run([ColumnScore("a")]).table.set_index("model").loc["col_x"]
    assert row["false_alarms_per_turbine_month"] <= 1.0 + 1e-9
    assert np.isfinite(row["threshold"])


def test_test_split_needs_the_frozen_flag() -> None:
    with pytest.raises(ValueError, match="frozen"):
        run([ColumnScore("a")], splits=("val", "test"))
    res = run([ColumnScore("a")], splits=("val", "test"), frozen=True)
    both = res.table.loc[res.table["model"] == "col_x"].set_index("split")
    assert set(both.index) == {"val", "test"}
    assert both.loc["val", "threshold"] == both.loc["test", "threshold"]  # frozen from validation


def test_label_peeking_baseline_is_rejected_by_the_contract_check() -> None:
    with pytest.raises(ContractError, match="label or status"):
        run([UsesLabels()])


def test_target_mismatch_and_unknown_split_are_errors() -> None:
    with pytest.raises(ValueError, match="target"):
        run([ColumnScore("b")])
    with pytest.raises(ValueError, match="unknown splits"):
        run([ColumnScore("a")], splits=("train",))


def test_excluding_a_turbine_changes_the_variant_and_drops_its_rows() -> None:
    table, episodes = synthetic()
    res = run_baselines(
        [ColumnScore("a")], table, episodes, CFG, "a", resamples=30, exclude_stations=[3]
    )
    assert set(res.table["variant"]) == {"excl_3"}
    scored = res.scores[("excl_3", "col_x", "val")]
    assert 3 not in set(scored["StationId"]) and variant_name([3, 1]) == "excl_1_3"
    full = run([ColumnScore("a")]).table.set_index("model").loc["col_x", "n_rows"]
    assert res.table.set_index("model").loc["col_x", "n_rows"] < full


def test_scores_cover_every_day_with_features_including_unlabelled_ones() -> None:
    table, _ = synthetic()
    res = run([ColumnScore("a")])
    scored = res.scores[("all", "col_x", "val")]
    assert len(scored) == int((table["split"] == "val").sum())
    assert scored["score"].notna().all()
    labelled_val = int(((table["split"] == "val") & table["label"].notna()).sum())
    assert len(scored) > labelled_val  # zone, in-episode and purged days are scored too


def test_floors_are_added_once_and_registry_builds_by_name() -> None:
    names = [b.name for b in with_floors([ColumnScore("a")], "a")]
    assert names == ["B0a_constant", "B0b_random", "col_x"]
    assert [b.name for b in with_floors(with_floors([], "a"), "a")].count("B0b_random") == 1
    assert build([], "a") == [] and len(build(None, "a")) == len(REGISTRY)
    with pytest.raises(KeyError):
        build(["nope"], "a")


def test_run_is_reproducible() -> None:
    a = run([ColumnScore("a")]).table
    b = run([ColumnScore("a")]).table
    pd.testing.assert_frame_equal(a, b)


def test_duplicate_model_names_are_rejected() -> None:
    with pytest.raises(ValueError, match="names must be unique"):
        run([ColumnScore("a"), ColumnScore("a")])


def test_changing_test_data_cannot_change_validation_results_or_thresholds() -> None:
    table, episodes = synthetic()
    original = run_baselines([ColumnScore("a")], table, episodes, CFG, "a", resamples=5)
    other = table.copy()
    other.loc[other.split.eq("test"), "x"] = 1e9
    other.loc[other.split.eq("test"), "label"] = 1
    changed = run_baselines([ColumnScore("a")], other, episodes, CFG, "a", resamples=5)
    pd.testing.assert_frame_equal(original.table, changed.table)
