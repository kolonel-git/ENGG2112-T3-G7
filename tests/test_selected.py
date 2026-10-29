"""Train-only feature ranking and pruning on synthetic data (D-066); no real data is read."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import pandas as pd
import pytest

from wt_predmaint.config import load_protocol
from wt_predmaint.models.base import check_score_contract
from wt_predmaint.models.lgbm import LightGBMBaseline
from wt_predmaint.models.logistic import LogisticBaseline
from wt_predmaint.models.selected import (
    SelectedBaseline,
    decorrelate,
    family_columns,
    ranked_columns,
    year_stable_auc_ranking,
)

SIGNAL = "w7mean_op_wtc_GeOilTmp_mean"  # drivetrain_temperature group
FLIPPER = "w7mean_op_wtc_HubPresA_mean"  # pressure group; its sign flips with the year
COPY = "w7mean_op_wtc_GenBeGTm_mean"  # drivetrain_temperature group; copy of the signal
NOISE = "w7mean_op_wtc_ConvWTmp_mean"  # power_electronics group


def config() -> dict[str, Any]:
    cfg = copy.deepcopy(load_protocol())
    cfg["labels"]["feature_window_days"] = 7
    chosen = {"num_leaves": 4, "n_estimators": 25, "learning_rate": 0.2, "min_child_samples": 10}
    cfg["lgbm"]["selected"] = {"a": dict(chosen), "b": dict(chosen)}
    cfg["baseline_b4"]["selected_c"] = {"a": 1.0, "b": 1.0}
    return cfg


CFG = config()


def make_train(seed: int = 4) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    parts = []
    for year in (2016, 2017, 2018, 2020):
        n = 300
        y = rng.integers(0, 2, n)
        signal = 2.0 * y + rng.normal(0, 0.7, n)
        sign = 1 if year % 2 == 0 else -1
        parts.append(
            pd.DataFrame(
                {
                    "StationId": np.resize([1, 2, 3, 4], n),
                    "Day": pd.date_range(f"{year}-01-01", periods=n, freq="D"),
                    "label": y,
                    "status": np.where(y == 1, "positive", "negative"),
                    "split": "train",
                    SIGNAL: signal,
                    COPY: signal + rng.normal(0, 0.01, n),
                    FLIPPER: sign * signal,
                    NOISE: rng.normal(size=n),
                    "wind_mean": rng.normal(8, 2, n),
                    "alarm_total_w7": rng.integers(0, 4, n),
                    "days_since_fault_day": rng.integers(0, 90, n),
                }
            )
        )
    return pd.concat(parts, ignore_index=True)


TRAIN = make_train()
CFG["exploratory_round3"] = {**CFG["exploratory_round3"], "rank_sample_rows": 1000}


def test_year_stable_auc_drops_a_column_whose_direction_flips_between_years() -> None:
    columns = [SIGNAL, COPY, FLIPPER, NOISE]
    ranking = year_stable_auc_ranking(TRAIN, columns)
    assert SIGNAL in ranking.index and COPY in ranking.index
    assert FLIPPER not in ranking.index  # AUC above 0.5 in some years, below in others
    assert ranking[SIGNAL] > 0.3
    assert ranking.index[0] in {SIGNAL, COPY}


def test_gain_cv_ranks_the_informative_columns_above_noise() -> None:
    order = ranked_columns(TRAIN, "a", "gain_cv", CFG)
    assert order.index(NOISE) > min(order.index(SIGNAL), order.index(COPY))
    assert set(order) <= set(TRAIN.columns)


def test_decorrelate_removes_the_near_copy_but_keeps_the_original() -> None:
    kept = decorrelate([SIGNAL, COPY, NOISE], TRAIN, 0.9, 1000, 1)
    assert kept == [SIGNAL, NOISE]
    full = decorrelate([SIGNAL, COPY, NOISE], TRAIN, 1.01, 1000, 1)
    assert full == [SIGNAL, COPY, NOISE]


@pytest.mark.parametrize("method", ["gain_cv", "year_stable_auc", "gain_cv_decorr"])
def test_pruned_models_use_only_the_kept_columns_and_keep_the_score_contract(method: str) -> None:
    model = SelectedBaseline(lambda: LightGBMBaseline("a"), "a", method=method, keep=2, name="X")
    model.fit(TRAIN, CFG)
    assert len(model.columns_) <= 2
    assert set(model.inner_.columns_) <= set(model.columns_)
    rows = TRAIN.sample(80, random_state=2)
    assert check_score_contract(model, rows) == []


def test_logistic_gets_the_same_pruned_columns() -> None:
    model = SelectedBaseline(lambda: LogisticBaseline("a"), "a", method="gain_cv", keep=3, name="X")
    model.fit(TRAIN, CFG)
    used = set(model.inner_.preprocessor_.input_columns_)
    assert used <= set(model.columns_)
    assert check_score_contract(model, TRAIN.sample(80, random_state=3)) == []


def test_selection_is_refitted_from_the_rows_given_not_remembered() -> None:
    first = SelectedBaseline(lambda: LightGBMBaseline("a"), "a", "year_stable_auc", 2, name="X")
    first.fit(TRAIN, CFG)
    shifted = TRAIN.copy()
    shifted[NOISE] = 3.0 * shifted["label"] + np.random.default_rng(1).normal(0, 0.3, len(shifted))
    second = SelectedBaseline(lambda: LightGBMBaseline("a"), "a", "year_stable_auc", 2, name="X")
    second.fit(shifted, CFG)
    assert NOISE in second.columns_ and NOISE not in first.columns_


def test_drop_group_removes_exactly_that_family() -> None:
    columns = [c for c in TRAIN.columns if c.startswith(("w7", "alarm", "days", "wind"))]
    groups = family_columns(columns, CFG)
    assert groups["drivetrain_temperature"] == [SIGNAL, COPY]
    assert groups["alarms"] == ["alarm_total_w7"]
    assert groups["fault_history"] == ["days_since_fault_day"]
    model = SelectedBaseline(
        lambda: LightGBMBaseline("a"), "a", drop_group="drivetrain_temperature"
    )
    model.fit(TRAIN, CFG)
    assert SIGNAL not in model.inner_.columns_ and COPY not in model.inner_.columns_
    assert NOISE in model.inner_.columns_


def test_bad_arguments_and_non_train_rows_are_refused() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        SelectedBaseline(lambda: LightGBMBaseline("a"), "a")
    with pytest.raises(ValueError, match="keep"):
        SelectedBaseline(lambda: LightGBMBaseline("a"), "a", method="gain_cv", keep=0)
    bad = TRAIN.copy()
    bad.loc[0, "split"] = "val"
    with pytest.raises(ValueError, match="outside the train split"):
        SelectedBaseline(lambda: LightGBMBaseline("a"), "a", "gain_cv", 2).fit(bad, CFG)
    with pytest.raises(ValueError, match="unknown"):
        ranked_columns(TRAIN, "a", "nope", CFG)
    with pytest.raises(ValueError, match="no columns"):
        SelectedBaseline(lambda: LightGBMBaseline("a"), "a", drop_group="pressure").fit(
            TRAIN.drop(columns=[FLIPPER]), CFG
        )
