"""Drift-robust feature sets and low-capacity models on synthetic data (D-064)."""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from wt_predmaint.config import load_protocol
from wt_predmaint.features.relative import (
    build_feature_set,
    fleet_relative,
    month_means,
    sensor_columns,
)
from wt_predmaint.models.base import check_score_contract, labelled
from wt_predmaint.models.lgbm import LightGBMBaseline
from wt_predmaint.models.sparse import SparseLogisticBaseline

CFG = copy.deepcopy(load_protocol())
CFG["labels"]["feature_window_days"] = 7
CFG["exploratory"]["min_turbines"] = 3
CHOSEN = {"num_leaves": 4, "n_estimators": 20, "learning_rate": 0.2, "min_child_samples": 5}
CFG["lgbm"]["selected"] = {"a": dict(CHOSEN), "b": dict(CHOSEN)}
SENS = "w7mean_op_wtc_GeOilTmp_mean"
ALARM = "alarm_total_w7"


def make_table(seed: int = 4, years: tuple[int, ...] = (2016, 2017)) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    parts = []
    for station in (1, 2, 3, 4):
        days = pd.concat(
            [pd.Series(pd.date_range(f"{y}-01-01", f"{y}-12-31")) for y in years],
            ignore_index=True,
        )
        n = len(days)
        labels = rng.integers(0, 2, n)
        drift = 0.01 * np.arange(n)  # shared by all turbines: fleet drift
        season = 5 * np.sin(2 * np.pi * days.dt.dayofyear / 365)
        parts.append(
            pd.DataFrame(
                {
                    "StationId": station,
                    "Day": days,
                    "label": pd.array(labels, dtype="Int8"),
                    "status": np.where(labels == 1, "positive", "negative"),
                    "split": "train",
                    SENS: drift + season + 2.0 * labels + rng.normal(0, 0.3, n),
                    ALARM: rng.integers(0, 4, n),
                }
            )
        )
    return pd.concat(parts, ignore_index=True)


def test_only_sensor_and_weather_columns_are_transformed() -> None:
    cols = [
        SENS,
        "w7std_wind_mean",
        "w7mean_rpm_mean",
        "alarm_3130_w7",
        "frac_idle",
        "days_since_fault_day",
    ]
    assert sensor_columns(cols) == [SENS, "w7std_wind_mean", "w7mean_rpm_mean"]


def test_fleet_relative_removes_shared_drift_and_needs_enough_turbines() -> None:
    table = make_table()
    rel = fleet_relative(table, [SENS], min_turbines=3)[SENS]
    by_day = rel.groupby(table["Day"]).median()
    assert by_day.abs().max() < 1e-4
    few = table.loc[table["StationId"] <= 2]
    assert fleet_relative(few, [SENS], min_turbines=3)[SENS].isna().all()


def test_a_later_day_cannot_change_an_earlier_fleet_value() -> None:
    table = make_table()
    before = fleet_relative(table, [SENS], 3)[SENS]
    late = table.copy()
    late.loc[late["Day"] >= "2017-06-01", SENS] += 1000.0
    after = fleet_relative(late, [SENS], 3)[SENS]
    early = table["Day"] < "2017-06-01"
    assert np.allclose(before[early], after[early], equal_nan=True)


def test_month_means_use_train_years_only() -> None:
    table = make_table(years=(2016, 2017, 2020))
    rel = fleet_relative(table, [SENS], 3)
    only_train = month_means(rel, table["Day"], [2016, 2017])
    with_2020 = month_means(rel, table["Day"], [2016, 2017, 2020])
    assert only_train.shape == (12, 1)
    changed = table.copy()
    changed.loc[changed["Day"].dt.year == 2020, SENS] += 500.0
    rel2 = fleet_relative(changed, [SENS], 3)
    assert np.allclose(month_means(rel2, changed["Day"], [2016, 2017]), only_train, equal_nan=True)
    assert with_2020.shape == only_train.shape


def test_feature_sets_keep_names_and_leave_other_columns_alone() -> None:
    cfg = copy.deepcopy(CFG)
    cfg["splits"]["train_years"] = [2016, 2017]
    table = make_table()
    raw = build_feature_set(table, cfg, "raw")
    assert raw is table
    for name in ("fleet", "fleet_season"):
        out = build_feature_set(table, cfg, name)
        assert list(out.columns) == list(table.columns)
        assert out[ALARM].equals(table[ALARM]) and out["label"].equals(table["label"])
        assert not np.allclose(out[SENS], table[SENS], equal_nan=True)
    with pytest.raises(ValueError, match="unknown"):
        build_feature_set(table, cfg, "bogus")


def test_season_removal_flattens_the_month_pattern() -> None:
    cfg = copy.deepcopy(CFG)
    cfg["splits"]["train_years"] = [2016, 2017]
    table = make_table()
    # turbines see the same season, so the fleet step already removes it; month means then ~0
    out = build_feature_set(table, cfg, "fleet_season")
    by_month = out[SENS].groupby(table["Day"].dt.month).mean()
    assert by_month.abs().max() < 0.05


def test_shallow_trees_and_sparse_logistic_keep_the_score_contract() -> None:
    table = make_table()
    for col in ("w7std_op_wtc_GeOilTmp_mean", "frac_idle"):
        table[col] = np.random.default_rng(1).normal(size=len(table))
    train = labelled(table, "train")
    shallow = LightGBMBaseline(
        "a",
        overrides={"num_leaves": 3, "max_depth": 2, "n_estimators": 10, "min_child_samples": 20},
        name="X_shallow",
    ).fit(train, CFG)
    assert shallow.name == "X_shallow" and shallow.params_["max_depth"] == 2
    sample = table.sample(200, random_state=0)
    assert check_score_contract(shallow, sample) == []
    sparse = SparseLogisticBaseline("a", c=0.05).fit(train, CFG)
    assert sparse.name == "X_sparse_logistic_C0.05" and sparse.n_nonzero_ >= 1
    tiny = SparseLogisticBaseline("a", c=0.001).fit(train, CFG)
    n_features = len(tiny.preprocessor_.get_feature_names_out())
    assert tiny.n_nonzero_ < n_features  # the L1 penalty really zeroes coefficients
    assert check_score_contract(sparse, sample) == []
    with pytest.raises(ValueError, match="positive"):
        SparseLogisticBaseline("a", c=0)
