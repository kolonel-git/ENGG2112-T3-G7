"""Robustness helpers on synthetic data; no real data is read."""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from wt_predmaint.config import load_protocol
from wt_predmaint.evaluation.robustness import (
    drop_columns,
    dropout_table,
    group_columns,
    leave_one_turbine_out,
)
from wt_predmaint.models.base import labelled
from wt_predmaint.models.lgbm import LightGBMBaseline

CFG = copy.deepcopy(load_protocol())
CHOSEN = {"num_leaves": 4, "n_estimators": 20, "learning_rate": 0.2, "min_child_samples": 5}
CFG["lgbm"]["selected"] = {"a": dict(CHOSEN), "b": dict(CHOSEN)}
CFG["labels"]["feature_window_days"] = 7

COLUMNS = [
    "w7mean_op_wtc_HubPresA_mean",
    "w7std_max_wtc_HydPress_max",
    "w7mean_op_wtc_GeOilTmp_mean",
    "w7mean_op_wtc_ConvWTmp_mean",
    "w7mean_op_wtc_PrWindSp_mean",
    "wind_mean",
    "w7mean_rpm_mean",
    "w7mean_op_wtc_AmbieTmp_mean",
    "alarm_total_w7",
    "days_since_fault_day",
]


def test_columns_are_assigned_to_groups_by_sensor_and_state() -> None:
    groups = group_columns(COLUMNS, CFG)
    assert groups["pressure"] == ["w7mean_op_wtc_HubPresA_mean", "w7std_max_wtc_HydPress_max"]
    assert groups["drivetrain_temperature"] == ["w7mean_op_wtc_GeOilTmp_mean"]
    assert groups["power_electronics"] == ["w7mean_op_wtc_ConvWTmp_mean"]
    assert groups["wind_rpm"] == ["w7mean_op_wtc_PrWindSp_mean", "wind_mean", "w7mean_rpm_mean"]
    assigned = {c for cols in groups.values() for c in cols}
    assert "w7mean_op_wtc_AmbieTmp_mean" not in assigned and "alarm_total_w7" not in assigned


def test_a_column_in_two_groups_is_an_error() -> None:
    cfg = copy.deepcopy(CFG)
    cfg["robustness"]["sensor_groups"]["power_electronics"]["tokens"].append("HubPresA")
    with pytest.raises(ValueError, match="both"):
        group_columns(COLUMNS, cfg)


def test_drop_columns_sets_missing_without_touching_the_input() -> None:
    rows = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    out = drop_columns(rows, ["a"])
    assert out["a"].isna().all() and out["b"].tolist() == [3.0, 4.0]
    assert rows["a"].tolist() == [1.0, 2.0]


def make_table(seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_per = 80
    parts = []
    for station in (1, 2, 3):
        labels = rng.integers(0, 2, n_per)
        frame = pd.DataFrame(
            {
                "StationId": station,
                "Day": pd.date_range("2020-01-01", periods=n_per),
                "label": pd.array(labels, dtype="Int8"),
                "status": np.where(labels == 1, "positive", "negative"),
                "split": ["train"] * 50 + ["val"] * 30,
            }
        )
        # the pressure feature carries the signal; the temperature feature is noise
        frame["w7mean_op_wtc_HubPresA_mean"] = 3.0 * labels + rng.normal(0, 0.4, n_per)
        frame["w7mean_op_wtc_GeOilTmp_mean"] = rng.normal(size=n_per)
        parts.append(frame)
    return pd.concat(parts, ignore_index=True)


def test_dropping_the_informative_group_hurts_and_the_other_does_not() -> None:
    table = make_table()
    train, val = labelled(table, "train"), labelled(table, "val")
    model = LightGBMBaseline("a").fit(train, CFG)
    episodes = pd.DataFrame({"StationId": [1], "Start": pd.to_datetime(["2020-02-25"])})
    out = dropout_table(model, model.columns_, val, episodes, 0.5, CFG)
    by = out.set_index("group")
    assert out.iloc[0]["group"] == "none" and out.iloc[0]["pr_auc_change"] == 0
    assert by.loc["pressure", "pr_auc_change"] < -0.1
    assert abs(by.loc["drivetrain_temperature", "pr_auc_change"]) < 0.1
    assert by.loc["pressure", "n_columns_dropped"] == 1


def test_leave_one_turbine_out_returns_one_row_per_turbine_with_counts() -> None:
    table = make_table()
    per, pooled_left, pooled_full = leave_one_turbine_out(lambda: LightGBMBaseline("a"), table, CFG)
    assert per["StationId"].tolist() == [1, 2, 3]
    val = labelled(table, "val")
    expected = val.groupby("StationId")["label"].agg(["size", "sum"])
    assert per["n_rows"].tolist() == expected["size"].tolist()
    assert per["n_positive"].tolist() == expected["sum"].astype(int).tolist()
    assert per["n_positive"].between(1, 29).all()
    assert pooled_left > 0.8 and pooled_full > 0.8
    assert per["pr_auc_left_out"].between(0, 1).all()
