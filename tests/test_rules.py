"""B1 tests: all inputs and episodes are synthetic."""

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from wt_predmaint.config import load_protocol
from wt_predmaint.evaluation.runner import run_baselines
from wt_predmaint.models.base import check_score_contract
from wt_predmaint.models.registry import build
from wt_predmaint.models.rules import NAMES, RecentFaultRule


def rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "StationId": 1,
            "Day": pd.date_range("2020-01-01", periods=5),
            "split": "train",
            "label": [0, 1, 0, 1, 0],
            "status": "negative",
            "fault_days_30": [3, 3, 4, 0, np.nan],
            "days_since_fault_day": [1, 20, 100, np.nan, np.nan],
        }, index=[10, 20, 30, 40, 50],
    )


@pytest.mark.parametrize("target", ["a", "b"])
@pytest.mark.parametrize("variant", NAMES)
def test_scores_order_missing_history_and_contract(target: str, variant: str) -> None:
    data = rows()
    rule = RecentFaultRule(target, variant).fit(data, load_protocol())
    score = rule.score(data)
    if variant == "count":
        assert score.tolist() == [3, 3, 4, 0, 0]
    elif variant == "recency":
        np.testing.assert_allclose(score, [1 / 2, 1 / 21, 1 / 101, 0, 0])
    else:
        assert score.loc[30] > score.loc[10] > score.loc[20] > score.loc[40]
        assert score.loc[40] == score.loc[50] == 0
    assert check_score_contract(rule, data) == []
    assert check_score_contract(rule, data.iloc[:0]) == []
    assert rule.score(data.assign(label=np.nan, status="in_episode")).equals(score)


def test_fit_rejects_validation_and_test_and_does_not_learn_outcomes() -> None:
    data, cfg = rows(), load_protocol()
    before = RecentFaultRule("a").fit(data, cfg).score(data)
    # Changing future rows cannot change fit or any score on the train rows.
    future = data.assign(split="val", label=1, fault_days_30=999)
    future.loc[:, "days_since_fault_day"] = 0
    whole = pd.concat([data, future], ignore_index=True)
    for split in ("val", "test"):
        with pytest.raises(ValueError, match="train"):
            RecentFaultRule("a").fit(future.assign(split=split), cfg)
    after = RecentFaultRule("a").fit(whole.loc[whole["split"] == "train"], cfg).score(data)
    pd.testing.assert_series_equal(before, after)
    flipped = RecentFaultRule("a").fit(data.assign(label=1 - data["label"]), cfg)
    pd.testing.assert_series_equal(before, flipped.score(data))


def test_registry_and_config_and_bad_history() -> None:
    cfg = deepcopy(load_protocol())
    data = rows()
    assert [b.name for b in build(list(NAMES.values()), "a")] == list(NAMES.values())
    cfg["b1_recent_fault"]["tie_weight"] = 1
    with pytest.raises(ValueError, match="tie_weight"):
        RecentFaultRule("a").fit(data, cfg)
    rule = RecentFaultRule("a").fit(data, load_protocol())
    for column in ("fault_days_30", "days_since_fault_day"):
        for bad in (-1, np.inf):
            with pytest.raises(ValueError, match=column):
                rule.score(data.assign(**{column: bad}))
    with pytest.raises(RuntimeError, match="fit"):
        RecentFaultRule("a").score(data)


def test_runner_detects_episode_without_false_alarms_and_reports_tied_failure() -> None:
    cfg = load_protocol()
    train = rows().iloc[:1]
    val = pd.DataFrame({
        "StationId": 1, "Day": pd.date_range("2021-01-01", periods=6),
        "split": "val", "label": [0, 0, 1, 1, 0, 0], "status": "negative",
        "fault_days_30": [1, 1, 3, 3, 2, 2], "days_since_fault_day": 5,
    })
    episodes = pd.DataFrame({
        "StationId": [1], "Start": pd.to_datetime(["2021-02-01"]),
        "End": pd.to_datetime(["2021-02-01"]),
    })
    table = pd.concat([train, val], ignore_index=True)
    result = run_baselines([RecentFaultRule("a")], table, episodes, cfg, "a", resamples=10)
    b1 = result.table.set_index("model").loc["B1_recent_fault"]
    assert b1["pr_auc"] == b1["recall"] == 1
    assert b1["false_alarms_per_turbine_month"] == 0
    assert b1["median_lead_days"] == 29
    assert set(result.table["model"]) == {"B0a_constant", "B0b_random", "B1_recent_fault"}
    # All rows tied: the strict threshold cannot select positives without false alarms.
    tied = table.assign(fault_days_30=3)
    result = run_baselines(
        [RecentFaultRule("a", "count")], tied, episodes, cfg, "a", resamples=10,
    )
    assert result.table.set_index("model").loc["B1_fault_count", "recall"] == 0
