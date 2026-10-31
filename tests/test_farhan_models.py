"""Synthetic B2/R1 contract, signal and healthy-reference tests."""

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from wt_predmaint.config import load_protocol
from wt_predmaint.models.alarm import AlarmCountRule, alarm_variant
from wt_predmaint.models.base import check_score_contract
from wt_predmaint.models.zscore import ZScoreReference


def rows():
    cfg = load_protocol()
    data = pd.DataFrame(
        {
            "StationId": [1] * 40 + [2] * 40,
            "split": "train",
            "label": 0,
            "healthy_fit": True,
            "Day": pd.date_range("2020-01-01", periods=80),
            "status": "negative",
            "alarm_total_w7": np.arange(80),
        }
    )
    for sensor in cfg["r1_zscore"]["sensors"]:
        data["op_" + sensor] = np.tile(np.arange(40), 2)
        data["w7mean_op_" + sensor] = 19.5
    return data


@pytest.mark.parametrize("target", ["a", "b"])
@pytest.mark.parametrize("cls", [AlarmCountRule, ZScoreReference])
def test_contract_and_training_isolation(cls, target):
    data, cfg = rows(), load_protocol()
    original = data.copy(deep=True)
    model = cls(target).fit(data, cfg)
    assert not check_score_contract(model, data)
    pd.testing.assert_frame_equal(data, original)
    expected = model.score(data)
    changed = data.assign(
        Day=pd.Timestamp("2099-01-01"), split="test", label=np.nan, status="in_episode"
    )
    pd.testing.assert_series_equal(expected, model.score(changed))
    pd.testing.assert_series_equal(expected, cls(target).fit(data, cfg).score(data))
    for invalid in (data.assign(split="val"), data.assign(label=np.nan)):
        with pytest.raises(ValueError):
            cls(target).fit(invalid, cfg)
    with pytest.raises(ValueError):
        cls("invalid")
    missing = data.copy()
    for col in missing:
        if col.startswith(("op_", "w7mean_", "alarm_")):
            missing[col] = np.nan
    assert model.score(missing).eq(0).all()


def test_alarm_variants_signal_and_ties():
    data, cfg = rows(), load_protocol()
    data["alarm_tier1_w7"] = 2
    data["alarm_tier2_w7"] = 3
    assert alarm_variant("a", "tier1_2").fit(data, cfg).score(data).eq(5).all()
    assert AlarmCountRule("a").fit(data, cfg).score(data).is_monotonic_increasing
    for candidate in cfg["b2_alarm_count"]["candidates"]:
        model = alarm_variant("b", candidate).fit(data, cfg)
        assert not check_score_contract(model, data)


def test_reference_signal_fallback_and_healthy_exclusion():
    data, cfg = rows(), load_protocol()
    model = ZScoreReference("a").fit(data, cfg)
    assert model.score(data).eq(0).all()
    abnormal = data.copy()
    sensor = cfg["r1_zscore"]["sensors"][0]
    abnormal["w7mean_op_" + sensor] = 100
    assert model.score(abnormal).gt(5).all()
    abnormal["StationId"] = 999
    assert np.isfinite(model.score(abnormal)).all()
    dirty = pd.concat(
        [data, data.assign(healthy_fit=False, **{"op_" + sensor: 99999})], ignore_index=True
    )
    pd.testing.assert_series_equal(
        model.score(abnormal), ZScoreReference("a").fit(dirty, cfg).score(abnormal)
    )
    sparse = deepcopy(cfg)
    sparse["r1_zscore"]["minimum_healthy_rows"] = 60
    fallback = ZScoreReference("a").fit(data, sparse)
    assert all(p["source"] == "fleet" for p in fallback.parameters)
    with pytest.raises(ValueError, match="no usable"):
        ZScoreReference("a").fit(data.assign(healthy_fit=False), cfg)


def test_runner_preserves_complete_history_and_keeps_mask_private():
    from wt_predmaint.evaluation.runner import run_baselines
    from wt_predmaint.models.base import require_train_only

    class Capture:
        target = "a"
        name = "capture"
        requires_healthy_rows = True

        def fit(self, train, cfg):
            require_train_only(train)
            self.train = train.copy()
            return self

        def score(self, rows):
            assert "healthy_fit" not in rows
            return pd.Series(0.0, index=rows.index)

    class Ordinary(Capture):
        name = "ordinary"
        requires_healthy_rows = False

        def fit(self, train, cfg):
            require_train_only(train)
            assert "healthy_fit" not in train
            return self

    cfg = load_protocol()
    train = (
        rows().iloc[:40].drop(columns="healthy_fit").assign(outage_month=False, suspect_wind=False)
    )
    train.loc[10, "status"] = "in_episode"
    train["label"] = train.label.astype(float)
    train.loc[10, "label"] = np.nan
    val = train.iloc[:4].assign(
        Day=pd.date_range("2021-01-01", periods=4), split="val", label=[0, 1, 0, 1]
    )
    table = pd.concat([train, val], ignore_index=True)
    episodes = pd.DataFrame(
        {
            "StationId": [1],
            "Start": [pd.Timestamp("2021-01-20")],
            "End": [pd.Timestamp("2021-01-20")],
        }
    )
    model = Capture()
    run_baselines([model, Ordinary()], table, episodes, cfg, "a", resamples=5)
    assert not model.train.loc[11:17, "healthy_fit"].any()
    assert model.train.loc[18, "healthy_fit"]
    changed = table.copy()
    changed.loc[changed.split.eq("val"), "status"] = "in_episode"
    changed.loc[changed.split.eq("val"), "outage_month"] = True
    other = Capture()
    run_baselines([other], changed, episodes, cfg, "a", resamples=5)
    pd.testing.assert_frame_equal(model.train, other.train)
