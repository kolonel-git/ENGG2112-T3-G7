"""D-071 models keep the main runner's train-only and score contracts."""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from wt_predmaint.config import load_protocol
from wt_predmaint.evaluation.runner import run_baselines
from wt_predmaint.models.additional import KINDS, AdditionalTreeBaseline
from wt_predmaint.models.base import check_score_contract, labelled


def configuration():
    cfg = copy.deepcopy(load_protocol())
    block = cfg["additional_validation_models"]
    block["n_jobs"] = 1
    for kind in ["random_forest", "extra_trees"]:
        block[kind].update(n_estimators=8, max_depth=3, min_samples_leaf=2)
    block["histogram_gradient_boosting"].update(max_iter=10, min_samples_leaf=3)
    return cfg


def table():
    rng = np.random.default_rng(42)
    n = 240
    y = np.arange(n) % 2
    signal = 4 * y - 2 + rng.normal(0, .1, n)
    frame = pd.DataFrame({
        "StationId": np.resize([1, 2, 3], n),
        "Day": pd.to_datetime(np.repeat(["2020-01-01", "2021-01-01", "2023-01-01"], [160, 40, 40]))
        + pd.to_timedelta(np.r_[np.arange(160), np.arange(40), np.arange(40)], unit="D"),
        "split": np.repeat(["train", "val", "test"], [160, 40, 40]),
        "label": pd.array(y, dtype="Int8"), "status": np.where(y, "positive", "negative"),
        "w7mean_signal": signal, "w7std_signal": np.abs(signal) + rng.normal(0, .01, n),
        "wind_mean": signal, "w7mean_constant": 7., "w7mean_missing": np.nan,
        "alarm_999_w7": y * 1000, "outage_month": False,
    }, index=pd.Index(np.arange(1000, 1000 + n), name="row"))
    frame.loc[frame.index[[3, 8]], "w7mean_signal"] = np.nan
    return frame


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("target", ["a", "b"])
def test_separable_signal_and_score_contract(kind, target):
    data, cfg = table(), configuration()
    train = labelled(data, "train")
    original = train.copy(deep=True)
    model = AdditionalTreeBaseline(target, kind).fit(train, cfg)
    pd.testing.assert_frame_equal(train, original)
    val = labelled(data, "val")
    scores = model.score(val)
    assert check_score_contract(model, val) == []
    assert scores.loc[val.label.eq(1)].mean() > scores.loc[val.label.eq(0)].mean()
    assert np.isfinite(scores).all()
    assert model.score(val.iloc[:0]).empty
    assert "w7mean_missing" not in model.preprocessor_.get_feature_names_out()
    assert "w7mean_constant" not in model.preprocessor_.get_feature_names_out()
    assert "alarm_999_w7" not in model.preprocessor_.get_feature_names_out()


@pytest.mark.parametrize("kind", KINDS)
def test_fit_rejects_validation_test_and_unlabelled_rows(kind):
    data, cfg = table(), configuration()
    for split in ["val", "test"]:
        with pytest.raises(ValueError, match="outside the train"):
            AdditionalTreeBaseline("a", kind).fit(labelled(data, split), cfg)
    train = labelled(data, "train")
    train.loc[train.index[0], "label"] = pd.NA
    with pytest.raises(ValueError, match="without a label"):
        AdditionalTreeBaseline("a", kind).fit(train, cfg)


@pytest.mark.parametrize("kind", KINDS)
def test_train_preprocessing_and_reproducibility(kind):
    data, cfg = table(), configuration()
    train = labelled(data, "train")
    first = AdditionalTreeBaseline("a", kind).fit(train, cfg)
    second = AdditionalTreeBaseline("a", kind).fit(train, cfg)
    val = labelled(data, "val")
    np.testing.assert_allclose(first.score(val), second.score(val))
    before = first.preprocessor_.medians_.copy()
    extreme = val.copy()
    extreme["w7mean_signal"] = 1e12
    first.score(extreme)
    pd.testing.assert_series_equal(first.preprocessor_.medians_, before)


@pytest.mark.parametrize("kind", KINDS)
def test_main_evaluation_runner_accepts_new_models(kind):
    data, cfg = table(), configuration()
    # Synthetic episodes only. The production notebook uses the existing A/B episode files.
    episodes = pd.DataFrame({"StationId": [1, 2, 3],
                             "Start": pd.to_datetime(["2021-02-01"] * 3),
                             "End": pd.to_datetime(["2021-02-02"] * 3)})
    result = run_baselines([AdditionalTreeBaseline("a", kind)], data, episodes, cfg,
                           "a", splits=("val",), frozen=False, resamples=5)
    assert len(result.table) == 3  # New model plus constant and random floors.
    assert set(result.table.split) == {"val"}
    row = result.table.loc[result.table.model.eq(KINDS[kind])].iloc[0]
    budget = cfg["evaluation"]["false_alarm_budget_per_turbine_month"]
    assert row.false_alarms_per_turbine_month <= budget
    assert "pr_auc_lo" in row.index
    with pytest.raises(ValueError, match="frozen"):
        run_baselines([AdditionalTreeBaseline("a", kind)], data, episodes, cfg,
                      "a", splits=("test",), frozen=False, resamples=5)


@pytest.mark.parametrize("kind", KINDS)
def test_unfitted_and_single_class_errors(kind):
    data, cfg = table(), configuration()
    with pytest.raises(RuntimeError, match="Fit"):
        AdditionalTreeBaseline("a", kind).score(data)
    train = labelled(data, "train")
    train["label"] = 0
    with pytest.raises(ValueError, match="both classes"):
        AdditionalTreeBaseline("a", kind).fit(train, cfg)
