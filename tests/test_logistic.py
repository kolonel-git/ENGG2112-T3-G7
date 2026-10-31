import copy
from typing import Any

import numpy as np
import pandas as pd
import pytest

from wt_predmaint.config import load_protocol
from wt_predmaint.models.base import check_score_contract, labelled
from wt_predmaint.models.logistic import LogisticBaseline, b4_feature_columns


def config() -> dict[str, Any]:
    cfg = copy.deepcopy(load_protocol())
    cfg["labels"]["feature_window_days"] = 3
    cfg["preprocessing_p0"] = {
        "min_observed_share": 0.5,
        "add_missing_indicators": True,
        "scale": True,
        "clip_quantiles": None,
    }
    cfg["baseline_b4"] = {
        "selected_c": {"a": 0.5, "b": 2.0},
        "solver": "liblinear",
        "max_iter": 1000,
        "tolerance": 1e-4,
        "nonfinite_score_fallback": 0.5,
    }
    return cfg


CFG = config()
FEATURES = [
    "w3mean_signal",
    "w3std_signal",
    "alarm_3130_w3",
    "alarm_tier1_w3",
    "alarm_total_w3",
    "frac_producing",
    "frac_idle",
    "frac_stopped",
    "wind_mean",
    "wind_std",
    "rpm_mean",
    "days_since_fault_day",
    "fault_days_30",
]


def make_table() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    n_train, n_val, n_test = 40, 12, 8
    n = n_train + n_val + n_test
    labels = np.arange(n) % 2
    signal = 4.0 * labels - 2.0 + rng.normal(0, 0.15, n)
    split = np.repeat(["train", "val", "test"], [n_train, n_val, n_test])
    table = pd.DataFrame(
        {
            "StationId": np.resize([1, 2, 3], n),
            "Day": pd.date_range("2020-01-01", periods=n, freq="D"),
            "label": pd.array(labels, dtype="Int8"),
            "status": np.where(labels == 1, "positive", "negative"),
            "split": split,
            "w3mean_signal": signal,
            "w3std_signal": np.abs(signal) + rng.normal(0, 0.05, n),
            "alarm_3130_w3": labels + np.arange(n) % 3,
            "alarm_tier1_w3": labels + np.arange(n) % 4,
            "alarm_total_w3": labels + np.arange(n) % 5,
            "frac_producing": np.linspace(0.2, 0.9, n),
            "frac_idle": np.linspace(0.8, 0.1, n),
            "frac_stopped": np.arange(n) % 7 / 7,
            "wind_mean": 5 + signal / 10,
            "wind_std": 1 + np.arange(n) % 6 / 10,
            "rpm_mean": 1000 + 20 * signal,
            "days_since_fault_day": np.arange(n) % 17,
            "fault_days_30": np.arange(n) % 6,
            "w7mean_wrong_window": signal * 100,
            "w3mean_outage_month": signal * 100,
            "w3std_n_intervals": signal * 100,
            "alarm_999_w3": labels * 100,
            "op_raw_mean": signal * 100,
            "max_raw_max": signal * 100,
            "outage_month": False,
            "suspect_wind": False,
            "n_intervals": 144,
            "n_producing": 120,
            "masked_frac": 0.0,
        },
        index=pd.Index(np.arange(100, 100 + n), name="row"),
    )
    table.loc[table.index[[1, 4]], "w3mean_signal"] = np.nan
    table.loc[table.index[-1], "label"] = pd.NA
    table.loc[table.index[-1], "status"] = "zone"
    return table


def fitted(target: str = "a", table: pd.DataFrame | None = None) -> LogisticBaseline:
    data = make_table() if table is None else table
    return LogisticBaseline(target).fit(labelled(data, "train"), CFG)


def test_feature_selector_uses_only_the_exact_b4_families_and_configured_window() -> None:
    table = make_table()
    selected = b4_feature_columns(table, CFG)
    assert selected == FEATURES
    assert not {
        "w7mean_wrong_window",
        "w3mean_outage_month",
        "w3std_n_intervals",
        "alarm_999_w3",
        "op_raw_mean",
        "max_raw_max",
        "StationId",
        "Day",
        "label",
        "status",
        "split",
        "outage_month",
        "suspect_wind",
        "n_intervals",
        "n_producing",
        "masked_frac",
    }.intersection(selected)


def test_constructor_supports_both_targets_and_rejects_unknown_target() -> None:
    for target in ("a", "b"):
        baseline = LogisticBaseline(target)
        assert baseline.name == "B4_logistic" and baseline.target == target
    with pytest.raises(ValueError, match="target"):
        LogisticBaseline("c")


def test_fit_reads_target_specific_settings_and_builds_balanced_l2_model() -> None:
    a, b = fitted("a"), fitted("b")
    assert a.selected_c_ == 0.5 and b.selected_c_ == 2.0
    assert a.model_.class_weight == "balanced"
    assert a.model_.penalty == "l2" and a.model_.solver == "liblinear"
    assert a.model_.max_iter == 1000 and a.model_.tol == 1e-4
    assert a.class_counts_ == {0: 20, 1: 20}


def test_score_follows_contract_is_float64_finite_and_does_not_modify_rows() -> None:
    table = make_table()
    before = table.copy(deep=True)
    baseline = fitted(table=table)
    scores = baseline.score(table)
    assert scores.index.equals(table.index)
    assert scores.dtype == np.float64 and np.isfinite(scores).all()
    assert check_score_contract(baseline, table) == []
    pd.testing.assert_frame_equal(table, before)


def test_separable_positive_rows_receive_higher_scores() -> None:
    table = make_table()
    train = labelled(table, "train")
    scores = fitted(table=table).score(train)
    assert scores.loc[train["label"] == 1].mean() > scores.loc[train["label"] == 0].mean() + 0.5


def test_missing_values_in_every_input_still_receive_a_finite_score() -> None:
    baseline = fitted()
    row = make_table().iloc[[45]].copy()
    row[FEATURES] = row[FEATURES].astype("float64")
    row.loc[:, FEATURES] = np.nan
    score = baseline.score(row)
    assert score.index.equals(row.index) and np.isfinite(score.iloc[0])


def test_fit_rejects_validation_rows_unlabelled_rows_and_one_class() -> None:
    table = make_table()
    with pytest.raises(ValueError, match="outside the train split"):
        LogisticBaseline("a").fit(table.loc[table["split"] == "val"], CFG)

    unlabelled = labelled(table, "train").copy()
    unlabelled.loc[unlabelled.index[0], "label"] = pd.NA
    with pytest.raises(ValueError, match="without a label"):
        LogisticBaseline("a").fit(unlabelled, CFG)

    one_class = labelled(table, "train").copy()
    one_class["label"] = 0
    with pytest.raises(ValueError, match="both label classes"):
        LogisticBaseline("a").fit(one_class, CFG)


def test_fit_is_isolated_from_non_train_rows() -> None:
    table = make_table()
    changed = table.copy()
    mask = changed["split"] != "train"
    changed.loc[mask, FEATURES] = changed.loc[mask, FEATURES] * 7 + 3
    changed.loc[mask, "label"] = 1
    scoring_rows = table.iloc[:25]
    a = fitted(table=table).score(scoring_rows)
    b = fitted(table=changed).score(scoring_rows)
    pd.testing.assert_series_equal(a, b)


def test_two_fits_are_deterministic() -> None:
    table = make_table()
    rows = table.iloc[10:50]
    pd.testing.assert_series_equal(fitted(table=table).score(rows), fitted(table=table).score(rows))


def test_coefficient_table_reports_ranked_positive_and_negative_attributions() -> None:
    baseline = fitted()
    result = baseline.coefficient_table(top_n=3)
    assert list(result.columns) == [
        "target",
        "C",
        "direction",
        "rank",
        "feature",
        "coefficient",
        "abs_coefficient",
    ]
    assert set(result["target"]) == {"A"} and set(result["C"]) == {0.5}
    assert set(result["feature"]) <= set(baseline.preprocessor_.get_feature_names_out())
    assert (result.loc[result["direction"] == "positive", "coefficient"] > 0).all()
    assert (result.loc[result["direction"] == "negative", "coefficient"] < 0).all()
    assert (result["abs_coefficient"] == result["coefficient"].abs()).all()
    assert result.groupby("direction").size().le(3).all()
    assert result.groupby("direction")["rank"].apply(list).map(
        lambda ranks: ranks == list(range(1, len(ranks) + 1))
    ).all()
    with pytest.raises(ValueError, match="positive"):
        baseline.coefficient_table(0)


def test_unfitted_model_cannot_score_or_report_coefficients() -> None:
    baseline = LogisticBaseline("a")
    with pytest.raises(RuntimeError, match="fit"):
        baseline.score(make_table().iloc[:1])
    with pytest.raises(RuntimeError, match="fit"):
        baseline.coefficient_table(2)


def test_invalid_frozen_model_settings_are_rejected() -> None:
    cfg = copy.deepcopy(CFG)
    cfg["baseline_b4"]["selected_c"]["a"] = 0
    with pytest.raises(ValueError, match="selected_c"):
        LogisticBaseline("a").fit(labelled(make_table(), "train"), cfg)
