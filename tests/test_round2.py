"""Self-baseline features and the turbine-ranking metric on synthetic data (D-065)."""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from wt_predmaint.config import load_protocol
from wt_predmaint.evaluation.ranking import daily_precision_at_k, ranking_summary
from wt_predmaint.features.relative import build_feature_set, self_relative

CFG = copy.deepcopy(load_protocol())
CFG["labels"]["feature_window_days"] = 7
CFG["exploratory"]["min_turbines"] = 3
CFG["exploratory_round2"].update(window_days=10, min_days=5)
SENS = "w7mean_op_wtc_GeOilTmp_mean"


def make_table(n: int = 60, stations: int = 4, seed: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    parts = []
    for station in range(1, stations + 1):
        parts.append(
            pd.DataFrame(
                {
                    "StationId": station,
                    "Day": pd.date_range("2020-01-01", periods=n),
                    SENS: 20.0 + station + 0.2 * np.arange(n) + rng.normal(0, 0.1, n),
                    "alarm_total_w7": rng.integers(0, 3, n),
                }
            )
        )
    return pd.concat(parts, ignore_index=True)


def test_self_baseline_uses_only_days_before_and_needs_min_days() -> None:
    table = make_table()
    rel = self_relative(table, [SENS], window=10, min_days=5)[SENS]
    first = table["Day"] == table["Day"].min()
    assert rel[first].isna().all()  # no history yet
    # a slow trend shows up as a positive gap of about the trend over half the window
    late = table["Day"] >= table["Day"].min() + pd.Timedelta(days=20)
    assert rel[late].between(0.5, 3.5).all()


def test_a_later_day_cannot_change_an_earlier_self_value_or_its_own() -> None:
    table = make_table()
    before = self_relative(table, [SENS], 10, 5)[SENS]
    changed = table.copy()
    cut = pd.Timestamp("2020-02-10")
    changed.loc[changed["Day"] >= cut, SENS] += 500.0
    after = self_relative(changed, [SENS], 10, 5)[SENS]
    early = table["Day"] < cut
    assert np.allclose(before[early], after[early], equal_nan=True)
    # on the day of the jump the baseline still excludes that day, so the gap is the jump itself
    jump = table["Day"] == cut
    assert (after[jump] - before[jump]).between(499, 501).all()


def test_gaps_are_missing_days_not_skipped_rows() -> None:
    table = make_table(n=40)
    sparse = table.loc[
        ~((table["StationId"] == 1) & table["Day"].between("2020-01-12", "2020-01-30"))
    ]
    rel = self_relative(sparse.reset_index(drop=True), [SENS], 10, 5)[SENS]
    after_gap = sparse.reset_index(drop=True)
    mask = (after_gap["StationId"] == 1) & (after_gap["Day"] == pd.Timestamp("2020-02-02"))
    assert rel[mask].isna().all()  # fewer than min_days observed in the 10 calendar days


def test_feature_sets_self_and_self_fleet_keep_names_and_other_columns() -> None:
    table = make_table()
    for name in ("self", "self_fleet"):
        out = build_feature_set(table, CFG, name)
        assert list(out.columns) == list(table.columns)
        assert out["alarm_total_w7"].equals(table["alarm_total_w7"])
        assert out[SENS].isna().any() and out[SENS].notna().any()


def make_scores() -> pd.DataFrame:
    rng = np.random.default_rng(3)
    days = pd.date_range("2021-01-01", periods=90)
    rows = []
    for day in days:
        for station in range(1, 13):
            label = int(station <= 4)  # turbines 1..4 are positive every day
            rows.append((station, day, label, float(label) * 2 + rng.normal(0, 0.1)))
    return pd.DataFrame(rows, columns=["StationId", "Day", "label", "score"])


def test_a_perfect_score_has_full_precision_and_a_random_score_has_the_chance_level() -> None:
    frame = make_scores()
    daily = daily_precision_at_k(frame, [3], min_turbines=10, seed=1)
    assert (daily["p_at_3"] == 1.0).all()
    assert daily["chance"].iloc[0] == pytest.approx(4 / 12)
    constant = frame.assign(score=0.0)
    daily0 = daily_precision_at_k(constant, [3], min_turbines=10, seed=1)
    assert daily0["p_at_3"].mean() == pytest.approx(4 / 12, abs=0.08)


def test_days_with_few_labelled_turbines_are_skipped_and_summary_has_an_interval() -> None:
    frame = make_scores()
    few = pd.concat([frame, frame.loc[frame["Day"] == frame["Day"].max()].iloc[:0]])
    few = few.loc[~((few["Day"] == few["Day"].min()) & (few["StationId"] > 5))]
    daily = daily_precision_at_k(few, [3, 5], min_turbines=10, seed=1)
    assert len(daily) == frame["Day"].nunique() - 1
    summary = ranking_summary(daily, [3, 5], resamples=200, seed=1)
    assert summary["k"].tolist() == [3, 5]
    assert (summary["difference_lo"] <= summary["difference"]).all()
    assert (summary["difference"] <= summary["difference_hi"]).all()
    assert summary.loc[0, "difference"] == pytest.approx(1 - 4 / 12)


def test_ranking_needs_the_columns() -> None:
    with pytest.raises(ValueError, match="needs"):
        daily_precision_at_k(pd.DataFrame({"Day": []}), [3], 10, 1)
