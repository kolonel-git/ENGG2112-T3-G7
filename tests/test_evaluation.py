import numpy as np
import pandas as pd
import pytest

from wt_predmaint.config import load_protocol
from wt_predmaint.evaluation.metrics import (
    bootstrap_turbines,
    detect_episodes,
    evaluate,
    evaluation_row,
    pr_auc,
    threshold_for_budget,
)
from wt_predmaint.evaluation.splits import labelled_split, leave_one_turbine_out

# All data below is synthetic.
T = pd.Timestamp
CFG = load_protocol()
START = T("2020-02-15")  # window: 2020-01-16 .. 2020-02-14


def turbine_frame(station: int, alert_days: list[str], n_days: int = 61) -> pd.DataFrame:
    days = pd.date_range("2020-01-01", periods=n_days, freq="D")
    label = ((days >= START - pd.Timedelta(days=30)) & (days < START)).astype(int)
    rng = np.random.default_rng(station)
    score = rng.uniform(0.0, 0.5, size=n_days)
    score[days.isin(pd.to_datetime(alert_days))] = 0.9
    return pd.DataFrame({"StationId": station, "Day": days, "label": label, "score": score})


def episodes_for(stations: list[int]) -> pd.DataFrame:
    return pd.DataFrame({"StationId": stations, "Start": START, "End": START})


def test_pr_auc_perfect_and_constant() -> None:
    y = np.array([0, 0, 1, 1])
    assert pr_auc(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert pr_auc(y, np.full(4, 0.5)) == pytest.approx(y.mean())
    assert np.isnan(pr_auc(np.zeros(4), np.arange(4.0)))


def test_threshold_respects_budget_and_ties() -> None:
    label = np.zeros(61, dtype=int)  # 61 days = 2.004 turbine-months, so 2 false alarms allowed
    score = np.arange(61, dtype=float)
    thr = threshold_for_budget(label, score, 1, 30.4375)
    assert (score > thr).sum() == 2
    tied = threshold_for_budget(label, np.full(61, 0.3), 1, 30.4375)
    assert (np.full(61, 0.3) > tied).sum() == 0  # ties are all in or all out
    assert threshold_for_budget(np.zeros(10, dtype=int), np.arange(10.0), 50, 30.4375) == -np.inf


def test_detection_and_lead_time() -> None:
    frame = turbine_frame(1, ["2020-02-05", "2020-02-10", "2020-01-10"])
    det = detect_episodes(frame, episodes_for([1]), threshold=0.6, positive_days=30)
    assert det.loc[0, "evaluable"] and det.loc[0, "detected"]
    assert det.loc[0, "lead_days"] == 10  # first alert in the window is 2020-02-05, start is 02-15


def test_alert_outside_window_does_not_detect() -> None:
    frame = turbine_frame(1, ["2020-01-10", "2020-02-20"])  # before the window and after the start
    det = detect_episodes(frame, episodes_for([1]), threshold=0.6, positive_days=30)
    assert det.loc[0, "evaluable"] and not det.loc[0, "detected"]
    assert np.isnan(det.loc[0, "lead_days"])


def test_episode_without_scored_days_is_not_evaluable() -> None:
    frame = turbine_frame(1, [], n_days=10)  # data stops long before the window
    det = detect_episodes(frame, episodes_for([1]), threshold=0.6, positive_days=30)
    assert not det.loc[0, "evaluable"]
    metrics = evaluate(frame, episodes_for([1]), 0.6, CFG)
    assert metrics["n_episodes"] == 0 and np.isnan(metrics["recall"])


def test_evaluate_counts_false_alarms_on_negative_rows_only() -> None:
    # one alert inside the window (not a false alarm) and two on negative days
    frame = turbine_frame(1, ["2020-02-05", "2020-01-03", "2020-03-01"])
    m = evaluate(frame, episodes_for([1]), 0.6, CFG)
    assert m["false_alarms_per_turbine_month"] == pytest.approx(2 / (61 / 30.4375))
    assert m["recall"] == 1.0 and m["n_detected"] == 1 and m["median_lead_days"] == 10
    assert m["prevalence"] == pytest.approx(30 / 61)


def test_budget_threshold_then_evaluate_keeps_false_alarms_within_budget() -> None:
    frame = turbine_frame(1, ["2020-02-05"])
    thr = threshold_for_budget(frame["label"].to_numpy(), frame["score"].to_numpy(), 1, 30.4375)
    m = evaluate(frame, episodes_for([1]), thr, CFG)
    assert m["false_alarms_per_turbine_month"] <= 1.0


def test_bootstrap_is_seeded_and_collapses_for_identical_turbines() -> None:
    frame = pd.concat([turbine_frame(s, ["2020-02-05"]) for s in (1, 2, 3)])
    # same rows for every turbine, so every resample gives the same recall and lead time
    frame["score"] = frame.groupby("Day")["score"].transform("first")
    eps = episodes_for([1, 2, 3])
    a = bootstrap_turbines(frame, eps, 0.6, CFG, resamples=50)
    b = bootstrap_turbines(frame, eps, 0.6, CFG, resamples=50)
    assert a == b
    assert a["recall"] == (1.0, 1.0)
    assert a["median_lead_days"] == (10.0, 10.0)


def test_bootstrap_interval_contains_point_estimate_and_row_has_all_columns() -> None:
    frames = []
    for s in range(1, 9):
        alerts = ["2020-02-05"] if s % 2 else ["2020-01-03"]
        frames.append(turbine_frame(s, alerts))
    frame = pd.concat(frames)
    eps = episodes_for(list(range(1, 9)))
    row = evaluation_row("demo", frame, eps, 0.6, CFG, resamples=200)
    assert row["recall"] == 0.5
    assert row["recall_lo"] <= row["recall"] <= row["recall_hi"]
    assert {"pr_auc_lo", "pr_auc_hi", "median_lead_days_lo", "model", "threshold"} <= set(row)


def test_scored_frame_rejects_unlabelled_or_nan_scores() -> None:
    frame = turbine_frame(1, [])
    bad = frame.copy()
    bad.loc[0, "label"] = np.nan
    with pytest.raises(ValueError):
        evaluate(bad, episodes_for([1]), 0.6, CFG)
    bad = frame.copy()
    bad.loc[0, "score"] = np.nan
    with pytest.raises(ValueError):
        evaluate(bad, episodes_for([1]), 0.6, CFG)


def test_labelled_split_and_leave_one_turbine_out() -> None:
    labels = pd.DataFrame(
        {
            "StationId": [1, 1, 2],
            "Day": pd.to_datetime(["2020-01-01", "2020-01-02", "2021-01-01"]),
            "label": pd.array([1, pd.NA, 0], dtype="Int8"),
            "split": ["train", "train", "val"],
        }
    )
    train = labelled_split(labels, "train")
    assert len(train) == 1 and train["label"].dtype == np.int64
    folds = list(leave_one_turbine_out([3, 1, 2, 2]))
    assert [h for _, h in folds] == [1, 2, 3]
    assert all(h not in tr and len(tr) == 2 for tr, h in folds)
    with pytest.raises(ValueError):
        labelled_split(labels, "dev")
