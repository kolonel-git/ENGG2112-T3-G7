"""Synthetic tests for Target C timing, coverage, causal features and train-only fitting."""

import numpy as np
import pandas as pd
import pytest

from wt_predmaint.features.short_term import short_term_features
from wt_predmaint.labels.alarm_forecast import alarm_labels, assign_splits

T = pd.Timestamp


def coverage(start="2020-01-01", end="2020-02-01"):
    return pd.DataFrame({"StationId": [1], "Start": [T(start)], "End": [T(end)]})


def events(times, station=1):
    return pd.DataFrame({"StationId": [station] * len(times), "TimeOn": pd.to_datetime(times)})


def points(times, station=1):
    return pd.DataFrame({"StationId": station, "TimeStamp": pd.to_datetime(times)})


@pytest.mark.parametrize(
    "time,expected",
    [("13:03", 1), ("13:00", 0), ("19:00", 1), ("19:00:01", 0), ("12:59", 0)],
)
def test_exact_six_hour_window(time, expected):
    result = alarm_labels(points(["2020-01-02 13:00"]), events([f"2020-01-02 {time}"]), coverage())
    assert result.label.iloc[0] == expected


def test_missing_coverage_is_unknown_but_witnessed_alarm_is_positive():
    rows = points(["2020-01-02 13:00", "2020-01-03 13:00"])
    result = alarm_labels(rows, events(["2020-01-02 13:03"]), coverage(end="2020-01-02"))
    assert result.label.iloc[0] == 1
    assert pd.isna(result.label.iloc[1])
    assert not result.horizon_observed.any()


def test_alarm_and_coverage_are_turbine_specific():
    result = alarm_labels(
        points(["2020-01-02 13:00"], station=2), events(["2020-01-02 13:03"]), coverage()
    )
    assert pd.isna(result.label.iloc[0])


def test_coverage_merges_touching_intervals_but_preserves_gaps():
    intervals = pd.concat(
        [
            coverage("2020-01-02 13:00", "2020-01-02 16:00"),
            coverage("2020-01-02 16:00", "2020-01-02 19:00"),
        ],
        ignore_index=True,
    )
    rows = points(["2020-01-02 13:00"])
    assert alarm_labels(rows, events([]), intervals).label.iloc[0] == 0
    intervals.loc[1, "Start"] += pd.Timedelta(seconds=1)
    assert pd.isna(alarm_labels(rows, events([]), intervals).label.iloc[0])


def test_invalid_coverage_rejected():
    with pytest.raises(ValueError, match="Start < End"):
        alarm_labels(points(["2020-01-02"]), events([]), coverage(end="2019-01-01"))


def test_split_boundary_purging():
    rows = points(["2020-12-31 17:50", "2020-12-31 18:00", "2021-01-01 00:00"])
    labels = alarm_labels(rows, events([]), coverage(end="2022-01-01"))
    result = assign_splits(
        labels,
        {
            "train_years": [2020],
            "val_years": [2021],
            "test_years": [2022],
        },
    )
    assert result.label.iloc[0] == 0
    assert pd.isna(result.label.iloc[1])
    assert result.split.tolist() == ["train", "train", "val"]


def scada():
    frame = points(pd.date_range("2020-01-02 12:00", periods=13, freq="10min"))
    frame["sensor_mean"] = np.arange(len(frame), dtype=float)
    frame["outage_month"] = True
    return frame


def test_future_sensors_and_alarms_do_not_change_past_features():
    frame = scada()
    alarms = events(["2020-01-02 12:30", "2020-01-02 13:03"])
    cutoff = T("2020-01-02 13:00")
    full = short_term_features(frame, alarms).query("TimeStamp <= @cutoff")
    truncated = short_term_features(
        frame.loc[frame.TimeStamp <= cutoff], alarms.loc[alarms.TimeOn <= cutoff]
    )
    pd.testing.assert_frame_equal(full, truncated)
    row = full.iloc[-1]
    assert row["1h_alarm_count"] == 1
    assert row["minutes_since_alarm"] == 30
    assert row["current_sensor_mean"] == 6
    assert row["1h_scada_rows"] == 6
    assert "outage_month" not in full


def test_windows_use_elapsed_time_and_exclude_other_turbines():
    a = scada().iloc[[0, -1]]
    b = scada().assign(StationId=2, sensor_mean=9999.0)
    result = short_term_features(pd.concat([a, b]), events([]))
    row = result.loc[result.StationId == 1].iloc[-1]
    assert row["1h_mean_sensor_mean"] == 12
    assert row["1h_scada_rows"] == 1
    assert row["7d_scada_rows"] == 2
    assert pd.isna(row.minutes_since_alarm)


def test_duplicate_scada_keys_rejected():
    with pytest.raises(ValueError, match="unique"):
        short_term_features(pd.concat([scada(), scada()]), events([]))


def test_hourly_anchors_keep_all_ten_minute_history():
    frame = scada()
    alarms = events(["2020-01-02 12:33"])
    full = short_term_features(frame, alarms)
    hourly = short_term_features(frame, alarms, prediction_minutes=60)
    expected = full.loc[full.TimeStamp.dt.minute == 0].reset_index(drop=True)
    pd.testing.assert_frame_equal(hourly, expected)
    assert hourly.iloc[1]["1h_scada_rows"] == 6


def test_model_fits_imputer_only_on_train_and_returns_english_predictions():
    from scripts.run_target_c import fit_and_evaluate

    train = points(pd.date_range("2020-01-01", periods=8, freq="10min"))
    train["x"] = [0, 0, 0, np.nan, 1, 1, 1, 1]
    train["label"] = [0, 0, 0, 0, 1, 1, 1, 1]
    val = train.copy().assign(x=[1000, 1000, 1000, np.nan, 2000, 2000, 2000, 2000])
    model, columns, metrics, predictions = fit_and_evaluate(train, val)
    assert columns == ["x"]
    assert model.named_steps["simpleimputer"].statistics_[0] == 1
    assert set(predictions.predicted_alarm) <= {"yes", "no"}
    assert metrics["validation_rows"] == 8
    assert "label" not in columns


def test_build_and_model_end_to_end(tmp_path, monkeypatch):
    """Exercise raw alarms -> partitioned C tables -> model and English CSV exports."""
    import json
    import sys

    from scripts.build_target_c import main as build
    from scripts.run_target_c import main as run

    raw = tmp_path / "raw"
    clean = tmp_path / "interim" / "clean"
    raw.mkdir()
    clean.mkdir(parents=True)
    times = pd.date_range("2020-12-28", "2021-01-05", freq="10min")
    frame = points(times)
    frame["sensor_mean"] = np.sin(np.arange(len(frame)) / 13)
    for year, part in frame.groupby((frame.TimeStamp - pd.Timedelta(minutes=10)).dt.year):
        part.to_parquet(clean / f"scada_{year}.parquet", index=False)
    # Use a code outside the existing A/B shortlist to confirm 'any alarm'.
    alarm_times = pd.date_range("2020-12-28 13:03", "2021-01-04 13:03", freq="D")
    alarm_log = events(alarm_times).rename(columns={"StationId": "StationNr"})
    alarm_log["Alarmcode"] = 999999
    alarm_log["TimeOff"] = pd.NaT
    alarm_log.to_csv(raw / "tblAlarmLog.csv", index=False)
    cov = tmp_path / "coverage.csv"
    coverage("2020-12-28", "2021-01-05").to_csv(cov, index=False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_target_c.py",
            "--raw",
            str(raw),
            "--interim",
            str(clean.parent),
            "--coverage",
            str(cov),
        ],
    )
    build()
    partitions = list((clean.parent / "target_c").glob("*.parquet"))
    assert len(partitions) == 2
    table = pd.concat([pd.read_parquet(path) for path in partitions])
    row = table.loc[table.TimeStamp == T("2021-01-01 13:00")].iloc[0]
    assert row.label == 1
    assert row["7d_scada_rows"] > 144  # history carried across source-year boundary
    assert pd.isna(table.loc[table.TimeStamp == times[0], "7d_alarm_count"].iloc[0])
    out = tmp_path / "results"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_target_c.py",
            "--input",
            str(clean.parent / "target_c"),
            "--out",
            str(out),
        ],
    )
    run()
    predictions = pd.read_csv(out / "validation_predictions.csv")
    assert set(predictions.predicted_alarm) <= {"yes", "no"}
    assert pd.to_datetime(predictions.TimeStamp).dt.year.eq(2021).all()
    metrics = json.loads((out / "validation_metrics.json").read_text())
    assert metrics["validation_rows"] == len(predictions)
    assert (out / "model.joblib").exists()

    from scripts.evaluate_target_c_local import main as run_batched

    batched_out = tmp_path / "batched_results"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_target_c_local.py",
            "--input",
            str(clean.parent / "target_c"),
            "--out",
            str(batched_out),
            "--max-train-rows",
            "200",
        ],
    )
    run_batched()
    batched = json.loads((batched_out / "validation_metrics.json").read_text())
    assert batched["train_sample_rows"] == 200
    assert batched["logistic"]["rows"] == len(predictions)
    assert not batched["test_years_scored"]
    assert (batched_out / "validation_predictions.parquet").exists()
