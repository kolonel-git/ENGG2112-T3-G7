"""H0 selection uses train history only; data below is entirely synthetic."""

from __future__ import annotations

import copy
<<<<<<< HEAD
=======
import subprocess
import sys
from pathlib import Path
>>>>>>> 232aa6bb757fb5af3eb831e1b32c6b0d11515a8d

import numpy as np
import pandas as pd
import pytest

from wt_predmaint.config import load_protocol
from wt_predmaint.models.healthy import healthy_days, healthy_intervals


def tables() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    cfg = copy.deepcopy(load_protocol())
    days = pd.date_range("2020-01-01", "2020-01-31")
    labels = pd.DataFrame(
        {
            "StationId": 1,
            "Day": days,
            "status": "negative",
            "label": pd.array([0] * len(days), dtype="Int8"),
            "split": "train",
        }
    )
    features = pd.DataFrame(
        {"StationId": 1, "Day": days, "outage_month": False, "suspect_wind": False},
        index=np.arange(100, 100 + len(days)),
    )
    return labels, features, cfg


@pytest.mark.parametrize("target", ["a", "b"])
def test_recovery_window_excludes_exactly_n_days_after_episode(target: str) -> None:
    labels, features, cfg = tables()
    labels.loc[labels.Day.between("2020-01-09", "2020-01-10"), "status"] = "in_episode"
    labels.loc[labels.status.eq("in_episode"), "label"] = pd.NA
    mask = healthy_days(labels, features, cfg, target)
    assert mask.index.equals(features.index) and mask.dtype == bool
    # First seven days lack full train history; recovery excludes Jan 11–17.
    assert not mask.loc[features.Day.le("2020-01-07")].any()
    assert mask.loc[features.Day.eq("2020-01-08")].all()
    assert not mask.loc[features.Day.between("2020-01-09", "2020-01-17")].any()
    assert mask.loc[features.Day.ge("2020-01-18")].all()


def test_only_negatives_with_explicit_good_flags_qualify() -> None:
    labels, features, cfg = tables()
    cfg["healthy"]["recovery_days"] = 0
    for i, status in enumerate(
        ["positive", "zone", "purged", "censored", "low_coverage", "in_episode"]
    ):
        labels.loc[i, "status"] = status
    features.loc[106, "outage_month"] = True
    features.loc[107, "suspect_wind"] = True
    features["outage_month"] = features["outage_month"].astype("boolean")
    features.loc[108, "outage_month"] = pd.NA
    labels.loc[9, "label"] = pd.NA
    result = healthy_days(labels, features, cfg, "a")
    assert not result.iloc[:10].any() and result.iloc[10:].all()


def test_validation_test_spoofed_splits_and_2026_never_qualify() -> None:
    labels, features, cfg = tables()
    cfg["healthy"]["recovery_days"] = 0
    heldout = pd.to_datetime(["2021-02-01", "2023-02-01", "2026-02-01"])
    extra = pd.DataFrame(
        {"StationId": 1, "Day": heldout, "outage_month": False, "suspect_wind": False}
    )
    features = pd.concat([features, extra], ignore_index=True)
    # Even a falsely declared training split cannot override calendar membership.
    labels = pd.concat(
        [labels, extra.assign(status="negative", label=0, split="train")], ignore_index=True
    )
    cfg["splits"]["train_years"].append(2026)
    mask = healthy_days(labels, features, cfg, "a")
    assert not mask.iloc[-3:].any()
    changed = labels.copy()
    changed.loc[changed.Day.dt.year.isin([2021, 2023]), ["status", "label"]] = ["in_episode", 1]
    other_features = features.copy()
    other_features.loc[other_features.Day.dt.year.isin([2021, 2023]), "suspect_wind"] = True
    pd.testing.assert_series_equal(mask, healthy_days(changed, other_features, cfg, "a"))


def test_missing_history_and_other_turbines_do_not_supply_recovery() -> None:
    labels, features, cfg = tables()
    labels = labels.loc[~labels.Day.eq("2020-01-15")]
    labels = pd.concat([labels, labels.assign(StationId=2, status="in_episode")], ignore_index=True)
    mask = healthy_days(labels, features, cfg, "a")
    assert not mask.loc[features.Day.between("2020-01-15", "2020-01-22")].any()
    assert mask.loc[features.Day.eq("2020-01-23")].all()


def test_training_block_start_does_not_use_excluded_year_history() -> None:
    labels, features, cfg = tables()
    december = pd.DataFrame(
        {
            "StationId": 1,
            "Day": pd.date_range("2019-12-20", "2019-12-31"),
            "status": "negative",
            "label": 0,
            "split": "train",
        }
    )
    with_extra = pd.concat([december, labels], ignore_index=True)
    pd.testing.assert_series_equal(
        healthy_days(labels, features, cfg, "a"), healthy_days(with_extra, features, cfg, "a")
    )


def test_ten_minute_mask_handles_midnight_runs_missing_values_and_holdouts() -> None:
    labels, features, cfg = tables()
    cfg["healthy"]["recovery_days"] = 0
    stamps = pd.date_range("2020-01-10 23:00", periods=24, freq="10min")
    scada = pd.DataFrame(
        {
            "StationId": 1,
            "TimeStamp": stamps,
            "wtc_GenRpm_mean": 100.0,
            "wtc_PrWindSp_mean": 5.0,
            "outage_month": False,
            "suspect_wind": False,
        }
    )
    # A qualifying 18-interval stop run spans midnight. Six later producing rows survive.
    scada.loc[18:, "wtc_GenRpm_mean"] = 1200.0
    scada.loc[19, "wtc_PrWindSp_mean"] = 3.9
    scada.loc[20, "wtc_GenRpm_mean"] = np.nan
    scada.loc[21, "wtc_PrWindSp_mean"] = np.inf
    scada.loc[22, "suspect_wind"] = True
    scada.index = np.arange(200, 224)
    mask = healthy_intervals(labels, features, scada, cfg, "a")
    assert mask.index.equals(scada.index)
    assert mask[mask].index.tolist() == [218, 223]
    holdout = scada.iloc[[0]].assign(TimeStamp=pd.Timestamp("2023-01-01 00:10"))
    extended = pd.concat([scada, holdout], ignore_index=True)
    extended_mask = healthy_intervals(labels, features, extended, cfg, "a")
    assert not extended_mask.iloc[-1]
    np.testing.assert_array_equal(extended_mask.iloc[:-1], mask)


def test_midnight_interval_belongs_to_previous_day() -> None:
    labels, features, cfg = tables()
    cfg["healthy"]["recovery_days"] = 0
    features.loc[features.Day.eq("2020-01-11"), "outage_month"] = True
    scada = pd.DataFrame(
        {
            "StationId": 1,
            "TimeStamp": pd.to_datetime(["2020-01-11 00:00", "2020-01-11 00:10"]),
            "wtc_GenRpm_mean": 1200.0,
            "wtc_PrWindSp_mean": 4.0,
            "outage_month": False,
            "suspect_wind": False,
        }
    )
    assert healthy_intervals(labels, features, scada, cfg, "a").tolist() == [True, False]


def test_inputs_are_unchanged_and_invalid_contracts_are_rejected() -> None:
    labels, features, cfg = tables()
    before_labels, before_features = labels.copy(deep=True), features.copy(deep=True)
    healthy_days(labels, features, cfg, "a")
    pd.testing.assert_frame_equal(labels, before_labels)
    pd.testing.assert_frame_equal(features, before_features)
    with pytest.raises(ValueError, match="target"):
        healthy_days(labels, features, cfg, "c")
    with pytest.raises(ValueError, match="unique"):
        healthy_days(pd.concat([labels, labels.iloc[[0]]]), features, cfg, "a")
    with pytest.raises(ValueError, match="quality flags"):
        healthy_days(labels, features.drop(columns="outage_month"), cfg, "a")
<<<<<<< HEAD
=======


def test_qualifying_stop_run_crosses_training_year_boundary() -> None:
    _, _, cfg = tables()
    days = pd.date_range("2017-12-01", "2018-01-31")
    labels = pd.DataFrame(
        {"StationId": 1, "Day": days, "status": "negative", "label": 0, "split": "train"}
    )
    features = labels[["StationId", "Day"]].assign(outage_month=False, suspect_wind=False)
    scada = pd.DataFrame(
        {
            "StationId": 1,
            "TimeStamp": pd.date_range("2017-12-31 23:00", periods=18, freq="10min"),
            "wtc_GenRpm_mean": 100.0,
            "wtc_PrWindSp_mean": 5.0,
            "outage_month": False,
            "suspect_wind": False,
        }
    )
    mask = healthy_days(labels, features, cfg, "a")
    assert mask.loc[features.Day.isin(pd.to_datetime(["2017-12-31", "2018-01-01"]))].all()
    assert not healthy_intervals(labels, features, scada, cfg, "a").any()


def test_report_marks_unavailable_years_and_never_keeps_validation(tmp_path: Path) -> None:
    labels, features, _ = tables()
    extra = features.iloc[[0]].assign(Day=pd.Timestamp("2023-01-10"))
    features = pd.concat([features, extra], ignore_index=True)
    labels = pd.concat(
        [labels, labels.iloc[[0]].assign(Day=pd.Timestamp("2023-01-10"), split="test")],
        ignore_index=True,
    )
    interim = tmp_path / "interim"
    for folder in ("features", "labels", "clean"):
        (interim / folder).mkdir(parents=True)
    features.to_parquet(interim / "features/daily_features.parquet", index=False)
    labels.to_parquet(interim / "labels/a_labels.parquet", index=False)
    for year in (2020, 2023):
        pd.DataFrame(
            {
                "StationId": 1,
                "TimeStamp": pd.date_range(f"{year}-01-10 00:10", periods=2, freq="10min"),
                "wtc_GenRpm_mean": 1200.0,
                "wtc_PrWindSp_mean": 5.0,
                "outage_month": False,
                "suspect_wind": False,
            }
        ).to_parquet(interim / f"clean/scada_{year}.parquet", index=False)
    output = tmp_path / "counts.csv"
    command = [
        sys.executable,
        str(Path(__file__).resolve().parents[1] / "scripts/report_healthy_rows.py"),
        "--interim",
        str(interim),
        "--targets",
        "a",
        "--output",
        str(output),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False, timeout=30)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    report = pd.read_csv(output).set_index("year")
    assert report.index.tolist() == list(range(2016, 2026))
    assert report.loc[2020, "healthy_days"] == 24
    assert report.loc[2020, "healthy_intervals"] == 2
    assert report.loc[2023, "healthy_days"] == report.loc[2023, "healthy_intervals"] == 0
    assert pd.isna(report.loc[2016, "healthy_intervals"])
    assert not report.loc[2016, "clean_scada_file_available"]
>>>>>>> 232aa6bb757fb5af3eb831e1b32c6b0d11515a8d
