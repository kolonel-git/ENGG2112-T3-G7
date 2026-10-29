"""Check temporal separation in the exploratory within-year baseline runner."""

import runpy
from pathlib import Path

import pandas as pd

folds = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "scripts/single_year/run_year_baselines.py")
)["chronological_folds"]


def test_folds_purge_full_horizons_and_share_dates_across_turbines():
    days = pd.date_range("2017-12-01", "2019-02-01")
    labels = pd.concat([
        pd.DataFrame({"StationId": station, "Day": days, "label": 0})
        for station in (1, 2)
    ])
    previous_test_days = set()
    for _, start, end, calibration, test in folds(labels, 2018, 60):
        assert not calibration.empty and not test.empty
        assert (calibration["Day"] + pd.Timedelta(days=60) < start).all()
        assert (test["Day"] >= start).all()
        assert (test["Day"] + pd.Timedelta(days=60) < end).all()
        assert calibration["Day"].dt.year.eq(2018).all()
        test_days = set(test["Day"])
        assert not previous_test_days.intersection(test_days)
        previous_test_days.update(test_days)
        assert set(test.loc[test["StationId"] == 1, "Day"]) == set(
            test.loc[test["StationId"] == 2, "Day"]
        )


def test_folds_exclude_unlabelled_days():
    labels = pd.DataFrame({
        "StationId": [1], "Day": [pd.Timestamp("2018-04-01")], "label": [pd.NA],
    })
    assert all(calibration.empty and test.empty
               for _, _, _, calibration, test in folds(labels, 2018, 60))
