"""Temporal separation, training-only preprocessing, and runnable output contracts."""

import json
import os
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from wt_predmaint.models.single_year import fit_classifier, risk_scores, training_folds

ROOT = Path(__file__).resolve().parents[1]


def test_train_calibration_and_test_horizons_do_not_overlap():
    days = pd.date_range("2017-12-01", "2019-02-01")
    data = pd.concat([pd.DataFrame({"StationId": s, "Day": days, "label": 0})
                      for s in (1, 2)])
    used_test = set()
    for _, start, end, train, calibration, test in training_folds(data, 2018, 60):
        assert (train["Day"] + pd.Timedelta(days=60) < calibration["Day"].min()).all()
        assert (calibration["Day"] + pd.Timedelta(days=60) < start).all()
        assert test["Day"].between(start, end, inclusive="left").all()
        assert (test["Day"] + pd.Timedelta(days=60) < pd.Timestamp("2019-01-01")).all()
        assert train["Day"].dt.year.eq(2018).all()
        assert not used_test.intersection(test["Day"])
        used_test.update(test["Day"])
        assert test.groupby("StationId")["Day"].apply(list).iloc[0] == (
            test.groupby("StationId")["Day"].apply(list).iloc[1])


def test_preprocessing_and_feature_selection_use_training_only():
    train = pd.DataFrame({"label": [0, 0, 1, 1], "sensor": [1., np.nan, 3., 4.],
                          "constant": [5.] * 4, "absent": [np.nan] * 4})
    model, columns, audit = fit_classifier(train, ["sensor", "constant", "absent"], 42)
    assert columns == ["sensor"]
    assert model["imputer"].statistics_[0] == 3
    assert audit["used"].tolist() == [True, False, False]
    test = pd.DataFrame({"sensor": [np.nan, 100000.]})
    before = model["scaler"].mean_.copy()
    scores = risk_scores(model, test, columns)
    assert np.isfinite(scores).all() and ((scores >= 0) & (scores <= 1)).all()
    np.testing.assert_array_equal(model["scaler"].mean_, before)
    np.testing.assert_array_equal(scores, risk_scores(model, test, columns))
    with pytest.raises(ValueError, match="both positive and negative"):
        fit_classifier(train.assign(label=0), ["sensor"], 42)


@pytest.mark.parametrize("scope", ["all_turbines", "single_turbine"])
def test_training_and_plotting_end_to_end(tmp_path, scope):
    interim = tmp_path / "interim"
    (interim / "features").mkdir(parents=True)
    (interim / "labels").mkdir()
    days = pd.date_range("2018-01-01", "2018-12-31")
    rng = np.random.default_rng(123)
    features = pd.concat([
        pd.DataFrame({"StationId": s, "Day": days,
                      "op_sensor_mean": rng.normal(size=len(days)),
                      "frac_producing": rng.uniform(size=len(days)),
                      "outage_month": False, "n_intervals": 144})
        for s in (1, 2)
    ], ignore_index=True)
    features.to_parquet(interim / "features/daily_features.parquet")
    labels = features[["StationId", "Day"]].assign(
        label=(features["Day"].dt.day % 2).astype(int), split="train")
    episodes = pd.DataFrame({"StationId": [1, 2, 1, 2],
                             "Start": pd.to_datetime(["2018-08-20", "2018-09-10",
                                                      "2018-10-20", "2018-11-10"])})
    for target in ("a", "b"):
        labels.to_parquet(interim / f"labels/{target}_labels.parquet")
        episodes.to_parquet(interim / f"labels/{target}_episodes.parquet")
    output, figures = tmp_path / "output", tmp_path / "figures"
    environment = {**os.environ, "MPLCONFIGDIR": str(tmp_path / "mpl")}
    for script, arguments in (
        ("train_classifier.py", ["--interim", str(interim), "--output-dir", str(output),
                                 "--resamples", "5", *(["--station", "1"] if scope == "single_turbine" else [])]),
        ("plot_classifier.py", ["--input-dir", str(output), "--output-dir", str(figures),
                                "--station", "1"]),
    ):
        subprocess.run([sys.executable, str(ROOT / "scripts/single_year" / scope / script), *arguments],
                       check=True, capture_output=True, text=True, env=environment, timeout=180)
    result = pd.read_csv(output / "metrics.csv")
    assert len(result) == 18  # 2 targets x 3 folds x 3 models
    if scope == "single_turbine":
        assert result["recall_lo"].isna().all()
        assert result["recall_hi"].isna().all()
    scores = pd.read_csv(output / "predictions.csv")
    if scope == "single_turbine":
        assert scores["StationId"].unique().tolist() == [1]
        assert saved_station(output) == 1
    for _, group in scores.groupby(["target", "fold"]):
        assert group.groupby("model").size().nunique() == 1
    saved = joblib.load(output / "a_fold1.joblib")
    assert "threshold" in saved and "pipeline" in saved and "features" in saved
    if scope == "single_turbine":
        own = labels.loc[labels["StationId"] == 1].merge(features, on=["StationId", "Day"])
        own_train = next(training_folds(own, 2018, 60))[3]
        np.testing.assert_allclose(saved["pipeline"]["imputer"].statistics_,
                                   own_train[saved["features"]].median().to_numpy())
    assert json.loads((output / "run.json").read_text())["year"] == 2018
    assert (figures / "A_fold1_diagnostics.png").stat().st_size > 1000
    assert (figures / "A_turbine_1_timeline.png").is_file()


def saved_station(output):
    return json.loads((output / "run.json").read_text())["station"]
