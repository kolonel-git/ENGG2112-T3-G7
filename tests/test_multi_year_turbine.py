"""Station isolation, local chronological protocol, and one-command evaluation."""

import copy
import json
import os
import runpy
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from wt_predmaint.config import load_protocol
from wt_predmaint.data.single_turbine_pipeline import experiment_config, read_station
from wt_predmaint.labels.horizon import build_horizon_labels

ROOT = Path(__file__).resolve().parents[1]


def test_daily_confusion_counts_and_undefined_precision():
    functions = runpy.run_path(str(ROOT / "scripts/multi_year/single_turbine/train_classifiers.py"))
    frame = pd.DataFrame({"label": [1, 0, 1, 0], "score": [0.9, 0.8, 0.2, 0.1]})
    result = functions["daily_metrics"](frame, 0.5)
    assert result["true_positive_days"] == result["false_positive_days"] == 1
    assert result["false_negative_days"] == result["true_negative_days"] == 1
    assert result["daily_precision"] == result["daily_recall"] == result["daily_f1"] == 0.5
    assert np.isnan(functions["daily_metrics"](frame, 1)["daily_precision"])


def test_sensor_rule_reference_is_not_refitted_on_test_values():
    functions = runpy.run_path(str(ROOT / "scripts/multi_year/single_turbine/train_classifiers.py"))
    column = "op_wtc_GearPres_mean"
    train = pd.DataFrame({column: [1., 2., 3., 4.]})
    rule = functions["sensor_rule_fit"](train, [column])
    before = rule["median"].copy()
    scores = functions["sensor_rule_score"](rule, pd.DataFrame({column: [2.5, 1000.]}))
    assert scores[0] == 0 and scores[1] > 100
    pd.testing.assert_series_equal(rule["median"], before)


def test_local_splits_and_station_coverage_do_not_modify_shared_configuration():
    cfg = load_protocol()
    original = copy.deepcopy(cfg)
    local = experiment_config(cfg, 2018, 2021, 2022, 2025)
    assert cfg == original
    assert local["splits"] == {"train_years": [2018, 2019, 2020, 2021],
                                "val_years": [2022], "test_years": [2023, 2024, 2025]}
    assert local["cleaning"]["expected_turbines"] == 1
    with pytest.raises(ValueError):
        experiment_config(cfg, 2018, 2023, 2022, 2025)


def test_chunk_reader_filters_other_turbines_before_parsing(tmp_path):
    path = tmp_path / "tblSCTurbine_2018_01.csv"
    pd.DataFrame({"StationId": [1, 2, 1], "TimeStamp": ["2018-01-01 00:10:00", "bad", "2018-01-01 00:20:00"],
                  "wtc_GenRpm_mean": [1000, 0, 1100]}).to_csv(path, index=False)
    result = read_station(path, 1, "scada", chunksize=1)
    assert result["StationId"].tolist() == [1, 1]
    assert result["TimeStamp"].dt.year.eq(2018).all()


def test_horizons_are_purged_at_training_calibration_and_test_boundaries():
    cfg = experiment_config(load_protocol(), 2018, 2021, 2022, 2025)
    days = pd.DataFrame({"StationId": 1, "Day": pd.date_range("2018-01-01", "2025-12-31")})
    ep = pd.DataFrame({"StationId": [1], "Start": pd.to_datetime(["2023-01-15"]),
                       "End": pd.to_datetime(["2023-01-15"])})
    labels = build_horizon_labels(days, ep, cfg)
    for split, next_start in (("train", "2022-01-01"), ("val", "2023-01-01")):
        scored = labels.loc[labels["split"].eq(split) & labels["label"].notna()]
        assert (scored["Day"] + pd.Timedelta(days=60) < pd.Timestamp(next_start)).all()


def test_one_command_training_and_plotting_with_prepared_data(tmp_path):
    cfg = experiment_config(load_protocol(), 2018, 2021, 2022, 2025)
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    days = pd.date_range("2018-01-01", "2025-12-31")
    rng = np.random.default_rng(42)
    features = pd.DataFrame({"StationId": 1, "Day": days,
                             "op_wtc_GearPres_mean": rng.normal(size=len(days)),
                             "op_wtc_GeOilTmp_mean": rng.normal(size=len(days)),
                             "frac_producing": rng.uniform(size=len(days))})
    features.to_parquet(prepared / "features.parquet")
    starts = pd.to_datetime([f"{year}-{month:02d}-15"
                             for year in range(2018, 2026) for month in (2, 5, 8, 11)])
    ep = pd.DataFrame({"StationId": 1, "Start": starts, "End": starts})
    labels = build_horizon_labels(features[["StationId", "Day"]], ep, cfg)
    for target in ("a", "b"):
        labels.to_parquet(prepared / f"{target}_labels.parquet")
        ep.to_parquet(prepared / f"{target}_episodes.parquet")
    (prepared / "preparation.json").write_text(json.dumps(
        {"station": 1, "protocol": cfg, "years": list(range(2018, 2026))}))
    environment = {**os.environ, "MPLCONFIGDIR": str(tmp_path / "mpl")}
    result = subprocess.run([sys.executable, str(ROOT / "scripts/multi_year/single_turbine/main.py"),
                              "--station", "1", "--skip-prepare", "--work-dir", str(tmp_path)],
                             capture_output=True, text=True, env=environment, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr
    scores = pd.read_csv(tmp_path / "results/predictions.csv", parse_dates=["Day"])
    assert scores["StationId"].unique().tolist() == [1]
    assert set(scores["Day"].dt.year) == {2023, 2024, 2025}
    assert scores["model"].nunique() == 5
    metrics = pd.read_csv(tmp_path / "results/metrics.csv")
    assert len(metrics.query("period == 'test_all'")) == 5
    assert scores["target"].unique().tolist() == ["A"]
    assert not list((tmp_path / "results").glob("b_*.joblib"))
    assert metrics["recall_lo"].isna().all()
    assert (tmp_path / "figures/A_model_comparison.png").stat().st_size > 1000


def test_main_stops_after_a_failed_stage(monkeypatch, tmp_path):
    namespace = runpy.run_path(str(ROOT / "scripts/multi_year/single_turbine/main.py"))
    calls = []
    def fail(command, **kwargs):
        calls.append(command)
        raise subprocess.CalledProcessError(7, command)
    monkeypatch.setattr(subprocess, "run", fail)
    monkeypatch.setattr(sys, "argv", ["main.py", "--work-dir", str(tmp_path)])
    with pytest.raises(SystemExit) as error:
        namespace["main"]()
    assert error.value.code == 7
    assert len(calls) == 1 and calls[0][1].endswith("prepare_data.py")
