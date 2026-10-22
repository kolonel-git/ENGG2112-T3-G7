"""Synthetic recovery, train isolation, causal features and score contract for B3."""

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from wt_predmaint.config import load_protocol
from wt_predmaint.models.base import check_score_contract
from wt_predmaint.models.power_curve import PowerCurveProxy


def sources():
    cfg = deepcopy(load_protocol())
    cfg["healthy"]["recovery_days"] = 0
    cfg["power_curve"]["min_bin_intervals"] = 1
    days = pd.date_range("2018-01-01", periods=12)
    train = pd.DataFrame(
        {
            "StationId": 1,
            "Day": days,
            "split": "train",
            "status": "negative",
            "label": 0,
            "outage_month": False,
            "suspect_wind": False,
        }
    )
    stamp = pd.date_range("2018-01-01 00:10", periods=12 * 144, freq="10min")
    wind = np.tile([4.25, 4.75, 5.25, 5.75], len(stamp) // 4)
    scada = pd.DataFrame(
        {
            "StationId": 1,
            "TimeStamp": stamp,
            "wtc_PrWindSp_mean": wind,
            "wtc_GenRpm_mean": 200 * wind,
            "outage_month": False,
            "suspect_wind": False,
        }
    )
    return cfg, train, scada


def test_recovers_curve_and_underperformance():
    cfg, train, scada = sources()
    model = PowerCurveProxy("a", scada=scada, labels=train).fit(train, cfg)
    np.testing.assert_allclose(model.curve.rpm, 200 * model.curve.wind)
    assert model.score(train).eq(0).all()
    bad = scada.copy()
    bad["wtc_GenRpm_mean"] -= 300
    stats = model.interval_daily(bad)
    assert (stats.exceed_intervals == stats.valid_intervals).all()
    assert (stats.shortfall_sum > 0).all()
    assert model.window_daily(stats, cfg).b3_score.iloc[-1] == 1
    assert check_score_contract(model, train) == []


def test_train_only_and_quality_and_future_invariance():
    cfg, train, scada = sources()
    future = scada.copy()
    future.TimeStamp += pd.DateOffset(years=3)
    future["wtc_GenRpm_mean"] = 0
    mixed = pd.concat([scada, future], ignore_index=True)
    a = PowerCurveProxy("a", scada=scada, labels=train).fit(train, cfg)
    b = PowerCurveProxy("a", scada=mixed, labels=train).fit(train, cfg)
    pd.testing.assert_frame_equal(a.curve, b.curve)
    assert a.tolerance == b.tolerance
    np.testing.assert_allclose(a.score(train), b.score(train))
    flagged = scada.copy()
    flagged["suspect_wind"] = True
    assert a.interval_daily(flagged).valid_intervals.eq(0).all()
    with pytest.raises(ValueError):
        a.fit(train.assign(split="val"), cfg)


def test_midnight_calendar_and_causal_window():
    cfg, train, scada = sources()
    model = PowerCurveProxy("b", scada=scada, labels=train).fit(train, cfg)
    scada = scada.copy()
    scada.loc[scada.TimeStamp >= "2018-01-09", "wtc_GenRpm_mean"] = 0
    full = model.window_daily(model.interval_daily(scada), cfg).set_index(["StationId", "Day"])
    for day in pd.date_range("2018-01-04", "2018-01-10"):
        cut = scada.loc[scada.TimeStamp <= day + pd.Timedelta(days=1)]
        actual = model.window_daily(model.interval_daily(cut), cfg).set_index(["StationId", "Day"])
        pd.testing.assert_series_equal(full.loc[(1, day)], actual.loc[(1, day)])
    midnight = scada.loc[scada.TimeStamp == "2018-01-02"]
    assert model.interval_daily(midnight).Day.iloc[0] == pd.Timestamp("2018-01-01")


def test_missing_evidence_and_calendar_gaps():
    cfg, train, scada = sources()
    model = PowerCurveProxy("a", scada=scada, labels=train).fit(train, cfg)
    missing = train.iloc[[0]].copy()
    missing["Day"] = pd.Timestamp("2026-01-01")
    assert model.score(missing).iloc[0] == 0
    sparse = model.interval_daily(scada)
    sparse = sparse.iloc[[0, 1, 2, 10]].copy()
    daily = model.window_daily(sparse, cfg)
    assert not daily.b3_available.any()  # four observed days are outside a seven-day window
    broken = scada.copy()
    broken["outage_month"] = True
    with pytest.raises(ValueError, match="No healthy"):
        PowerCurveProxy("a", scada=broken, labels=train).fit(train, cfg)


def test_test_years_are_loaded_only_when_explicitly_scored():
    cfg, train, scada = sources()
    future = scada.copy()
    future.TimeStamp += pd.DateOffset(years=5)
    future["wtc_GenRpm_mean"] = 0
    model = PowerCurveProxy("a", scada=pd.concat([scada, future]), labels=train).fit(train, cfg)
    assert model.daily.Day.dt.year.max() <= 2022
    rows = train.copy()
    rows.Day += pd.DateOffset(years=5)
    assert model.score(rows).iloc[-1] == 1
    assert model.tolerance == 0
