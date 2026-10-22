import numpy as np
import pandas as pd
import pytest

from wt_predmaint.config import load_protocol
from wt_predmaint.features import leakage
from wt_predmaint.features.daily import (
    add_window_features,
    alarm_window_features,
    assemble_features,
    daily_aggregates,
    day_of_interval,
    fault_history_features,
    feature_columns,
)
from wt_predmaint.features.leakage import check_no_lookahead, truncate_sources

# All data below is synthetic.
CFG = load_protocol()
T = pd.Timestamp


def scada_days(n_days: int = 20, station: int = 1, start: str = "2020-03-01") -> pd.DataFrame:
    """Cleaned-style 10-min SCADA; intervals end at 00:10 .. 24:00 of each day."""
    ts = pd.date_range(T(start) + pd.Timedelta(minutes=10), periods=144 * n_days, freq="10min")
    rng = np.random.default_rng(station)
    n = len(ts)
    day_index = np.repeat(np.arange(n_days), 144)
    return pd.DataFrame(
        {
            "StationId": station,
            "TimeStamp": ts,
            "wtc_GenRpm_mean": np.where(rng.uniform(size=n) < 0.7, 1500.0, 50.0),
            "wtc_PrWindSp_mean": rng.uniform(2, 12, n),
            "wtc_GeOilTmp_mean": 40 + day_index + rng.normal(0, 0.1, n),  # trends up by day
            "wtc_GeOilTmp_max": 45 + day_index + rng.normal(0, 0.1, n),
            "n_masked": 0,
            "outage_month": False,
            "suspect_wind": False,
        }
    )


def summary_days(n_days: int = 20, station: int = 1, start: str = "2020-03-01") -> pd.DataFrame:
    days = pd.date_range(start, periods=n_days, freq="D")
    err = np.zeros(n_days)
    err[[5, 6, 14]] = 6.0  # fault days
    return pd.DataFrame(
        {
            "StationId": station,
            "TimeStamp": days,
            "TimeErr": err,
            "partial_day": False,
            "bad_hours": False,
        }
    )


def alarm_events(station: int = 1) -> pd.DataFrame:
    # a 3130 alarm whose interval ends exactly at midnight, one mid-day, one 20 on a later day
    return pd.DataFrame(
        {
            "StationId": station,
            "Alarmcode": [3130, 3130, 20],
            "TimeStamp": pd.to_datetime(
                ["2020-03-05 00:00", "2020-03-05 12:00", "2020-03-10 06:00"]
            ),
        }
    )


def test_interval_end_at_midnight_belongs_to_the_day_that_just_ended() -> None:
    ts = pd.Series(pd.to_datetime(["2020-03-05 00:00", "2020-03-05 00:10", "2020-03-06 00:00"]))
    days = day_of_interval(ts)
    assert days.tolist() == [T("2020-03-04"), T("2020-03-05"), T("2020-03-05")]


def test_daily_aggregates_shape_and_state_fractions() -> None:
    scada = scada_days(3)
    daily = daily_aggregates(scada, CFG)
    assert len(daily) == 3 and (daily["n_intervals"] == 144).all()
    assert daily["Day"].tolist() == list(pd.date_range("2020-03-01", periods=3))
    prod = (scada["wtc_GenRpm_mean"] >= 1000).to_numpy().reshape(3, 144).mean(axis=1)
    assert daily["frac_producing"].to_numpy() == pytest.approx(prod, abs=1e-6)
    assert {"op_wtc_GeOilTmp_mean", "max_wtc_GeOilTmp_max"} <= set(daily.columns)


def test_operating_means_use_producing_intervals_only_and_need_enough_of_them() -> None:
    scada = scada_days(2)
    scada.loc[scada["wtc_GenRpm_mean"] < 1000, "wtc_GeOilTmp_mean"] = 999.0  # only when idle
    daily = daily_aggregates(scada, CFG)
    assert (daily["op_wtc_GeOilTmp_mean"] < 100).all()
    sparse = scada.copy()
    sparse["wtc_GenRpm_mean"] = 50.0  # no producing intervals at all
    sparse.loc[:5, "wtc_GenRpm_mean"] = 1500.0  # six, below the 12 required
    assert daily_aggregates(sparse, CFG)["op_wtc_GeOilTmp_mean"].isna().all()


def test_window_features_are_trailing_and_respect_min_days() -> None:
    daily = daily_aggregates(scada_days(12), CFG)
    out = add_window_features(daily, CFG).set_index("Day")
    col = "op_wtc_GeOilTmp_mean"
    w = out["w7mean_" + col]
    assert w.iloc[:3].isna().all()  # fewer than window_min_days observed days
    assert w.iloc[6] == pytest.approx(out[col].iloc[0:7].mean(), rel=1e-5)
    assert w.iloc[11] == pytest.approx(out[col].iloc[5:12].mean(), rel=1e-5)


def test_window_does_not_bridge_missing_days_as_if_adjacent() -> None:
    daily = daily_aggregates(scada_days(12), CFG)
    gap = daily.drop(index=[4, 5, 6, 7])  # days 5 to 8 missing
    out = add_window_features(gap, CFG).set_index("Day")
    last = out.loc["2020-03-12", "w7mean_op_wtc_GeOilTmp_mean"]  # window covers days 6 to 12
    assert last == pytest.approx(daily.loc[[8, 9, 10, 11], "op_wtc_GeOilTmp_mean"].mean(), rel=1e-5)


def test_alarm_counts_use_ceil_convention_and_trailing_window() -> None:
    days = pd.DataFrame({"StationId": 1, "Day": pd.date_range("2020-03-01", periods=20)})
    out = alarm_window_features(alarm_events(), days, CFG).set_index("Day")
    # TimeStamp 2020-03-05 00:00 is the end of an interval on 03-04, 12:00 falls on 03-05
    assert out.loc["2020-03-04", "alarm_3130_w7"] == 1
    assert out.loc["2020-03-05", "alarm_3130_w7"] == 2
    assert out.loc["2020-03-11", "alarm_3130_w7"] == 1  # 03-04 dropped out of the window
    assert out.loc["2020-03-12", "alarm_3130_w7"] == 0
    assert out.loc["2020-03-10", "alarm_20_w7"] == 1 and out.loc["2020-03-09", "alarm_20_w7"] == 0
    assert (
        out.loc["2020-03-05", "alarm_tier1_w7"] == 2
        and out.loc["2020-03-05", "alarm_total_w7"] == 2
    )


def test_fault_history_counts_only_the_past() -> None:
    out = fault_history_features(summary_days(), CFG).set_index("Day")
    assert np.isnan(out.loc["2020-03-05", "days_since_fault_day"])  # first fault is day 6
    assert out.loc["2020-03-06", "days_since_fault_day"] == 0
    assert out.loc["2020-03-07", "days_since_fault_day"] == 0
    assert out.loc["2020-03-10", "days_since_fault_day"] == 3
    assert (
        out.loc["2020-03-14", "fault_days_30"] == 2 and out.loc["2020-03-15", "fault_days_30"] == 3
    )


def test_feature_columns_exclude_keys_flags_and_counts() -> None:
    table = assemble_features(
        daily_aggregates(scada_days(), CFG), alarm_events(), summary_days(), CFG
    )
    cols = feature_columns(table)
    assert not {"StationId", "Day", "outage_month", "suspect_wind", "n_intervals"} & set(cols)
    assert "w7mean_op_wtc_GeOilTmp_mean" in cols and "alarm_total_w7" in cols
    assert "days_since_fault_day" in cols


def test_no_lookahead_in_the_real_pipeline() -> None:
    scada, events, summary = scada_days(), alarm_events(), summary_days()
    cases = [(1, T(d)) for d in ("2020-03-08", "2020-03-12", "2020-03-15", "2020-03-19")]
    assert check_no_lookahead(scada, events, summary, CFG, cases) == []


def test_leakage_check_catches_a_feature_that_looks_ahead(monkeypatch: pytest.MonkeyPatch) -> None:
    real = leakage.assemble_features

    def leaky(daily, events, summary, cfg):  # adds tomorrow's temperature as a "feature"
        table = real(daily, events, summary, cfg)
        table["tomorrow"] = table["op_wtc_GeOilTmp_mean"].shift(-1)
        return table

    monkeypatch.setattr(leakage, "assemble_features", leaky)
    scada, events, summary = scada_days(), alarm_events(), summary_days()
    bad = check_no_lookahead(scada, events, summary, CFG, [(1, T("2020-03-12"))])
    assert len(bad) == 1 and "tomorrow" in bad[0]["columns"]


def test_truncate_sources_keeps_exactly_what_exists_at_end_of_day() -> None:
    scada, events, summary = scada_days(), alarm_events(), summary_days()
    s, e, d = truncate_sources(scada, events, summary, T("2020-03-05"))
    assert s["TimeStamp"].max() == T("2020-03-06 00:00")  # last interval of 03-05 ends at midnight
    assert e["TimeStamp"].max() == T("2020-03-05 12:00")
    assert d["TimeStamp"].max() == T("2020-03-05")
