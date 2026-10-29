"""Label variants for the sensitivity round (D-068) on synthetic data."""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from wt_predmaint.config import load_protocol
from wt_predmaint.labels.episodes import stop_episodes_b
from wt_predmaint.labels.variants import (
    build_variant,
    drop_weather_runs,
    variant_cfg,
    weather_alarms,
)

T = pd.Timestamp
CFG = copy.deepcopy(load_protocol())
CFG["splits"] = {"train_years": [2020], "val_years": [2021], "test_years": [2022]}


def test_variant_cfg_changes_only_the_asked_settings_and_leaves_the_original() -> None:
    out = variant_cfg(CFG, 7, a_hours=8, b_intervals=36)
    assert out["labels"]["positive_window_days"] == 7
    assert out["labels"]["exclusion_end_days"] == 14
    assert out["labels"]["targets"]["daily_fault_time"]["fault_day_min_hours"] == 8
    assert out["labels"]["targets"]["scada_stops"]["stop_min_intervals"] == 36
    assert CFG["labels"]["positive_window_days"] == 30
    assert CFG["labels"]["targets"]["scada_stops"]["stop_min_intervals"] == 18
    same = variant_cfg(CFG, 30)
    assert same["labels"] == CFG["labels"]


def alarms() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "StationId": [1, 1, 1, 2],
            "Tier": [2, 1, 2, 2],
            "TimeStamp": [T("2020-01-03 11:00"), T("2020-01-10 00:00")]
            + [T("2020-01-12 00:00")] * 2,
            "TimeOff": [
                T("2020-01-03 12:00"),
                T("2020-01-10 05:00"),
                pd.NaT,
                T("2020-01-12 01:00"),
            ],
        }
    )


def test_weather_alarms_keep_the_tier_and_treat_a_missing_end_as_a_point() -> None:
    spans = weather_alarms(alarms(), [2])
    assert len(spans) == 3 and set(spans["StationId"]) == {1, 2}
    point = spans.loc[(spans["StationId"] == 1) & (spans["Start"] == T("2020-01-12"))].iloc[0]
    assert point["End"] == point["Start"]


def test_runs_are_dropped_only_when_a_weather_alarm_overlaps_them() -> None:
    runs = pd.DataFrame(
        {
            "StationId": [1, 1, 1, 1, 2],
            "Start": [T("2020-01-03 10:00"), T("2020-01-05 10:00"), T("2020-01-12 00:20"),
                      T("2020-01-10 00:00"), T("2020-01-03 10:00")],
            "End": [T("2020-01-03 14:00"), T("2020-01-05 14:00"), T("2020-01-12 04:00"),
                    T("2020-01-10 04:00"), T("2020-01-03 14:00")],
        }
    )  # fmt: skip
    spans = weather_alarms(alarms(), [2])
    kept = drop_weather_runs(runs, spans, margin_intervals=1)
    assert kept.index.tolist() == [1, 2, 3, 4]  # only the first run overlaps a weather alarm
    # the tier-1 alarm on 2020-01-10 does not count; a point alarm just before the run does
    # not either unless it falls inside the margin
    near = drop_weather_runs(runs.iloc[[2]], spans, margin_intervals=1)
    assert len(near) == 1
    wide = drop_weather_runs(runs.iloc[[2]], spans, margin_intervals=3)
    assert len(wide) == 0  # the point alarm at 00:00 is within 30 minutes of a run starting 00:20


def scada() -> pd.DataFrame:
    stamps = pd.date_range("2020-01-01 00:10", "2020-01-21 00:00", freq="10min")
    d = pd.DataFrame(
        {
            "StationId": 1,
            "TimeStamp": stamps,
            "wtc_GenRpm_mean": 1000.0,
            "wtc_PrWindSp_mean": 8.0,
            "outage_month": False,
        }
    )
    for day in ("2020-01-03", "2020-01-15"):
        stop = (d["TimeStamp"] > T(f"{day} 10:00")) & (d["TimeStamp"] <= T(f"{day} 14:00"))
        d.loc[stop, "wtc_GenRpm_mean"] = 0.0
    return d


def test_stop_episodes_filter_removes_the_weather_episode_but_not_the_observed_days() -> None:
    plain, days = stop_episodes_b(scada(), CFG)
    assert len(plain) == 2
    spans = weather_alarms(alarms(), [2])
    filtered, days2 = stop_episodes_b(scada(), CFG, lambda r: drop_weather_runs(r, spans, 1))
    assert len(filtered) == 1 and filtered["Start"].iloc[0] == T("2020-01-15")
    assert days.equals(days2)


def test_build_variant_b_needs_alarms_and_gives_fewer_episodes_when_dropping() -> None:
    cfg = variant_cfg(CFG, 30)
    cfg["label_sensitivity"]["weather_tiers"] = [2]
    with pytest.raises(ValueError, match="alarm table"):
        build_variant("b", cfg, pd.DataFrame(), scada(), None, drop_weather=True)
    keep, _ = build_variant("b", cfg, pd.DataFrame(), scada(), alarms(), drop_weather=False)
    drop, _ = build_variant("b", cfg, pd.DataFrame(), scada(), alarms(), drop_weather=True)
    assert len(keep) == 2 and len(drop) == 1


def test_build_variant_a_threshold_and_horizon_change_episodes_and_labels() -> None:
    days = pd.date_range("2020-01-01", "2020-12-31", freq="D")
    hours = np.zeros(len(days))
    hours[days == T("2020-03-01")] = 5.0
    hours[days == T("2020-06-01")] = 10.0
    daily = pd.DataFrame(
        {
            "StationId": 1,
            "TimeStamp": days,
            "TimeErr": hours,
            "partial_day": False,
            "bad_hours": False,
        }
    )
    base_ep, base_lab = build_variant("a", variant_cfg(CFG, 30), daily, pd.DataFrame())
    strict_ep, _ = build_variant("a", variant_cfg(CFG, 30, a_hours=8), daily, pd.DataFrame())
    assert len(base_ep) == 2 and len(strict_ep) == 1
    _, short = build_variant("a", variant_cfg(CFG, 7), daily, pd.DataFrame())
    assert (short["label"] == 1).sum() == 14  # 7 positive days before each of the two episodes
    assert (base_lab["label"] == 1).sum() == 60
    with pytest.raises(ValueError, match="target"):
        build_variant("c", CFG, daily, pd.DataFrame())
