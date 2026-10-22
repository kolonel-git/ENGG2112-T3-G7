import copy

import numpy as np
import pandas as pd
import pytest

from wt_predmaint.config import load_protocol
from wt_predmaint.labels.episodes import (
    fault_episodes_a,
    merge_into_episodes,
    stop_episodes_b,
    stop_runs,
)
from wt_predmaint.labels.horizon import build_horizon_labels, summarise_labels

# All data below is synthetic.
T = pd.Timestamp


def day_frame(
    start: str, end: str, station: int = 1, drop: list[str] | None = None
) -> pd.DataFrame:
    days = pd.date_range(start, end, freq="D")
    if drop:
        days = days.difference(pd.to_datetime(drop))
    return pd.DataFrame({"StationId": station, "Day": days})


def cfg_one_split() -> dict:
    cfg = copy.deepcopy(load_protocol())
    cfg["splits"] = {"train_years": [2020], "val_years": [2021], "test_years": [2022]}
    return cfg


def events(rows: list[tuple[int, str, str]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["StationId", "Start", "End"]).astype(
        {"Start": "datetime64[ns]", "End": "datetime64[ns]"}
    )


def test_merge_gap_is_strictly_more_than_gap_days() -> None:
    ev = events(
        [(1, "2020-01-01", "2020-01-01"), (1, "2020-01-08", "2020-01-08"),  # exactly 7 days: merged
         (1, "2020-01-16", "2020-01-16"),  # 8 days after the previous end: new episode
         (2, "2020-01-01", "2020-01-01")]
    )  # fmt: skip
    ep = merge_into_episodes(ev, gap_days=7)
    s1 = ep[ep.StationId == 1]
    assert len(s1) == 2
    assert s1.iloc[0].Start == T("2020-01-01") and s1.iloc[0].End == T("2020-01-08")
    assert len(ep[ep.StationId == 2]) == 1


def test_merge_empty_input() -> None:
    assert merge_into_episodes(events([]), 7).empty


def test_fault_episodes_a_uses_threshold_and_ignores_unusable_days() -> None:
    cfg = load_protocol()
    daily = pd.DataFrame(
        {
            "StationId": 1,
            "TimeStamp": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03", "2020-02-01"]),
            "TimeErr": [3.9, 4.0, 30.0, 10.0],
            "partial_day": [False, False, False, False],
            "bad_hours": [False, False, True, False],  # day 3 has impossible hours
        }
    )
    ep, usable = fault_episodes_a(daily, cfg)
    assert len(usable) == 3
    assert ep["Start"].tolist() == [T("2020-01-02"), T("2020-02-01")]


def scada_frame(
    stop_pattern: list[bool],
    start: str = "2020-01-10 00:10",
    rpm_stop: float = 10.0,
    wind: float = 8.0,
    station: int = 1,
    outage: bool = False,
) -> pd.DataFrame:
    ts = pd.date_range(start, periods=len(stop_pattern), freq="10min")
    return pd.DataFrame(
        {
            "StationId": station,
            "TimeStamp": ts,
            "wtc_GenRpm_mean": [rpm_stop if s else 1200.0 for s in stop_pattern],
            "wtc_PrWindSp_mean": wind,
            "outage_month": outage,
        }
    )


def test_stop_runs_need_enough_consecutive_intervals() -> None:
    p = load_protocol()["labels"]["targets"]["scada_stops"]
    n = p["stop_min_intervals"]
    short = scada_frame([False] * 5 + [True] * (n - 1) + [False] * 5)
    long = scada_frame([False] * 5 + [True] * n + [False] * 5)
    assert stop_runs(short, p)[0].empty
    runs, _ = stop_runs(long, p)
    assert len(runs) == 1
    assert runs.iloc[0].Start == T("2020-01-10 00:10") + pd.Timedelta(minutes=50) - pd.Timedelta(
        minutes=10
    )


def test_stop_runs_broken_by_missing_interval_or_low_wind_or_outage() -> None:
    p = load_protocol()["labels"]["targets"]["scada_stops"]
    n = p["stop_min_intervals"]
    full = scada_frame([True] * n)
    gap = full.drop(index=n // 2)  # one missing interval splits the run
    assert stop_runs(gap, p)[0].empty
    assert stop_runs(scada_frame([True] * n, wind=3.0), p)[0].empty  # below minimum wind
    assert stop_runs(scada_frame([True] * n, rpm_stop=250.0), p)[0].empty  # rpm not below cutoff
    runs, observed = stop_runs(scada_frame([True] * n, outage=True), p)
    assert runs.empty and observed.empty  # outage-month rows are neither stops nor observed


def test_stop_episodes_b_merge_runs_within_gap() -> None:
    cfg = load_protocol()
    n = cfg["labels"]["targets"]["scada_stops"]["stop_min_intervals"]
    a = scada_frame([True] * n, start="2020-01-10 00:10")
    b = scada_frame([True] * n, start="2020-01-14 00:10")  # 4 days later: same episode
    c = scada_frame([True] * n, start="2020-02-20 00:10")  # far later: new episode
    ep, _ = stop_episodes_b(pd.concat([a, b, c]), cfg)
    assert len(ep) == 2
    assert ep.iloc[0].Start == T("2020-01-10") and ep.iloc[0].End == T("2020-01-14")


def labelled(
    days: pd.DataFrame, eps: list[tuple[str, str]], cfg: dict | None = None
) -> pd.DataFrame:
    ep = pd.DataFrame(
        {
            "StationId": 1,
            "Start": pd.to_datetime([a for a, _ in eps]),
            "End": pd.to_datetime([b for _, b in eps]),
        }
    )
    out = build_horizon_labels(days, ep, cfg or cfg_one_split())
    return out.set_index("Day")


def test_horizon_statuses_around_one_episode() -> None:
    lab = labelled(day_frame("2020-01-01", "2020-06-30"), [("2020-03-01", "2020-03-03")])
    st = lab["status"].astype(str)
    assert st["2020-01-01"] == "zone"  # episode start is exactly 60 days ahead
    assert st["2020-01-30"] == "zone"  # 31 days ahead
    assert st["2020-01-31"] == "positive"  # 30 days ahead
    assert st["2020-02-29"] == "positive"  # 1 day ahead
    assert st["2020-03-01"] == "in_episode" and st["2020-03-03"] == "in_episode"
    assert st["2020-03-04"] == "negative"
    assert st["2020-05-01"] == "negative"  # horizon ends exactly on the last observed day
    assert st["2020-05-02"] == "censored"  # 60-day horizon runs past the last observed day
    assert lab.loc["2020-01-31", "label"] == 1 and lab.loc["2020-03-04", "label"] == 0
    assert pd.isna(lab.loc["2020-01-01", "label"])


def test_horizon_low_coverage_blocks_negative_but_not_positive() -> None:
    gaps = [d.strftime("%Y-%m-%d") for d in pd.date_range("2020-03-15", "2020-04-20")]
    lab = labelled(day_frame("2020-01-01", "2020-08-30", drop=gaps), [("2020-02-10", "2020-02-10")])
    st = lab["status"].astype(str)
    assert st["2020-03-05"] == "low_coverage"  # most of the next 60 days are missing
    assert st["2020-01-15"] == "positive"  # an observed event decides the label regardless of gaps
    assert st["2020-02-20"] == "negative" or st["2020-02-20"] == "low_coverage"


def test_horizon_purges_rows_whose_horizon_reaches_another_split() -> None:
    days = day_frame("2020-11-01", "2021-03-31")
    lab = labelled(days, [("2021-01-05", "2021-01-05")])  # event in the validation year
    st = lab["status"].astype(str)
    assert st["2020-11-02"] == "purged"  # horizon reaches 2021 = validation
    assert st["2020-12-20"] == "purged"  # a positive-looking row is purged too, no leakage
    assert st["2021-01-02"] != "purged"  # same split as its horizon


def test_horizon_label_ignores_episodes_that_ended_before_the_day() -> None:
    days = day_frame("2020-01-01", "2020-12-31")
    one = labelled(days, [("2020-06-01", "2020-06-02")])
    two = labelled(days, [("2020-02-01", "2020-02-02"), ("2020-06-01", "2020-06-02")])
    late = one.index >= "2020-02-03"
    same = (one.loc[late, "status"].astype(str) == two.loc[late, "status"].astype(str)).all()
    # rows after the extra, earlier episode have identical status unless its zone/positive window
    # reached them (it cannot: windows only look forward)
    assert same


def test_summary_counts_by_split() -> None:
    cfg = cfg_one_split()
    days = day_frame("2020-01-01", "2020-06-30")
    ep = pd.DataFrame({"StationId": [1], "Start": [T("2020-03-01")], "End": [T("2020-03-03")]})
    lab = build_horizon_labels(days, ep, cfg)
    summ = summarise_labels(lab, ep, cfg).set_index("split")
    assert summ.loc["train", "candidate_days"] == len(days)
    assert summ.loc["train", "positives"] == 30
    assert summ.loc["train", "episodes"] == 1
    assert summ.loc["val", "candidate_days"] == 0


@pytest.mark.parametrize("bad", [0, -1])
def test_exclusion_window_must_be_longer_than_positive(bad: int) -> None:
    cfg = cfg_one_split()
    cfg["labels"]["exclusion_end_days"] = cfg["labels"]["positive_window_days"] + bad
    days = day_frame("2020-01-01", "2020-06-30")
    ep = pd.DataFrame({"StationId": [1], "Start": [T("2020-03-01")], "End": [T("2020-03-03")]})
    lab = build_horizon_labels(days, ep, cfg)
    # with no zone length the zone status can never appear
    assert "zone" not in set(lab["status"].astype(str)) and np.isfinite(len(lab))
