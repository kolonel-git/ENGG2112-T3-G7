"""Episode builders for the two O2 targets (D-003, D-007, D-008).

Target A: fault days from tblDailySummary.TimeErr. Target B: runs of SCADA stop intervals.
Both return one row per episode with day-resolution Start and End dates (turbine, UTC day).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pandas as pd

INTERVAL = pd.Timedelta(minutes=10)
EPISODE_COLUMNS = ["StationId", "Start", "End"]


def merge_into_episodes(events: pd.DataFrame, gap_days: float) -> pd.DataFrame:
    """Merge events (StationId, Start, End) into episodes.

    A new episode starts when an event begins more than `gap_days` after the previous event of
    the same turbine ended. Returns StationId, Start (first event start), End (last event end),
    sorted by turbine and start.
    """
    if events.empty:
        return pd.DataFrame(columns=EPISODE_COLUMNS)
    ev = events.sort_values(["StationId", "Start"]).reset_index(drop=True)
    prev_end = ev.groupby("StationId")["End"].transform(lambda s: s.cummax().shift())
    new_episode = prev_end.isna() | ((ev["Start"] - prev_end) > pd.Timedelta(days=gap_days))
    episode = new_episode.groupby(ev["StationId"]).cumsum()
    out = (
        ev.groupby(["StationId", episode.rename("Episode")])
        .agg(Start=("Start", "min"), End=("End", "max"))
        .reset_index()
        .drop(columns="Episode")
    )
    return out.sort_values(["StationId", "Start"]).reset_index(drop=True)


def usable_daily(daily: pd.DataFrame) -> pd.DataFrame:
    """Turbine-days of tblDailySummary that are neither partial nor have impossible hours."""
    return daily.loc[
        ~(daily["partial_day"] | daily["bad_hours"]), ["StationId", "TimeStamp"]
    ].rename(columns={"TimeStamp": "Day"})


def fault_episodes_a(daily: pd.DataFrame, cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Target A. Returns (episodes, usable_days).

    A fault day has TimeErr >= fault_day_min_hours (usable days only). Fault days more than
    episode_gap_days apart start a new episode; an episode spans first to last fault day.
    """
    p = cfg["labels"]["targets"]["daily_fault_time"]
    usable = daily.loc[~(daily["partial_day"] | daily["bad_hours"])]
    fault = usable.loc[usable["TimeErr"] >= p["fault_day_min_hours"], ["StationId", "TimeStamp"]]
    events = fault.assign(Start=fault["TimeStamp"], End=fault["TimeStamp"])[
        ["StationId", "Start", "End"]
    ]
    return merge_into_episodes(events, p["episode_gap_days"]), usable_daily(daily)


def stop_runs(scada: pd.DataFrame, p: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Qualifying stop runs and the observed days, from cleaned 10-min SCADA.

    Intervals in outage months are neither stops nor observed. A stop interval has generator rpm
    below stop_max_rpm and primary wind speed of at least stop_min_wind_ms. A run is consecutive
    10-min intervals of one turbine; runs of at least stop_min_intervals qualify. An observed day
    (interval-start day) needs min_observed_intervals_per_day intervals outside outage months.
    Returns (runs with StationId, Start, End as timestamps, observed days StationId, Day).
    """
    s = scada.loc[
        ~scada["outage_month"], ["StationId", "TimeStamp", "wtc_GenRpm_mean", "wtc_PrWindSp_mean"]
    ]
    s = s.sort_values(["StationId", "TimeStamp"]).reset_index(drop=True)
    s["Day"] = (s["TimeStamp"] - INTERVAL).dt.floor("D")
    n_day = s.groupby(["StationId", "Day"]).size().rename("n").reset_index()
    observed = n_day.loc[n_day["n"] >= p["min_observed_intervals_per_day"], ["StationId", "Day"]]

    stop = (s["wtc_GenRpm_mean"] < p["stop_max_rpm"]) & (
        s["wtc_PrWindSp_mean"] >= p["stop_min_wind_ms"]
    )
    consecutive = (s["StationId"] == s["StationId"].shift()) & (
        s["TimeStamp"] - s["TimeStamp"].shift() == INTERVAL
    )
    new_run = ~(stop & stop.shift(fill_value=False) & consecutive)
    run_id = new_run.cumsum()
    runs = (
        s.loc[stop]
        .groupby(run_id[stop])
        .agg(
            StationId=("StationId", "first"),
            first=("TimeStamp", "min"),
            last=("TimeStamp", "max"),
            n=("TimeStamp", "size"),
        )
    )
    runs = runs.loc[runs["n"] >= p["stop_min_intervals"]]
    out = pd.DataFrame(
        {"StationId": runs["StationId"], "Start": runs["first"] - INTERVAL, "End": runs["last"]}
    ).reset_index(drop=True)
    return out, observed.reset_index(drop=True)


def stop_episodes_b(
    scada: pd.DataFrame,
    cfg: dict[str, Any],
    runs_filter: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Target B. Returns (episodes with day-resolution Start/End, observed days).

    ``runs_filter`` (label sensitivity, D-068) may remove stop runs before they are merged; the
    observed days are unaffected.
    """
    p = cfg["labels"]["targets"]["scada_stops"]
    runs, observed = stop_runs(scada, p)
    if runs_filter is not None:
        runs = runs_filter(runs).reset_index(drop=True)
    episodes = merge_into_episodes(runs, p["episode_gap_days"])
    episodes["Start"] = episodes["Start"].dt.floor("D")
    episodes["End"] = (episodes["End"] - INTERVAL).dt.floor("D")  # End is an interval end
    return episodes, observed
