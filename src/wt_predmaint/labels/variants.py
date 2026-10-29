"""Label variants for the sensitivity round (D-068), built in memory from a changed protocol copy.

A variant changes the positive window (the exclusion zone ends at twice the window), the target A
fault-day threshold, the target B stop length, and, for target B, whether stop runs that start
with a weather alarm active are dropped. The builders are the ones the frozen labels use
(``episodes``, ``horizon``), so the reference variant reproduces the frozen label files.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import pandas as pd

from wt_predmaint.labels.episodes import INTERVAL, fault_episodes_a, stop_episodes_b
from wt_predmaint.labels.horizon import build_horizon_labels


def variant_cfg(
    cfg: dict[str, Any],
    horizon: int,
    a_hours: float | None = None,
    b_intervals: int | None = None,
) -> dict[str, Any]:
    """Copy of the protocol with another positive window, fault-day threshold or stop length."""
    out = copy.deepcopy(cfg)
    out["labels"]["positive_window_days"] = int(horizon)
    out["labels"]["exclusion_end_days"] = 2 * int(horizon)
    targets = out["labels"]["targets"]
    if a_hours is not None:
        targets["daily_fault_time"]["fault_day_min_hours"] = a_hours
    if b_intervals is not None:
        targets["scada_stops"]["stop_min_intervals"] = int(b_intervals)
    return out


def weather_alarms(alarms: pd.DataFrame, tiers: list[int]) -> pd.DataFrame:
    """Active spans (StationId, Start, End) of alarms in ``tiers``; no end means a point in time."""
    a = alarms.loc[alarms["Tier"].isin([int(t) for t in tiers])]
    end = a["TimeOff"].where(a["TimeOff"].notna(), a["TimeStamp"])
    out = pd.DataFrame(
        {
            "StationId": a["StationId"],
            "Start": a["TimeStamp"],
            "End": np.maximum(end, a["TimeStamp"]),
        }
    )
    return out.sort_values(["StationId", "Start"]).reset_index(drop=True)


def drop_weather_runs(
    runs: pd.DataFrame, spans: pd.DataFrame, margin_intervals: int
) -> pd.DataFrame:
    """Stop runs with no weather alarm active from ``margin_intervals`` before them to their end."""
    margin = INTERVAL * int(margin_intervals)
    keep = pd.Series(True, index=runs.index)
    for station, g in runs.groupby("StationId"):
        s = spans.loc[spans["StationId"] == station]
        if s.empty:
            continue
        starts = s["Start"].to_numpy(dtype="datetime64[ns]")
        ends_so_far = np.maximum.accumulate(s["End"].to_numpy(dtype="datetime64[ns]"))
        run_end = g["End"].to_numpy(dtype="datetime64[ns]")
        run_from = (g["Start"] - margin).to_numpy(dtype="datetime64[ns]")
        k = np.searchsorted(starts, run_end, side="right") - 1
        hit = (k >= 0) & (ends_so_far[np.clip(k, 0, None)] >= run_from)
        keep.loc[g.index] = ~hit
    return runs.loc[keep]


def build_variant(
    target: str,
    cfg: dict[str, Any],
    daily: pd.DataFrame,
    scada: pd.DataFrame,
    alarms: pd.DataFrame | None = None,
    drop_weather: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(episodes, labels) for one target under ``cfg`` (a ``variant_cfg`` copy)."""
    if target == "a":
        episodes, days = fault_episodes_a(daily, cfg)
    elif target == "b":
        runs_filter = None
        if drop_weather:
            if alarms is None:
                raise ValueError("dropping weather stops needs the alarm table")
            sens = cfg["label_sensitivity"]
            spans = weather_alarms(alarms, list(sens["weather_tiers"]))
            margin = int(sens["weather_margin_intervals"])
            runs_filter = lambda runs: drop_weather_runs(runs, spans, margin)  # noqa: E731
        episodes, days = stop_episodes_b(scada, cfg, runs_filter)
    else:
        raise ValueError("target must be 'a' or 'b'")
    return episodes, build_horizon_labels(days, episodes, cfg)
