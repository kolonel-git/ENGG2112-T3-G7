"""Causal 10-minute features for Target C, available at each interval END."""

from __future__ import annotations

import numpy as np
import pandas as pd

KEYS = ["StationId", "TimeStamp"]
WINDOWS = {"1h": pd.Timedelta(hours=1), "24h": pd.Timedelta(days=1), "7d": pd.Timedelta(days=7)}


def short_term_features(
    scada: pd.DataFrame, events: pd.DataFrame, prediction_minutes: int = 10
) -> pd.DataFrame:
    """Use (t-window, t] sensor observations and exact past alarm starts, per turbine.

    Missing intervals are not filled with future data. Coverage counts accompany
    partial windows. Daily summaries and retrospective quality flags are excluded.
    """
    if prediction_minutes not in {10, 20, 30, 60}:
        raise ValueError("prediction_minutes must be 10, 20, 30 or 60")
    if scada.duplicated(KEYS).any() or scada[KEYS].isna().any().any():
        raise ValueError("SCADA keys must be non-null and unique")
    sensors = [c for c in scada if c.endswith(("_mean", "_max"))]
    if not sensors:
        raise ValueError("no SCADA mean/max sensor columns found")
    pieces = []
    for station, g in scada.groupby("StationId"):
        g = g.sort_values("TimeStamp").set_index("TimeStamp")
        values = g[sensors].replace([np.inf, -np.inf], np.nan)
        anchors = g.index[(g.index.minute % prediction_minutes == 0) & (g.index.second == 0)]
        parts = [values.loc[anchors].add_prefix("current_")]
        times = anchors.to_numpy()
        alarms = events.loc[events.StationId == station, "TimeOn"].dropna().sort_values()
        alarm_times = alarms.to_numpy()
        past_end = np.searchsorted(alarm_times, times, side="right")
        for name, window in WINDOWS.items():
            roll = values.rolling(window, closed="right", min_periods=1)
            parts.extend(
                [
                    roll.mean().loc[anchors].add_prefix(f"{name}_mean_"),
                    roll.std().loc[anchors].add_prefix(f"{name}_std_"),
                ]
            )
        res = pd.concat(parts, axis=1)
        for name, window in WINDOWS.items():
            res[f"{name}_scada_rows"] = (
                pd.Series(1, index=g.index).rolling(window).sum().loc[anchors]
            )
            res[f"{name}_alarm_count"] = past_end - np.searchsorted(
                alarm_times, times - window.to_timedelta64(), side="right"
            )
        res["minutes_since_alarm"] = np.nan
        known = past_end > 0
        if known.any():
            res.loc[known, "minutes_since_alarm"] = (
                times[known] - alarm_times[past_end[known] - 1]
            ) / np.timedelta64(1, "m")
        # Calendar fields are known at prediction time.
        res["hour_sin"] = np.sin(2 * np.pi * anchors.hour / 24)
        res["hour_cos"] = np.cos(2 * np.pi * anchors.hour / 24)
        pieces.append(res.assign(StationId=station).reset_index())
    if not pieces:
        raise ValueError("SCADA input is empty")
    return pd.concat(pieces, ignore_index=True).sort_values(KEYS).reset_index(drop=True)
