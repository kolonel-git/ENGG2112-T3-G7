"""Daily and 7-day features for O2 (D-021).

Row (StationId, Day) describes what is known at the END of turbine-day D, the same moment as the
label (D-005, D-013). SCADA ``TimeStamp`` is the interval end, so day D holds the intervals whose
START lies in D, that is TimeStamp in (D 00:00, D+1 00:00]. Windows are trailing: day D and the
six days before it. Nothing here looks at a later day; ``features/leakage.py`` tests that.

Column groups:

* ``op_<sensor>_mean``: daily mean of each 10-min mean sensor over producing intervals only
  (generator rpm at or above ``cleaning.producing_min_rpm``), NaN with too few such intervals.
* ``max_<sensor>_max``: daily maximum of each 10-min maximum sensor.
* state columns: interval count, fractions producing, idle and stopped, wind and rpm summary,
  share of intervals with a masked sensor value, outage-month and suspect-wind flags.
* ``w7mean_`` and ``w7std_``: trailing 7-day mean and standard deviation of the columns above.
* ``alarm_<code>_w7``, ``alarm_tier<k>_w7``, ``alarm_total_w7``: alarm counts over the window.
* ``days_since_fault_day`` and ``fault_days_30``: past fault time from ``tblDailySummary``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from wt_predmaint.data.clean import interval_start

KEYS = ["StationId", "Day"]
RPM, WIND = "wtc_GenRpm_mean", "wtc_PrWindSp_mean"
STATE_WINDOWED = [
    "frac_producing",
    "frac_idle",
    "frac_stopped",
    "wind_mean",
    "wind_std",
    "rpm_mean",
]
FAULT_LOOKBACK_DAYS = 30


def day_of_interval(timestamp: pd.Series) -> pd.Series:
    """Day a 10-min interval belongs to: the day of its start."""
    return interval_start(timestamp).dt.floor("D")


def daily_aggregates(scada: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """One row per (StationId, Day) from cleaned 10-min SCADA (D-015)."""
    min_rpm = float(cfg["cleaning"]["producing_min_rpm"])
    stops = cfg["labels"]["targets"]["scada_stops"]
    min_prod = int(cfg["features"]["min_producing_intervals"])

    mean_cols = [c for c in scada.columns if c.endswith("_mean")]
    max_cols = [c for c in scada.columns if c.endswith("_max")]
    work = scada[["StationId", *mean_cols, *max_cols, "n_masked", "outage_month", "suspect_wind"]]
    work = work.assign(Day=day_of_interval(scada["TimeStamp"]))
    rpm, wind = work[RPM], work[WIND]
    work = work.assign(
        producing=rpm >= min_rpm,
        idle=rpm < float(stops["stop_max_rpm"]),
        stopped=(rpm < float(stops["stop_max_rpm"])) & (wind >= float(stops["stop_min_wind_ms"])),
        masked=work["n_masked"] > 0,
    )

    grouped = work.groupby(KEYS)
    state = grouped.agg(
        n_intervals=(RPM, "size"),
        frac_producing=("producing", "mean"),
        frac_idle=("idle", "mean"),
        frac_stopped=("stopped", "mean"),
        wind_mean=(WIND, "mean"),
        wind_std=(WIND, "std"),
        rpm_mean=(RPM, "mean"),
        masked_frac=("masked", "mean"),
        outage_month=("outage_month", "max"),
        suspect_wind=("suspect_wind", "max"),
    )
    producing = work.loc[work["producing"]]
    op = producing.groupby(KEYS)[mean_cols].mean().add_prefix("op_")
    n_prod = producing.groupby(KEYS).size().rename("n_producing")
    op = op.where(n_prod.reindex(op.index) >= min_prod)
    mx = grouped[max_cols].max().add_prefix("max_")

    out = pd.concat([state, n_prod, op, mx], axis=1).copy()
    out["n_producing"] = out["n_producing"].fillna(0).astype(int)
    float_cols = out.select_dtypes("floating").columns
    out = out.astype(dict.fromkeys(float_cols, "float32"))
    return out.reset_index().copy()


def _calendar(frame: pd.DataFrame) -> pd.DataFrame:
    """One station's rows on a gap-free daily index (missing days are all-NaN rows)."""
    g = frame.set_index("Day").sort_index()
    return g.reindex(pd.date_range(g.index.min(), g.index.max(), freq="D"))


def add_window_features(daily: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Add trailing-window mean and std (window ``labels.feature_window_days``) per turbine."""
    window = int(cfg["labels"]["feature_window_days"])
    min_days = int(cfg["features"]["window_min_days"])
    value_cols = [c for c in daily.columns if c.startswith(("op_", "max_"))] + STATE_WINDOWED
    pieces = []
    for station, g in daily.groupby("StationId"):
        full = _calendar(g.drop(columns="StationId"))
        roll = full[value_cols].astype("float64").rolling(window, min_periods=min_days)
        res = pd.concat(
            [
                full,
                roll.mean().add_prefix(f"w{window}mean_"),
                roll.std().add_prefix(f"w{window}std_"),
            ],
            axis=1,
        )
        res = res.loc[res["n_intervals"].notna()].copy().rename_axis("Day").reset_index()
        pieces.append(res.assign(StationId=station))
    out = pd.concat(pieces, ignore_index=True)
    out = out[["StationId", *[c for c in out.columns if c != "StationId"]]]
    float_cols = out.select_dtypes("floating").columns
    out = out.astype(dict.fromkeys(float_cols, "float32"))
    return out


def alarm_window_features(
    events: pd.DataFrame, days: pd.DataFrame, cfg: dict[str, Any]
) -> pd.DataFrame:
    """Alarm counts over the trailing window for each (StationId, Day) in ``days``.

    ``events`` is the alarm table (``StationId``, ``Alarmcode``, ``TimeStamp``). Days without
    alarms count as zero, including days without SCADA coverage (see the outage flag).
    """
    window = int(cfg["labels"]["feature_window_days"])
    codes = [int(c["code"]) for c in cfg["labels"]["alarm_codes"]]
    tier_of = {int(c["code"]): int(c["tier"]) for c in cfg["labels"]["alarm_codes"]}
    ev = events.loc[events["Alarmcode"].isin(codes)]
    ev = ev.assign(Day=day_of_interval(ev["TimeStamp"]))
    counts = ev.groupby(["StationId", "Day", "Alarmcode"]).size().unstack(fill_value=0)
    counts = counts.reindex(columns=codes, fill_value=0)

    pieces = []
    for station, g in days.groupby("StationId"):
        index = pd.date_range(g["Day"].min(), g["Day"].max(), freq="D")
        if station in counts.index.get_level_values(0):
            c = counts.xs(station, level="StationId").reindex(index, fill_value=0)
        else:
            c = pd.DataFrame(0, index=index, columns=codes)
        roll = c.rolling(window, min_periods=1).sum()
        res = roll.rename(columns=lambda k: f"alarm_{k}_w{window}")
        for tier in sorted(set(tier_of.values())):
            members = [k for k in codes if tier_of[k] == tier]
            res[f"alarm_tier{tier}_w{window}"] = roll[members].sum(axis=1)
        res[f"alarm_total_w{window}"] = roll.sum(axis=1)
        res = res.rename_axis("Day").reset_index()
        res.insert(0, "StationId", station)
        pieces.append(res)
    out = pd.concat(pieces, ignore_index=True)
    return out.astype({c: "int32" for c in out.columns if c.startswith("alarm_")})


def fault_history_features(summary: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Past fault time per (StationId, Day) from ``tblDailySummary`` (target A definition).

    ``days_since_fault_day``: days from the latest fault day up to and including D (0 if D is one;
    NaN if none yet). ``fault_days_30``: fault days among the 30 days ending at D.
    """
    p = cfg["labels"]["targets"]["daily_fault_time"]
    usable = summary.loc[~(summary["partial_day"] | summary["bad_hours"])]
    pieces = []
    for station, g in usable.groupby("StationId"):
        g = g.sort_values("TimeStamp")
        index = pd.date_range(g["TimeStamp"].min(), g["TimeStamp"].max(), freq="D")
        fault = (
            (g.set_index("TimeStamp")["TimeErr"] >= float(p["fault_day_min_hours"]))
            .reindex(index, fill_value=False)
            .astype(int)
        )
        last = pd.Series(index.where(fault.to_numpy() == 1), index=index).ffill()
        since = (pd.Series(index, index=index) - last).dt.days.astype("float32")
        res = pd.DataFrame(
            {
                "days_since_fault_day": since,
                "fault_days_30": fault.rolling(FAULT_LOOKBACK_DAYS, min_periods=1).sum(),
            }
        )
        res = res.loc[res.index.isin(g["TimeStamp"])].rename_axis("Day").reset_index()
        res.insert(0, "StationId", station)
        pieces.append(res)
    out = pd.concat(pieces, ignore_index=True)
    out["fault_days_30"] = out["fault_days_30"].astype("int32")
    return out


def assemble_features(
    daily_agg: pd.DataFrame, events: pd.DataFrame, summary: pd.DataFrame, cfg: dict[str, Any]
) -> pd.DataFrame:
    """Daily aggregates -> window features -> plus alarm and fault-history features."""
    table = add_window_features(daily_agg, cfg)
    alarms = alarm_window_features(events, table[KEYS], cfg)
    faults = fault_history_features(summary, cfg)
    table = table.merge(alarms, on=KEYS, how="left").merge(faults, on=KEYS, how="left")
    return table.sort_values(KEYS, ignore_index=True).copy()  # .copy() defragments


def feature_columns(table: pd.DataFrame) -> list[str]:
    """Model inputs: everything except keys and the three data-quality flags."""
    drop = {*KEYS, "outage_month", "suspect_wind", "n_intervals", "n_producing", "masked_frac"}
    return [c for c in table.columns if c not in drop and not np.issubdtype(table[c].dtype, bool)]
