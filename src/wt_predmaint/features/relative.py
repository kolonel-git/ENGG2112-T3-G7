"""Drift-robust feature sets for the exploratory runs (D-064).

``fleet``: each sensor-derived feature minus the median of the same feature over the turbines
observed on the same day (at least ``exploratory.min_turbines``), which removes change shared by
the whole fleet, such as weather and year-to-year sensor drift. A day only uses the same day's
other turbines, so nothing looks ahead. ``fleet_season`` also subtracts the month-of-year mean of
the fleet-relative value learned on the train years only. Alarm counts, fault history and the
operating-state fractions are left unchanged.

``self`` (D-065): each sensor feature minus the median of the same turbine's own values over
the previous ``window_days`` days, the window ending the day before, so a turbine's recent
history is its baseline and nothing after the day is used. ``self_fleet`` applies the same to
the fleet-relative values.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

from wt_predmaint.models.logistic import b4_feature_columns

KEYS = ["StationId", "Day"]
PREFIXES = ("w7mean_", "w7std_", "op_", "max_")
WEATHER_STATE = {"wind_mean", "wind_std", "rpm_mean"}
SENSOR = re.compile(r"^wtc_.+_(?:mean|max|min)$")


def _base(column: str) -> str:
    changed = True
    while changed:
        changed = False
        for prefix in PREFIXES:
            if column.startswith(prefix):
                column, changed = column.removeprefix(prefix), True
    return column


def sensor_columns(columns: list[str]) -> list[str]:
    """Columns built from a sensor or from wind and rotor speed (the ones that can drift)."""
    out = []
    for column in columns:
        base = _base(column)
        if SENSOR.match(base) or base in WEATHER_STATE:
            out.append(column)
    return out


def fleet_relative(table: pd.DataFrame, columns: list[str], min_turbines: int) -> pd.DataFrame:
    """``columns`` minus the same-day median over turbines; missing if too few turbines."""
    grouped = table.groupby("Day")[columns]
    values = table[columns].astype("float64")
    out = values - grouped.transform("median").astype("float64")
    return out.where(grouped.transform("count") >= min_turbines).astype("float32")


def self_relative(
    table: pd.DataFrame, columns: list[str], window: int, min_days: int
) -> pd.DataFrame:
    """``columns`` minus the turbine's own trailing median; the window ends the day before.

    Days without enough observed history (fewer than ``min_days`` in the window) are missing.
    Each turbine is put on a gap-free daily calendar so the window means calendar days.
    """
    out = pd.DataFrame(np.nan, index=table.index, columns=columns, dtype="float32")
    for _, g in table.groupby("StationId"):
        values = g.set_index("Day")[columns].astype("float64")
        if values.index.has_duplicates:
            raise ValueError("StationId and Day must be unique")
        full = values.reindex(pd.date_range(values.index.min(), values.index.max(), freq="D"))
        baseline = full.rolling(window, min_periods=min_days).median().shift(1)
        rel = (full - baseline).reindex(values.index)
        out.loc[g.index, columns] = rel.to_numpy(dtype="float32")
    return out


def month_means(relative: pd.DataFrame, days: pd.Series, train_years: list[int]) -> pd.DataFrame:
    """Month-of-year mean of each column over rows in the train years (12 rows)."""
    mask = days.dt.year.isin(train_years).to_numpy()
    return relative.loc[mask].groupby(days.loc[mask].dt.month.to_numpy()).mean()


def build_feature_set(features: pd.DataFrame, cfg: dict[str, Any], name: str) -> pd.DataFrame:
    """The feature table with its sensor-derived columns replaced as ``name`` says.

    ``raw`` returns the table unchanged. Column names stay the same, so every model reads the
    transformed table without changes. The month means for ``fleet_season`` use train years only.
    """
    if name == "raw":
        return features
    if name in {"self", "self_fleet"}:
        r2 = cfg["exploratory_round2"]
        columns = sensor_columns(b4_feature_columns(features, cfg))
        base = features[[*KEYS, *columns]]
        if name == "self_fleet":
            base = features[KEYS].copy()
            base[columns] = fleet_relative(
                features, columns, int(cfg["exploratory"]["min_turbines"])
            )
        rel = self_relative(base, columns, int(r2["window_days"]), int(r2["min_days"]))
        out = features.copy()
        out[columns] = rel[columns].to_numpy(dtype="float32")
        return out
    if name not in {"fleet", "fleet_season"}:
        raise ValueError(f"unknown feature set {name!r}")
    columns = sensor_columns(b4_feature_columns(features, cfg))
    relative = fleet_relative(features, columns, int(cfg["exploratory"]["min_turbines"]))
    if name == "fleet_season":
        means = month_means(
            relative, features["Day"], [int(y) for y in cfg["splits"]["train_years"]]
        )
        aligned = means.reindex(features["Day"].dt.month.to_numpy())
        relative = (relative.to_numpy(dtype="float64") - aligned.to_numpy(dtype="float64")).astype(
            "float32"
        )
        relative = pd.DataFrame(relative, index=features.index, columns=columns)
    out = features.copy()
    out[columns] = np.asarray(relative, dtype="float32")
    return out
