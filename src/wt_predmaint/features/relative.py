"""Drift-robust feature sets for the exploratory runs (D-064).

``fleet``: each sensor-derived feature minus the median of the same feature over the turbines
observed on the same day (at least ``exploratory.min_turbines``), which removes change shared by
the whole fleet, such as weather and year-to-year sensor drift. A day only uses the same day's
other turbines, so nothing looks ahead. ``fleet_season`` also subtracts the month-of-year mean of
the fleet-relative value learned on the train years only. Alarm counts, fault history and the
operating-state fractions are left unchanged.
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
