"""Target C: a new raw turbine alarm in (prediction time, prediction time + horizon]."""

from __future__ import annotations

import numpy as np
import pandas as pd

HORIZON = pd.Timedelta(hours=6)


def alarm_labels(
    rows: pd.DataFrame,
    events: pd.DataFrame,
    coverage: pd.DataFrame,
    horizon: pd.Timedelta = HORIZON,
) -> pd.DataFrame:
    """Label each station independently using exact TimeOn, never rounded alarm times.

    Coverage columns are StationId, Start, End. Each interval certifies complete raw
    alarm observation on (Start, End]. Overlapping/touching intervals are merged.
    A witnessed alarm proves a positive even with incomplete coverage; a negative
    requires the entire horizon to be covered. Unknown labels remain nullable.
    """
    if horizon <= pd.Timedelta(0):
        raise ValueError("horizon must be positive")
    out = rows[["StationId", "TimeStamp"]].copy().reset_index(drop=True)
    out["horizon_end"] = out["TimeStamp"] + horizon
    out["label"] = pd.Series(pd.NA, index=out.index, dtype="Int8")
    out["horizon_observed"] = False
    if coverage[["Start", "End"]].isna().any().any() or (coverage.End <= coverage.Start).any():
        raise ValueError("coverage intervals need non-null Start < End")
    for station, g in out.groupby("StationId"):
        starts = events.loc[events.StationId == station, "TimeOn"].dropna().sort_values()
        t, end = g.TimeStamp.to_numpy(), g.horizon_end.to_numpy()
        count = np.searchsorted(starts.to_numpy(), end, side="right") - np.searchsorted(
            starts.to_numpy(), t, side="right"
        )
        intervals = coverage.loc[coverage.StationId == station].sort_values("Start")
        merged: list[list[pd.Timestamp]] = []
        for row in intervals.itertuples():
            if merged and row.Start <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], row.End)
            else:
                merged.append([row.Start, row.End])
        observed = np.zeros(len(g), dtype=bool)
        for left, right in merged:
            observed |= ((g.TimeStamp >= left) & (g.horizon_end <= right)).to_numpy()
        out.loc[g.index, "horizon_observed"] = observed
        out.loc[g.index[observed], "label"] = 0
        out.loc[g.index[count > 0], "label"] = 1
    return out


def assign_splits(labels: pd.DataFrame, splits: dict) -> pd.DataFrame:
    """Purge horizons crossing a split or an unused year; retain historical feature context."""
    out = labels.copy()
    year_to_split = {}
    for split in ("train", "val", "test"):
        for year in splits[f"{split}_years"]:
            if year in year_to_split:
                raise ValueError("split years must be disjoint")
            year_to_split[year] = split
    out["split"] = out.TimeStamp.dt.year.map(year_to_split).fillna("unused")
    # Six-hour horizons can cross at most one year boundary.
    end_split = out.horizon_end.dt.year.map(year_to_split).fillna("unused")
    invalid = (out.split == "unused") | (out.split != end_split)
    out.loc[invalid, "label"] = pd.NA
    return out
