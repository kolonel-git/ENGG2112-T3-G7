"""Horizon labels for O2 (D-005, D-019).

Prediction time t is the END of turbine-day D (D 24:00 UTC). For each candidate day:

Status precedence, highest first:

* ``in_episode``: D lies inside an episode span (Start <= D <= End). Dropped: we predict new
  episodes only.
* ``purged``: the horizon (D+exclusion_end) reaches a different split, so the label would use
  events from another split. Dropped.
* ``positive`` (label 1): an episode starts in the next ``positive_days`` days (D+1 .. D+30).
* ``zone``: no positive, but an episode starts in days D+31 .. D+exclusion_end. Dropped.
* ``censored``: no positive, but the horizon extends past the last observed day. Dropped.
* ``low_coverage``: no positive, but fewer than ``min_coverage`` of the days in
  D+1 .. D+exclusion_end were observed (data gaps could hide an episode). Dropped.
* ``negative`` (label 0): everything else.

Features for row D may use data up to the end of day D only.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

DAY = np.timedelta64(1, "D")
STATUSES = ["negative", "positive", "in_episode", "zone", "purged", "censored", "low_coverage"]


def split_of_year(cfg: dict[str, Any]) -> dict[int, str]:
    out: dict[int, str] = {}
    for name in ("train", "val", "test"):
        for year in cfg["splits"][f"{name}_years"]:
            out[int(year)] = name
    return out


def _status_one_turbine(
    days: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    positive_days: int,
    exclusion_end_days: int,
    min_coverage: float,
    last_observed: np.datetime64,
    split_of_day: np.ndarray,
    split_of_horizon: np.ndarray,
) -> np.ndarray:
    n = len(days)
    status = np.full(n, "negative", dtype=object)
    # episode starts in (D, D + positive], and in (D + positive, D + exclusion_end]
    lo = np.searchsorted(starts, days, side="right")
    hi_pos = np.searchsorted(starts, days + positive_days * DAY, side="right")
    hi_zone = np.searchsorted(starts, days + exclusion_end_days * DAY, side="right")
    positive = hi_pos > lo
    zone = (hi_zone > hi_pos) & ~positive
    # already inside an episode span
    idx = np.searchsorted(starts, days, side="right") - 1
    in_episode = np.zeros(n, dtype=bool)
    valid = idx >= 0
    in_episode[valid] = ends[idx[valid]] >= days[valid]
    # observation of the horizon
    horizon_end = days + exclusion_end_days * DAY
    censored = horizon_end > last_observed
    observed_in = np.searchsorted(days, horizon_end, side="right") - np.searchsorted(
        days, days, side="right"
    )
    low_coverage = (observed_in / exclusion_end_days) < min_coverage
    purged = split_of_day != split_of_horizon

    # later assignments win: in_episode > purged > positive > zone > censored > low_coverage
    status[low_coverage] = "low_coverage"
    status[censored] = "censored"
    status[zone] = "zone"
    status[positive] = "positive"
    status[purged] = "purged"
    status[in_episode] = "in_episode"
    return status


def build_horizon_labels(
    days: pd.DataFrame, episodes: pd.DataFrame, cfg: dict[str, Any]
) -> pd.DataFrame:
    """Label every candidate turbine-day.

    ``days``: StationId, Day (the observed candidate days, one row per turbine-day).
    ``episodes``: StationId, Start, End (day resolution).
    Returns StationId, Day, status, label (1, 0, or <NA> when the row is dropped) and split.
    """
    lab = cfg["labels"]
    positive_days = int(lab["positive_window_days"])
    exclusion_end = int(lab["exclusion_end_days"])
    min_cov = float(lab["min_horizon_coverage"])
    year_split = split_of_year(cfg)

    pieces = []
    for station, g in days.sort_values(["StationId", "Day"]).groupby("StationId"):
        d = g["Day"].to_numpy(dtype="datetime64[D]")
        ep = episodes.loc[episodes["StationId"] == station].sort_values("Start")
        starts = ep["Start"].to_numpy(dtype="datetime64[D]")
        ends = ep["End"].to_numpy(dtype="datetime64[D]")
        year_of_day = d.astype("datetime64[Y]").astype(int) + 1970
        horizon = d + exclusion_end * DAY
        year_of_horizon = horizon.astype("datetime64[Y]").astype(int) + 1970
        sp_day = np.array([year_split.get(int(y), "none") for y in year_of_day], dtype=object)
        sp_hor = np.array([year_split.get(int(y), "none") for y in year_of_horizon], dtype=object)
        # a horizon in a year outside all splits (e.g. 2019) is handled by coverage/censoring
        sp_hor = np.where(sp_hor == "none", sp_day, sp_hor)
        status = _status_one_turbine(
            d, starts, ends, positive_days, exclusion_end, min_cov, d[-1], sp_day, sp_hor
        )
        pieces.append(
            pd.DataFrame({"StationId": station, "Day": g["Day"].to_numpy(), "status": status,
                          "split": sp_day})
        )  # fmt: skip
    out = pd.concat(pieces, ignore_index=True)
    out["label"] = pd.array(
        np.where(out["status"] == "positive", 1, np.where(out["status"] == "negative", 0, -1)),
        dtype="Int8",
    )
    out.loc[out["label"] == -1, "label"] = pd.NA
    out["status"] = pd.Categorical(out["status"], categories=STATUSES)
    return out


def summarise_labels(
    labels: pd.DataFrame, episodes: pd.DataFrame, cfg: dict[str, Any]
) -> pd.DataFrame:
    """Row statuses, positives and episodes per split, for reporting (D-006 asks for counts)."""
    year_split = split_of_year(cfg)
    ep_split = episodes["Start"].dt.year.map(year_split).fillna("none")
    rows = []
    for split in ("train", "val", "test"):
        sub = labels.loc[labels["split"] == split]
        used = sub.loc[sub["label"].notna()]
        counts = sub["status"].value_counts()
        rows.append(
            {
                "split": split,
                "candidate_days": len(sub),
                "labelled_rows": len(used),
                "positives": int((used["label"] == 1).sum()),
                "positive_rate": round(float((used["label"] == 1).mean()), 4)
                if len(used)
                else None,
                "turbines_with_positives": int(used.loc[used["label"] == 1, "StationId"].nunique()),
                "episodes": int((ep_split == split).sum()),
                **{f"n_{s}": int(counts.get(s, 0)) for s in STATUSES},
            }
        )
    return pd.DataFrame(rows)
