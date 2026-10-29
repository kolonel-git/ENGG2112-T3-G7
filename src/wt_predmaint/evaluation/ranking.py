"""Turbine-ranking metric: does the score put the turbines that are about to fault first? (D-065)

Each day, the turbines that have a label that day are ranked by score. Precision among the top ``k``
is compared with the chance level, the share of positive labels among those turbines that day.
Differences are averaged over days; intervals resample calendar months. This is descriptive: daily
windows overlap, so the intervals understate dependence.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def daily_precision_at_k(
    frame: pd.DataFrame, ks: Sequence[int], min_turbines: int, seed: int
) -> pd.DataFrame:
    """One row per day: labelled turbines, chance level, and precision at each ``k``.

    ``frame`` has StationId, Day, label and score. Ties in score are broken at random with ``seed``
    so a constant score gets the chance level on average. Days with fewer than ``min_turbines``
    labelled turbines are skipped.
    """
    if not {"StationId", "Day", "label", "score"} <= set(frame):
        raise ValueError("frame needs StationId, Day, label and score")
    rng = np.random.default_rng(seed)
    work = frame[["Day", "label", "score"]].copy()
    work["tie"] = rng.random(len(work))
    work = work.sort_values(["Day", "score", "tie"], ascending=[True, False, False])
    work["rank"] = work.groupby("Day").cumcount() + 1
    rows = []
    for day, g in work.groupby("Day", sort=True):
        if len(g) < min_turbines:
            continue
        row = {"Day": day, "n_turbines": len(g), "chance": float(g["label"].mean())}
        for k in ks:
            top = g.loc[g["rank"] <= k, "label"]
            row[f"p_at_{k}"] = float(top.mean())
        rows.append(row)
    return pd.DataFrame(rows)


def ranking_summary(
    daily: pd.DataFrame, ks: Sequence[int], resamples: int, seed: int
) -> pd.DataFrame:
    """Mean precision at ``k``, mean chance, their difference and a month-bootstrap interval."""
    if daily.empty:
        raise ValueError("no days with enough labelled turbines")
    months = daily["Day"].dt.to_period("M")
    groups = {m: g.index.to_numpy() for m, g in daily.groupby(months)}
    keys = list(groups)
    rng = np.random.default_rng(seed)
    out = []
    for k in ks:
        diff = (daily[f"p_at_{k}"] - daily["chance"]).to_numpy()
        boot = np.empty(resamples)
        for i in range(resamples):
            pick = rng.integers(0, len(keys), len(keys))
            idx = np.concatenate([groups[keys[j]] for j in pick])
            boot[i] = diff[idx].mean()
        out.append(
            {
                "k": k,
                "days": len(daily),
                "mean_precision": float(daily[f"p_at_{k}"].mean()),
                "mean_chance": float(daily["chance"].mean()),
                "difference": float(diff.mean()),
                "difference_lo": float(np.percentile(boot, 2.5)),
                "difference_hi": float(np.percentile(boot, 97.5)),
            }
        )
    return pd.DataFrame(out)
