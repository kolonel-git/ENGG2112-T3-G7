"""Evaluation harness for O2 (D-009, D-020).

A scored frame has one row per labelled turbine-day: StationId, Day, label (0 or 1) and score
(higher means more likely to precede an episode). Rows without a label (D-019) must not be in it.

* An alert is a row with ``score > threshold`` (strict, so tied scores are all in or all out).
* A false alarm is an alert on a row with label 0. The false alarm rate is per turbine-month,
  where one turbine-month is ``days_per_month`` scored turbine-days.
* An episode is detected if an alert falls on a scored day in the ``positive_window_days`` before
  its start. Lead time is the days from the first such alert to the episode start.
* The threshold is chosen on validation to meet the false alarm budget, then frozen for test.
* Confidence intervals resample turbines with replacement, not rows (rows overlap heavily).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

DAY = np.timedelta64(1, "D")
METRICS = ["pr_auc", "recall", "median_lead_days", "false_alarms_per_turbine_month"]


def _check(frame: pd.DataFrame) -> None:
    missing = {"StationId", "Day", "label", "score"} - set(frame.columns)
    if missing:
        raise ValueError(f"scored frame is missing columns {sorted(missing)}")
    if frame.empty:
        raise ValueError("Cannot evaluate an empty scored frame: no labelled turbine-days")
    if frame["label"].isna().any() or not frame["label"].isin([0, 1]).all():
        raise ValueError("scored frame must contain only labels 0 and 1 (drop unlabelled rows)")
    if frame["score"].isna().any():
        raise ValueError("scores must not be NaN")


def pr_auc(y: np.ndarray, score: np.ndarray) -> float:
    """Average precision; NaN when there are no positives."""
    y = np.asarray(y)
    if y.sum() == 0:
        return float("nan")
    return float(average_precision_score(y, score))


def turbine_months(n_days: int, days_per_month: float) -> float:
    return n_days / days_per_month


def threshold_for_budget(
    label: np.ndarray, score: np.ndarray, budget: float, days_per_month: float
) -> float:
    """Lowest threshold whose false alarms stay within ``budget`` per turbine-month.

    Alerts are ``score > threshold``. Returns -inf if even alerting everywhere fits the budget.
    """
    label = np.asarray(label)
    score = np.asarray(score, dtype=float)
    if not len(label):
        raise ValueError("Cannot choose a threshold without labelled validation data")
    allowed = int(np.floor(budget * turbine_months(len(label), days_per_month)))
    negatives = np.sort(score[label == 0])[::-1]
    if allowed >= len(negatives):
        return float("-inf")
    return float(negatives[allowed])


def detect_episodes(
    frame: pd.DataFrame, episodes: pd.DataFrame, threshold: float, positive_days: int
) -> pd.DataFrame:
    """Per episode: is it evaluable, detected, and by how many days.

    An episode is evaluable if at least one scored day lies in its window (the purge and the
    in-episode rule can remove them all). Returns StationId, Start, evaluable, detected,
    lead_days (NaN if not detected).
    """
    _check(frame)
    rows = []
    for station, g in frame.sort_values("Day").groupby("StationId"):
        days = g["Day"].to_numpy(dtype="datetime64[D]")
        alerts = days[g["score"].to_numpy() > threshold]
        starts = episodes.loc[episodes["StationId"] == station, "Start"]
        for start in starts.to_numpy(dtype="datetime64[D]"):
            lo, hi = start - positive_days * DAY, start - DAY
            scored = np.searchsorted(days, hi, side="right") - np.searchsorted(days, lo)
            i, j = np.searchsorted(alerts, lo), np.searchsorted(alerts, hi, side="right")
            detected = bool(j > i)
            rows.append(
                {
                    "StationId": station,
                    "Start": pd.Timestamp(start),
                    "evaluable": bool(scored > 0),
                    "detected": detected,
                    "lead_days": float((start - alerts[i]) / DAY) if detected else np.nan,
                }
            )
    cols = ["StationId", "Start", "evaluable", "detected", "lead_days"]
    return pd.DataFrame(rows, columns=cols)


def episodes_in_split(episodes: pd.DataFrame, years: list[int]) -> pd.DataFrame:
    return episodes.loc[episodes["Start"].dt.year.isin(years)]


def evaluate(
    frame: pd.DataFrame, episodes: pd.DataFrame, threshold: float, cfg: dict[str, Any]
) -> dict[str, float]:
    """Point estimates of the four D-009 metrics at a frozen threshold."""
    _check(frame)
    ev = cfg["evaluation"]
    positive_days = int(cfg["labels"]["positive_window_days"])
    det = detect_episodes(frame, episodes, threshold, positive_days)
    det = det.loc[det["evaluable"]]
    alert = frame["score"].to_numpy() > threshold
    false_alarms = int((alert & (frame["label"].to_numpy() == 0)).sum())
    months = turbine_months(len(frame), float(ev["days_per_month"]))
    return {
        "n_rows": len(frame),
        "n_positive_rows": int(frame["label"].sum()),
        "prevalence": float(frame["label"].mean()),
        "n_episodes": len(det),
        "n_detected": int(det["detected"].sum()),
        "pr_auc": pr_auc(frame["label"].to_numpy(), frame["score"].to_numpy()),
        "recall": float(det["detected"].mean()) if len(det) else float("nan"),
        "median_lead_days": float(det["lead_days"].median()) if det["detected"].any() else np.nan,
        "false_alarms_per_turbine_month": false_alarms / months,
    }


def bootstrap_turbines(
    frame: pd.DataFrame,
    episodes: pd.DataFrame,
    threshold: float,
    cfg: dict[str, Any],
    resamples: int | None = None,
) -> dict[str, tuple[float, float]]:
    """Percentile intervals for the four metrics, resampling turbines with replacement.

    The threshold is fixed, so per-episode detection is computed once and only the turbine
    weights change between draws. Draws where a metric is undefined (for example no positives)
    are ignored for that metric.
    """
    _check(frame)
    ev = cfg["evaluation"]
    n_boot = int(resamples if resamples is not None else ev["bootstrap_resamples"])
    level = float(ev["ci_level"])
    dpm = float(ev["days_per_month"])
    positive_days = int(cfg["labels"]["positive_window_days"])
    det = detect_episodes(frame, episodes, threshold, positive_days)
    det = det.loc[det["evaluable"]]

    stations = np.sort(frame["StationId"].unique())
    by = {s: g for s, g in frame.groupby("StationId")}
    y = [by[s]["label"].to_numpy() for s in stations]
    sc = [by[s]["score"].to_numpy() for s in stations]
    fa = np.array([int(((x > threshold) & (t == 0)).sum()) for x, t in zip(sc, y, strict=True)])
    n_days = np.array([len(t) for t in y])
    n_ep = np.array([(det["StationId"] == s).sum() for s in stations])
    n_det = np.array([det.loc[det["StationId"] == s, "detected"].sum() for s in stations])
    leads = [det.loc[(det["StationId"] == s) & det["detected"], "lead_days"].to_numpy()
             for s in stations]  # fmt: skip

    rng = np.random.default_rng(int(cfg["seed"]))
    out: dict[str, list[float]] = {m: [] for m in METRICS}
    for _ in range(n_boot):
        draw = rng.integers(0, len(stations), size=len(stations))
        out["pr_auc"].append(
            pr_auc(np.concatenate([y[i] for i in draw]), np.concatenate([sc[i] for i in draw]))
        )
        eps = n_ep[draw].sum()
        out["recall"].append(n_det[draw].sum() / eps if eps else np.nan)
        lead = np.concatenate([leads[i] for i in draw])
        out["median_lead_days"].append(float(np.median(lead)) if len(lead) else np.nan)
        out["false_alarms_per_turbine_month"].append(
            fa[draw].sum() / turbine_months(int(n_days[draw].sum()), dpm)
        )
    lo, hi = 100 * (1 - level) / 2, 100 * (1 + level) / 2
    result = {}
    for m, values in out.items():
        arr = np.asarray(values, dtype=float)
        arr = arr[~np.isnan(arr)]
        result[m] = (
            (float(np.percentile(arr, lo)), float(np.percentile(arr, hi)))
            if len(arr)
            else (np.nan, np.nan)
        )
    return result


def evaluation_row(
    name: str,
    frame: pd.DataFrame,
    episodes: pd.DataFrame,
    threshold: float,
    cfg: dict[str, Any],
    resamples: int | None = None,
) -> dict[str, Any]:
    """One results-table row: point estimates plus ``<metric>_lo`` and ``<metric>_hi``."""
    row: dict[str, Any] = {"model": name, "threshold": threshold}
    row.update(evaluate(frame, episodes, threshold, cfg))
    for metric, (lo, hi) in bootstrap_turbines(frame, episodes, threshold, cfg, resamples).items():
        row[f"{metric}_lo"], row[f"{metric}_hi"] = lo, hi
    return row
