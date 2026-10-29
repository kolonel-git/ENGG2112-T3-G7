"""H0: train-only healthy fitting rows for B3, R1 and R2 (D-030).

These are masks, not Baseline scorers: future-event labels may select training
examples but must never become risk-score inputs. Pass the complete label table
for the requested target, including in-episode/unlabelled train days, not just
the already-filtered negatives. No disk access or fitted state is hidden here.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from wt_predmaint.data.clean import interval_start
from wt_predmaint.labels.episodes import stop_runs
from wt_predmaint.labels.horizon import STATUSES
from wt_predmaint.models.base import KEYS, attach_labels, require_train_only

FLAGS = ["outage_month", "suspect_wind"]


def _settings(cfg: dict[str, Any]) -> tuple[int, int, int, float]:
    settings = cfg["healthy"]
    start, end, recovery = (settings[k] for k in ("start_year", "end_year", "recovery_days"))
    if any(isinstance(v, bool) or not isinstance(v, int) for v in (start, end, recovery)):
        raise ValueError("healthy years and recovery_days must be integers")
    if start > end or recovery < 0:
        raise ValueError("Invalid healthy year range or negative recovery_days")
    cut_in = float(settings["cut_in_wind_ms"])
    if not np.isfinite(cut_in) or cut_in < 0:
        raise ValueError("healthy.cut_in_wind_ms must be finite and nonnegative")
    return start, end, recovery, cut_in


def _train_days(days: pd.Series, cfg: dict[str, Any]) -> pd.Series:
    start, end, _, _ = _settings(cfg)
    return days.dt.year.isin(cfg["splits"]["train_years"]) & days.dt.year.between(start, end)


def _good_flags(frame: pd.DataFrame) -> pd.Series:
    for flag in FLAGS:
        if not frame[flag].dropna().isin([False, True]).all():
            raise ValueError(f"{flag} must be boolean or 0/1, not text")
    # Missing quality flags are unknown, never evidence of healthy behaviour.
    return frame[FLAGS].eq(False).fillna(False).all(axis=1)  # noqa: E712


def healthy_days(
    labels: pd.DataFrame, features: pd.DataFrame, cfg: dict[str, Any], target: str
) -> pd.Series:
    """Boolean mask aligned to features, including False for all nontraining rows.

    A healthy row is a labelled negative in BOTH the declared train split and
    configured training years, has both quality flags explicitly false, and has
    a complete preceding recovery window of train-label days without an episode.
    Thus the N days following an episode's end are excluded. Missing history and
    the beginning of a training block are also excluded until N days are known.
    `target` identifies which complete label table the caller supplies (a or b).
    Feature dates are day starts; 10-minute callers must use healthy_intervals.
    """
    if target not in {"a", "b"}:
        raise ValueError("target must be 'a' or 'b'")
    _, _, recovery, _ = _settings(cfg)
    required_labels = {*KEYS, "label", "status", "split"}
    if required_labels - set(labels) or set(KEYS + FLAGS) - set(features):
        raise ValueError(
            "Healthy masks need label keys/status/label/split and feature keys/quality flags"
        )
    result = pd.Series(False, index=features.index, name=f"healthy_{target}", dtype=bool)
    candidate = _train_days(features["Day"], cfg)
    if "split" in features:
        candidate &= features["split"].eq("train")
    positions = np.flatnonzero(candidate.fillna(False).to_numpy(dtype=bool))
    if not len(positions):
        return result
    f = features.iloc[positions][KEYS + FLAGS].copy()
    if not f["Day"].eq(f["Day"].dt.normalize()).all():
        raise ValueError("Training feature Day must be day-resolution midnight timestamps")
    lab = labels.loc[_train_days(labels["Day"], cfg) & labels["split"].eq("train")].copy()
    if "target" in lab:
        lab = lab.loc[lab["target"].str.lower().eq(target)]
    if f.duplicated(KEYS).any() or lab.duplicated(KEYS).any():
        raise ValueError("Healthy-mask inputs must have unique turbine/day keys")
    table = attach_labels(f, lab, cfg)
    keep = (
        table["split"].eq("train")
        & table["status"].eq("negative")
        & table["label"].eq(0).fillna(False)
        & _good_flags(table)
    )
    if recovery:
        history_ok = np.zeros(len(table), dtype=bool)
        years = [
            y
            for y in cfg["splits"]["train_years"]
            if cfg["healthy"]["start_year"] <= y <= cfg["healthy"]["end_year"]
        ]
        calendar = pd.date_range(
            pd.Timestamp(year=min(years), month=1, day=1) - pd.Timedelta(days=recovery),
            pd.Timestamp(year=max(years), month=12, day=31),
        )
        for station, group in table.groupby("StationId", sort=False):
            observed = lab.loc[lab["StationId"].eq(station)].set_index("Day")
            # An explicit status proves the day was observed; purged/zone days
            # can supply past episode history, though they cannot be healthy.
            known = observed["status"].isin(set(STATUSES) - {"in_episode"})
            safe = known.reindex(calendar, fill_value=False).fillna(False).astype(bool)
            recovered = (
                safe.shift(1, fill_value=False)
                .rolling(recovery, min_periods=recovery)
                .sum()
                .eq(recovery)
            )
            history_ok[group.index.to_numpy()] = recovered.reindex(group["Day"]).to_numpy(
                dtype=bool
            )
        keep &= history_ok
    require_train_only(table.loc[keep])
    result.iloc[positions] = keep.to_numpy(dtype=bool)
    return result


def healthy_intervals(
    labels: pd.DataFrame,
    features: pd.DataFrame,
    scada: pd.DataFrame,
    cfg: dict[str, Any],
    target: str,
) -> pd.Series:
    """10-minute mask aligned to scada; day assignment uses interval START.

    Require a healthy day, clear interval quality flags, finite rpm/wind and wind
    >= configured cut-in proxy. Exclude whole qualifying stop runs using the same
    definition as labels/episodes.py. Stop detection sees train intervals only.
    Pass complete clean train history, not prefiltered healthy intervals, so runs
    crossing midnight are detected before applying the day mask.
    """
    _, _, _, cut_in = _settings(cfg)
    rpm, wind = "wtc_GenRpm_mean", "wtc_PrWindSp_mean"
    required = {"StationId", "TimeStamp", rpm, wind, *FLAGS}
    if required - set(scada):
        raise ValueError(f"Missing SCADA columns: {sorted(required - set(scada))}")
    day_mask = healthy_days(labels, features, cfg, target)
    healthy_keys = pd.MultiIndex.from_frame(features.loc[day_mask, KEYS])
    days = interval_start(scada["TimeStamp"]).dt.floor("D")
    train = _train_days(days, cfg)
    if "split" in scada:
        train &= scada["split"].eq("train")
    positions = np.flatnonzero(train.fillna(False).to_numpy(dtype=bool))
    result = pd.Series(False, index=scada.index, name=f"healthy_{target}", dtype=bool)
    if not len(positions):
        return result
    work = scada.iloc[positions][list(required)].copy().reset_index(drop=True)
    work["Day"] = days.iloc[positions].to_numpy()
    if work.duplicated(["StationId", "TimeStamp"]).any():
        raise ValueError("SCADA must be cleaned to unique turbine/timestamp keys")
    for column in (rpm, wind):
        work[column] = pd.to_numeric(work[column], errors="coerce").astype(float)
    good = _good_flags(work)
    # stop_runs expects nonmissing boolean outage flags; unknown means outage.
    work["outage_month"] = work["outage_month"].fillna(True).astype(bool)
    runs, _ = stop_runs(work, cfg["labels"]["targets"]["scada_stops"])
    in_stop = np.zeros(len(work), dtype=bool)
    for station, group in work.groupby("StationId", sort=False):
        r = runs.loc[runs["StationId"].eq(station)].sort_values("Start")
        if r.empty:
            continue
        stamps = group["TimeStamp"].to_numpy(dtype="datetime64[ns]")
        indices = (
            np.searchsorted(r["Start"].to_numpy(dtype="datetime64[ns]"), stamps, side="left") - 1
        )
        valid = indices >= 0
        in_stop[group.index.to_numpy()[valid]] = (
            stamps[valid] <= r["End"].to_numpy(dtype="datetime64[ns]")[indices[valid]]
        )
    keep = (
        pd.MultiIndex.from_frame(work[KEYS]).isin(healthy_keys)
        & good.to_numpy()
        & np.isfinite(work[rpm])
        & np.isfinite(work[wind])
        & work[wind].ge(cut_in)
        & ~in_stop
    )
    result.iloc[positions] = np.asarray(keep, dtype=bool)
    return result
