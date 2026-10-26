"""Fit the ridge normal-behaviour models on healthy train days and write residual features (D-054).

Writes ``data/interim/features/nbm_residuals.parquet`` (daily and smoothed standardised residuals
plus the indicator, for every day with features) and three tables in ``reports/tables``:
``nbm_fit.csv`` (fit and validation-year check per sensor), ``nbm_parameters.csv`` (coefficients)
and ``nbm_lead.csv`` (indicator against time to the next episode, validation years only).
The fit uses healthy train days of both targets (A and B); test years are never read here.

    python scripts/build_nbm.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from wt_predmaint.config import load_protocol
from wt_predmaint.models.base import KEYS, attach_labels
from wt_predmaint.models.healthy import healthy_days
from wt_predmaint.models.nbm import NormalBehaviourModel, indicator, short_name, smooth_residuals

ROOT = Path(__file__).resolve().parents[1]
LEAD_BINS = [
    (0, 7, "1-7 days"),
    (7, 14, "8-14 days"),
    (14, 30, "15-30 days"),
    (30, 60, "31-60 days"),
]


def lead_table(
    frame: pd.DataFrame, episodes: pd.DataFrame, labels: pd.DataFrame, target: str
) -> pd.DataFrame:
    """Mean indicator by days to the next episode start; in-episode days are excluded."""
    days = frame[[*KEYS, "nbm_indicator"]].dropna()
    days = days.merge(labels[[*KEYS, "status"]], on=KEYS, how="inner")
    days = days.loc[days["status"].ne("in_episode")].sort_values("Day")
    starts = episodes[["StationId", "Start"]].sort_values("Start")
    nxt = pd.merge_asof(
        days, starts, left_on="Day", right_on="Start", by="StationId", direction="forward",
        allow_exact_matches=False,
    )  # fmt: skip
    gap = (nxt["Start"] - nxt["Day"]).dt.days
    rows = []
    for low, high, name in LEAD_BINS:
        sel = nxt.loc[(gap > low) & (gap <= high), "nbm_indicator"]
        rows.append((name, sel))
    rows.append(("more than 60 days or none", nxt.loc[~(gap <= 60), "nbm_indicator"]))
    return pd.DataFrame(
        [
            {
                "target": target,
                "time_to_next_episode": name,
                "turbine_days": len(sel),
                "mean_indicator": sel.mean(),
                "median_indicator": sel.median(),
            }
            for name, sel in rows
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interim", type=Path, default=ROOT / "data/interim")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/protocol.yaml")
    parser.add_argument("--tables", type=Path, default=ROOT / "reports/tables")
    args = parser.parse_args()
    cfg = load_protocol(args.config)
    nbm = cfg["nbm"]

    features = pd.read_parquet(args.interim / "features/daily_features.parquet")
    labels = {t: pd.read_parquet(args.interim / f"labels/{t}_labels.parquet") for t in "ab"}
    episodes = {t: pd.read_parquet(args.interim / f"labels/{t}_episodes.parquet") for t in "ab"}
    stations = cfg["baseline_table"]["exclude_stations"]
    features = features.loc[~features["StationId"].isin(stations)].reset_index(drop=True)

    masks = {t: healthy_days(labels[t], features, cfg, t) for t in "ab"}
    healthy = masks["a"] & masks["b"]
    counts = {t: int(m.sum()) for t, m in masks.items()}
    print(f"healthy train days: A {counts['a']}, B {counts['b']}, both {int(healthy.sum())}")

    model = NormalBehaviourModel(cfg).fit(features, healthy)
    res = model.residuals(features)
    smooth = smooth_residuals(
        res, features[KEYS], int(nbm["smoothing_days"]), int(cfg["features"]["window_min_days"])
    )
    out = pd.concat([features[KEYS], res, smooth], axis=1)
    out["nbm_indicator"] = indicator(smooth)
    args.tables.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.interim / "features/nbm_residuals.parquet", index=False)
    model.parameters().to_csv(args.tables / "nbm_parameters.csv", index=False)

    # Validation-year check on days that look healthy: negative label, clean flags, no episode zone.
    table = attach_labels(features[[*KEYS, "outage_month", "suspect_wind"]], labels["a"], cfg)
    calm = table["split"].eq("val") & table["status"].eq("negative")
    calm &= table["outage_month"].eq(False) & table["suspect_wind"].eq(False)
    fit_rows = []
    for _, p in model.parameters().iterrows():
        col = f"nbm_res_{p['sensor']}"
        r = res.loc[calm.to_numpy(), col].dropna()
        fit_rows.append(
            {
                "sensor": p["sensor"],
                "n_fit_train": int(p["n_fit"]),
                "r2_train": p["r2_train"],
                "val_days": len(r),
                "val_mean_z": r.mean(),
                "val_std_z": r.std(),
            }
        )
    pd.DataFrame(fit_rows).to_csv(args.tables / "nbm_fit.csv", index=False)

    val = out.merge(table[[*KEYS, "split"]], on=KEYS).query("split == 'val'")
    leads = [lead_table(val, episodes[t], labels[t], t) for t in "ab"]
    pd.concat(leads).to_csv(args.tables / "nbm_lead.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(pd.DataFrame(fit_rows).round(3))
        print(pd.concat(leads).round(3))
    print(f"sensors modelled: {[short_name(t) for t in model.fitted]}; wrote {len(out)} rows")
    assert np.isfinite(out["nbm_indicator"].dropna()).all()


if __name__ == "__main__":
    main()
