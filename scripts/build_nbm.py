"""Fit the ridge normal-behaviour models on healthy train days and write residual features.

Writes ``data/interim/features/nbm_residuals.parquet`` (daily and smoothed standardised residuals
plus the indicator, for every day with features) and tables in ``reports/tables``:

* ``nbm_fit.csv``: fit and validation-year check per sensor;
* ``nbm_parameters.csv``: coefficients;
* ``nbm_lead.csv``: indicator against time to the next episode for the saved variant (validation);
* ``nbm_lead_variants.csv``: the same for own-turbine and fleet-relative residuals;
* ``nbm_fleet_compare.csv``: validation drift of both variants (D-055).

The fit uses healthy train days of both targets (A and B) and leaves out the turbines in
``baseline_table.exclude_stations``, which are still scored. Test years are never read here.
Which variant is saved follows ``nbm.fleet_relative`` in the protocol.

    python scripts/build_nbm.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from wt_predmaint.config import load_protocol
from wt_predmaint.models.base import KEYS, attach_labels
from wt_predmaint.models.nbm import fit_nbm, nbm_features, short_name

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
    starts = episodes[["StationId", "Start"]].copy()
    # Parquet inputs can encode StationId with different integer widths.  merge_asof
    # requires the grouping key to have exactly the same dtype on both sides.
    starts["StationId"] = starts["StationId"].astype(days["StationId"].dtype)
    starts = starts.sort_values("Start")
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


def validation_check(out: pd.DataFrame, calm: pd.Series, sensors: list[str]) -> pd.DataFrame:
    """Mean and sd of each daily residual on healthy-looking validation days."""
    rows = []
    for sensor in sensors:
        r = out.loc[calm.to_numpy(), f"nbm_res_{sensor}"].dropna()
        rows.append(
            {"sensor": sensor, "val_days": len(r), "val_mean_z": r.mean(), "val_std_z": r.std()}
        )
    return pd.DataFrame(rows)


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
    held_out = cfg["baseline_table"]["exclude_stations"]
    model, healthy = fit_nbm(features, labels, cfg)
    print(f"healthy train days used for the fit: {int(healthy.sum())}")
    sensors = [short_name(t) for t in model.fitted]
    args.tables.mkdir(parents=True, exist_ok=True)
    model.parameters().to_csv(args.tables / "nbm_parameters.csv", index=False)

    # Validation-year check on days that look healthy: negative label, clean flags, fitted turbines.
    table = attach_labels(features[[*KEYS, "outage_month", "suspect_wind"]], labels["a"], cfg)
    calm = table["split"].eq("val") & table["status"].eq("negative")
    calm &= table["outage_month"].eq(False) & table["suspect_wind"].eq(False)
    calm &= ~table["StationId"].isin(held_out)

    variants = {name: nbm_features(model, features, cfg, fleet) for name, fleet in
                (("own_turbine", False), ("fleet_relative", True))}  # fmt: skip
    compare = []
    for name, out in variants.items():
        check = validation_check(out, calm, sensors)
        compare.append(check.assign(variant=name))
        print(
            f"{name}: mean |val mean z| {check['val_mean_z'].abs().mean():.3f}, "
            f"mean val sd z {check['val_std_z'].mean():.3f}"
        )
    pd.concat(compare).to_csv(args.tables / "nbm_fleet_compare.csv", index=False)

    chosen = "fleet_relative" if nbm["fleet_relative"] else "own_turbine"
    out = variants[chosen]
    out.to_parquet(args.interim / "features/nbm_residuals.parquet", index=False)
    fit = model.parameters()[["sensor", "n_fit", "r2_train"]].rename(
        columns={"n_fit": "n_fit_train"}
    )
    fit = fit.merge(validation_check(out, calm, sensors), on="sensor")
    fit.to_csv(args.tables / "nbm_fit.csv", index=False)

    split = table[[*KEYS, "split"]]
    lead_frames = []
    for name, frame in variants.items():
        val = frame.merge(split, on=KEYS).query("split == 'val'")
        both = pd.concat([lead_table(val, episodes[t], labels[t], t) for t in "ab"])
        lead_frames.append(both.assign(variant=name))
    pd.concat(lead_frames).to_csv(args.tables / "nbm_lead_variants.csv", index=False)
    leads = [f for f, name in zip(lead_frames, variants, strict=True) if name == chosen]
    leads[0].drop(columns="variant").to_csv(args.tables / "nbm_lead.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(f"saved variant: {chosen}")
        print(fit.round(3))
        print(pd.concat(lead_frames).round(3).to_string())
    print(f"sensors modelled: {sensors}; wrote {len(out)} rows")


if __name__ == "__main__":
    main()
