"""Reproduce H0 counts per turbine/year/target for the configured 2016–2025 scope.

Uses shared prepared features/labels and clean SCADA, never Excel workbooks.
Missing yearly clean files are reported as unavailable (blank interval counts),
not silently counted as zero. This does not train or evaluate a baseline model.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from wt_predmaint.config import load_protocol
from wt_predmaint.data.clean import interval_start
from wt_predmaint.models.healthy import healthy_days, healthy_intervals

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interim", type=Path, default=ROOT / "data/interim")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/protocol.yaml")
    parser.add_argument("--targets", nargs="+", choices=["a", "b"], default=["a", "b"])
    parser.add_argument("--output", type=Path, default=ROOT / "reports/tables/healthy_rows.csv")
    args = parser.parse_args()
    cfg = load_protocol(args.config)
    start, end = cfg["healthy"]["start_year"], cfg["healthy"]["end_year"]
    features = pd.read_parquet(
        args.interim / "features/daily_features.parquet",
        columns=["StationId", "Day", "outage_month", "suspect_wind"],
    )
    features = features.loc[features["Day"].dt.year.between(start, end)].copy()
    if features.empty:
        parser.error("No prepared features in the configured healthy year range")
    stations = sorted(features["StationId"].unique())
    rows: list[dict] = []
    labels_by_target = {
        target: pd.read_parquet(args.interim / f"labels/{target}_labels.parquet")
        for target in args.targets
    }
    masks = {t: healthy_days(labels_by_target[t], features, cfg, t) for t in args.targets}
    columns = [
        "StationId",
        "TimeStamp",
        "wtc_GenRpm_mean",
        "wtc_PrWindSp_mean",
        "outage_month",
        "suspect_wind",
    ]
    training_files = [
        args.interim / f"clean/scada_{year}.parquet"
        for year in cfg["splits"]["train_years"]
        if start <= year <= end
    ]
    available_train = [pd.read_parquet(p, columns=columns) for p in training_files if p.exists()]
    history = pd.concat(available_train, ignore_index=True) if available_train else None
    counts = {}
    if history is not None:
        history_year = interval_start(history["TimeStamp"]).dt.year
        for target in args.targets:
            selected = healthy_intervals(labels_by_target[target], features, history, cfg, target)
            counts[target] = selected.groupby([history["StationId"], history_year]).sum()
    for year in range(start, end + 1):
        path = args.interim / f"clean/scada_{year}.parquet"
        # A year file's interval START already belongs to that year (clean_year).
        if path.exists() and year in cfg["splits"]["train_years"]:
            scada = history.loc[history_year.eq(year)]
        else:
            scada = pd.read_parquet(path, columns=columns) if path.exists() else None
        for station in stations:
            day_rows = features["StationId"].eq(station) & features["Day"].dt.year.eq(year)
            interval_rows = (
                scada["StationId"].eq(station) & interval_start(scada["TimeStamp"]).dt.year.eq(year)
                if scada is not None
                else None
            )
            for target in args.targets:
                kept = int((day_rows & masks[target]).sum())
                if year not in cfg["splits"]["train_years"] and kept:
                    raise AssertionError("Nontraining healthy days reached the report")
                rows.append(
                    {
                        "target": target.upper(),
                        "StationId": station,
                        "year": year,
                        "is_train_year": year in cfg["splits"]["train_years"],
                        "feature_days": int(day_rows.sum()),
                        "healthy_days": kept,
                        "clean_scada_file_available": scada is not None,
                        "interval_rows": int(interval_rows.sum()) if scada is not None else None,
                        "healthy_intervals": int(counts.get(target, {}).get((station, year), 0))
                        if scada is not None
                        else None,
                    }
                )
        print(
            f"{year}: clean SCADA {'available' if scada is not None else 'unavailable'}", flush=True
        )
    report = pd.DataFrame(rows)
    report[["interval_rows", "healthy_intervals"]] = report[
        ["interval_rows", "healthy_intervals"]
    ].astype("Int64")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(args.output, index=False)
    print(
        report.groupby("target")[["feature_days", "healthy_days", "healthy_intervals"]]
        .sum()
        .to_string()
    )
    print(f"Wrote {args.output}; blank interval counts mean unavailable source files.")


if __name__ == "__main__":
    main()
