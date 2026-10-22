"""Run B3 and reference floors on validation for A and B; never test by default."""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from wt_predmaint.config import load_protocol
from wt_predmaint.evaluation.runner import run_baselines
from wt_predmaint.models.base import attach_labels
from wt_predmaint.models.power_curve import PowerCurveProxy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=["a", "b", "both"], default="both")
    parser.add_argument("--interim", type=Path, default=ROOT / "data/interim")
    parser.add_argument("--resamples", type=int)
    args = parser.parse_args()
    cfg = load_protocol()
    features = pd.read_parquet(args.interim / "features/daily_features.parquet")
    out = ROOT / "reports/tables"
    figures = ROOT / "reports/figures"
    out.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    for target in ("a", "b") if args.target == "both" else (args.target,):
        labels = pd.read_parquet(args.interim / f"labels/{target}_labels.parquet")
        episodes = pd.read_parquet(args.interim / f"labels/{target}_episodes.parquet")
        model = PowerCurveProxy(target, interim=args.interim)
        table = attach_labels(features, labels, cfg)
        if not ((table.split == "val") & table.label.notna()).any():
            raise ValueError("Prepare labelled validation years before running B3")
        result = run_baselines([model], table, episodes, cfg, target, resamples=args.resamples)
        result.table.to_csv(out / f"b3_validation_{target}.csv", index=False)
        model.curve.to_csv(out / f"b3_curve_{target}.csv", index=False)
        model.daily.to_parquet(args.interim / f"features/b3_daily_{target}.parquet", index=False)
        pd.DataFrame(
            [
                {
                    "target": target,
                    "healthy_fit_intervals": model.fit_intervals,
                    "tolerance_rpm": model.tolerance,
                }
            ]
        ).to_csv(out / f"b3_fit_{target}.csv", index=False)
        availability = (
            model.daily.assign(year=model.daily.Day.dt.year)
            .groupby(["StationId", "year"])
            .agg(days=("b3_available", "size"), available=("b3_available", "sum"))
        )
        availability.to_csv(out / f"b3_availability_{target}.csv")
        scores_dir = args.interim / "scores"
        scores_dir.mkdir(parents=True, exist_ok=True)
        for (variant, name, split), scores in result.scores.items():
            scores.to_parquet(
                scores_dir / f"{target}_{variant}_{name}_{split}.parquet", index=False
            )
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(model.curve.wind, model.curve.rpm, marker=".", label="Healthy pooled median RPM")
        ax.set(
            xlabel="Primary wind speed (m/s)",
            ylabel="Generator RPM",
            title=f"B3 target {target.upper()}: wind/RPM proxy (not measured power)",
        )
        ax.legend()
        fig.tight_layout()
        fig.savefig(figures / f"b3_curve_{target}.png", dpi=150)
        plt.close(fig)
        print(result.table.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
