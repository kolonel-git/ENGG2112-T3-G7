"""Plot one turbine's timeline with bands that show why days are blank.

Until models exist, the plotted column is any column of the feature table (for example a 7-day
sensor mean). Once a model writes scores, pass them with --scores (a parquet with StationId, Day
and a column named by --column).

    python scripts/plot_risk_timeline.py --target a --station 2304510 --column alarm_total_w7
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import pandas as pd  # noqa: E402

from wt_predmaint.evaluation.plotting import (  # noqa: E402
    blank_reasons,
    calendar_frame,
    plot_risk_timeline,
    status_summary,
)

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--interim", type=Path, default=ROOT / "data" / "interim")
    p.add_argument("--target", choices=["a", "b"], default="a")
    p.add_argument("--station", type=int, help="turbine id (default: first in the labels)")
    p.add_argument("--column", default="alarm_total_w7")
    p.add_argument("--scores", type=Path, help="parquet with StationId, Day and --column")
    p.add_argument("--start")
    p.add_argument("--end")
    p.add_argument("--out", type=Path, default=ROOT / "reports" / "figures")
    args = p.parse_args()

    labels = pd.read_parquet(args.interim / "labels" / f"{args.target}_labels.parquet")
    episodes = pd.read_parquet(args.interim / "labels" / f"{args.target}_episodes.parquet")
    source = args.scores or args.interim / "features" / "daily_features.parquet"
    values = pd.read_parquet(source, columns=["StationId", "Day", args.column])
    station = args.station or int(labels["StationId"].min())

    cal = calendar_frame(values, labels, station, args.column, args.start, args.end)
    fig = plot_risk_timeline(
        cal,
        episodes,
        station,
        ylabel=args.column,
        title=f"Turbine {station}, target {args.target.upper()}: {args.column}",
    )
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"timeline_{args.target}_{station}_{args.column}.png"
    fig.savefig(path, dpi=150)
    print(f"wrote {path}")
    print(status_summary(cal).to_string())
    print("days with a missing value, by reason:")
    print(blank_reasons(cal).astype(int).to_string())


if __name__ == "__main__":
    main()
