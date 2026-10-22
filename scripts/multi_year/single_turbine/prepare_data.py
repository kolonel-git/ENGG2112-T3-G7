"""Prepare 2018–2025 raw data for a station-isolated multi-year experiment."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from wt_predmaint.config import load_protocol
from wt_predmaint.data.single_turbine_pipeline import experiment_config, prepare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--station", type=int, default=2304510)
    parser.add_argument("--start-year", type=int, default=2018)
    parser.add_argument("--train-end", type=int, default=2021)
    parser.add_argument("--calibration-year", type=int, default=2022)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--raw", type=Path, default=ROOT / "data/raw")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--targets", nargs="+", choices=["a", "b"], default=["a"])
    args = parser.parse_args()
    cfg = experiment_config(load_protocol(), args.start_year, args.train_end,
                            args.calibration_year, args.end_year)
    prepare(args.raw, args.output_dir, args.station, cfg, args.rebuild, tuple(args.targets))


if __name__ == "__main__":
    main()
