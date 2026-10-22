"""Run preparation, model fitting, held-out evaluation, and plots in sequence."""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
STATION = 2304510
START_YEAR, TRAIN_END, CALIBRATION_YEAR, END_YEAR = 2018, 2021, 2022, 2025


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--station", type=int, default=STATION)
    parser.add_argument("--start-year", type=int, default=START_YEAR)
    parser.add_argument("--train-end", type=int, default=TRAIN_END)
    parser.add_argument("--calibration-year", type=int, default=CALIBRATION_YEAR)
    parser.add_argument("--end-year", type=int, default=END_YEAR)
    parser.add_argument("--targets", nargs="+", choices=["a", "b"], default=["a"],
                        help="Default: daily-fault-time target A only")
    parser.add_argument("--raw", type=Path, default=ROOT / "data/raw")
    parser.add_argument("--work-dir", type=Path, help="Optional experiment folder for prepared/results/figures")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild even unchanged raw-data caches")
    parser.add_argument("--skip-prepare", action="store_true", help="Explicitly reuse existing prepared data")
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT / "src"))
    from wt_predmaint.config import load_protocol
    from wt_predmaint.data.single_turbine_pipeline import experiment_config
    cfg = experiment_config(load_protocol(), args.start_year, args.train_end,
                            args.calibration_year, args.end_year)
    scope = f"{args.start_year}_{args.end_year}/turbine_{args.station}"
    prepared = ROOT / "data/interim/multi_year" / scope
    results = ROOT / "data/processed/multi_year" / scope
    figures = ROOT / "reports/figures/multi_year" / scope
    if args.work_dir:
        prepared = args.work_dir.resolve() / "prepared"
        results = args.work_dir.resolve() / "results"
        figures = args.work_dir.resolve() / "figures"
    folder = Path(__file__).resolve().parent
    if args.skip_prepare:
        import json
        saved = json.loads((prepared / "preparation.json").read_text(encoding="utf-8"))
        if saved["protocol"] != cfg or saved["station"] != args.station:
            parser.error("Prepared configuration does not match this run; omit --skip-prepare")
        if not set(args.targets).issubset(saved.get("targets", ["a", "b"])):
            parser.error("Requested targets were not prepared; omit --skip-prepare")
    steps = []
    if not args.skip_prepare:
        arguments = ["--station", str(args.station), "--start-year", str(args.start_year),
                     "--train-end", str(args.train_end), "--calibration-year", str(args.calibration_year),
                     "--end-year", str(args.end_year), "--raw", str(args.raw.resolve()),
                     "--output-dir", str(prepared), "--targets", *args.targets]
        steps.append(("prepare_data.py", arguments + (["--rebuild"] if args.rebuild else [])))
    steps.extend([
        ("train_classifiers.py", ["--input-dir", str(prepared), "--output-dir", str(results),
                                  "--targets", *args.targets]),
        ("plot_results.py", ["--input-dir", str(results), "--output-dir", str(figures)]),
    ])
    for script, arguments in steps:
        print(f"\nRunning {script}", flush=True)
        try:
            subprocess.run([sys.executable, str(folder / script), *arguments], cwd=ROOT, check=True)
        except subprocess.CalledProcessError as exc:
            parser.exit(exc.returncode, f"Stopped: {script} failed. Later stages were not run.\n")
    print(f"\nFinished. Results: {results}\nFigures: {figures}")


if __name__ == "__main__":
    main()
