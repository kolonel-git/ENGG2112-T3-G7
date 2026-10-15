"""Combine monthly raw CSVs into an annual workbook, one sheet per StationId."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from wt_predmaint.data.clean_workbook import clean_workbook  # noqa: E402
from wt_predmaint.data.turbine_export import export_turbines  # noqa: E402

# CHANGE ONLY THIS folder name to process another year under data/raw/.
# Example: "2019" reads data/raw/2019 and writes turbines_2019.xlsx.
YEAR_FOLDER = "2018"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", default=YEAR_FOLDER, help="Year folder under data/raw")
    parser.add_argument("--input-dir", type=Path, help="Override data/raw/<year> input folder")
    parser.add_argument("--pattern", help="Override year selection, e.g. *_2018_06.csv")
    parser.add_argument("--output", type=Path, help="Override processed workbook path")
    parser.add_argument("--interim-output", type=Path, help="Override uncleaned workbook path")
    parser.add_argument("--keep-missing", action="store_true",
                        help="Export directly to --output without removing incomplete rows")
    args = parser.parse_args()
    if not (len(args.year) == 4 and args.year.isascii() and args.year.isdigit()):
        parser.error("Year folder must be a four-digit year, for example 2018 or 2019")
    input_dir = args.input_dir or ROOT / "data" / "raw" / args.year
    if not input_dir.is_dir():
        parser.error(f"Input folder does not exist: {input_dir}")
    pattern = args.pattern or f"*_{args.year}_*.csv"
    output = args.output or ROOT / f"data/processed/turbine_exports/turbines_{args.year}.xlsx"
    interim = args.interim_output or ROOT / (
        f"data/interim/turbine_exports/turbines_{args.year}.xlsx"
    )
    if not args.keep_missing and output.resolve() == interim.resolve():
        parser.error("Processed output and interim output must have different paths")
    print(f"Reading year folder: {input_dir.resolve()}", flush=True)
    report = export_turbines(input_dir, output if args.keep_missing else interim, pattern)
    print(json.dumps(report, indent=2))
    if not args.keep_missing:
        cleaned = clean_workbook(interim, output)
        kept = sum(s["kept_rows"] for s in cleaned["sheets"].values())
        removed = sum(s["removed_rows"] for s in cleaned["sheets"].values())
        print(f"Processed rows: {kept:,} kept; {removed:,} removed")
        print(f"Uncleaned workbook: {interim.resolve()}")
    print(f"Workbook: {output.resolve()}")


if __name__ == "__main__":
    main()
