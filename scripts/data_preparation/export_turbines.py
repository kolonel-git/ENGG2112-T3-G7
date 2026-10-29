"""Export selected turbine features into one Excel workbook, one sheet per StationId."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from wt_predmaint.data.turbine_export import export_turbines  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--pattern", default="*.csv", help="E.g. *_2018_06.csv")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "data/interim/turbine_exports/turbines.xlsx")
    args = parser.parse_args()
    report = export_turbines(args.input_dir, args.output, args.pattern)
    print(json.dumps(report, indent=2))
    print(f"Workbook: {args.output.resolve()}")


if __name__ == "__main__":
    main()
