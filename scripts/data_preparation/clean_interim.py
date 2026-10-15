"""Clean interim turbine workbooks into processed copies, preserving their relative paths."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from wt_predmaint.data.clean_workbook import clean_workbook  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "data/interim")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/processed")
    args = parser.parse_args()
    sources = sorted(args.input_dir.rglob("*.xlsx"))
    if not sources:
        parser.error("No interim .xlsx files found")
    for source in sources:
        if source.name.startswith("~$"):
            continue
        output = args.output_dir / source.relative_to(args.input_dir)
        report = clean_workbook(source, output)
        kept = sum(s["kept_rows"] for s in report["sheets"].values())
        removed = sum(s["removed_rows"] for s in report["sheets"].values())
        print(f"{output}: kept {kept:,}; removed {removed:,}", flush=True)


if __name__ == "__main__":
    main()
