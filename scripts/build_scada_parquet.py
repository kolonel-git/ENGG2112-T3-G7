"""Convert key columns of the raw SCADA CSVs to one parquet per table and year.

Output: data/interim/scada/<table>_<year>.parquet (git-ignored).

    python scripts/build_scada_parquet.py --workers 6
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

# Support running this file directly from an IDE without installing the package.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from wt_predmaint.data.scada import TABLES, read_scada_year




def convert(args: tuple[Path, Path, str, int]) -> str:
    raw, out, table, year = args
    dest = out / f"{table}_{year}.parquet"
    if dest.exists():
        return f"exists {dest.name} (delete it to rebuild)"
    try:
        df = read_scada_year(raw, table, year)
    except FileNotFoundError:
        return f"skip   {table} {year} (no files)"
    df.to_parquet(dest)
    return f"wrote  {dest.name}: {len(df):,} rows, {df.shape[1]} cols"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--raw", type=Path, default=ROOT / "data" / "raw")
    p.add_argument("--out", type=Path, default=ROOT / "data" / "interim" / "scada")
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    jobs = [(args.raw, args.out, t, y) for y in range(2016, 2027) for t in TABLES]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for line in pool.map(convert, jobs):
            print(line, flush=True)


if __name__ == "__main__":
    main()
