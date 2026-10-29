"""Build data/interim/alarm_events.parquet from the raw tblAlarmLog files in data/raw.

Reads tblAlarmLog CSVs (loose or inside the year zips), keeps the in-scope codes from
configs/protocol.yaml, and prints event counts per code and year.

    python scripts/build_alarm_table.py
    python scripts/build_alarm_table.py --raw data/raw --keep-other
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Support running this file directly from an IDE without installing the package.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from wt_predmaint.config import load_protocol
from wt_predmaint.data.alarms import (
    alarm_code_table,
    build_alarm_events,
    count_summary,
    iter_alarm_log_files,
    load_alarm_log,
    other_alarm_events,
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--raw", type=Path, default=ROOT / "data" / "raw")
    p.add_argument("--out", type=Path, default=ROOT / "data" / "interim")
    p.add_argument(
        "--keep-other",
        action="store_true",
        help="also write out-of-scope events (alarm_other.parquet) for row masking",
    )
    args = p.parse_args()

    codes = alarm_code_table(load_protocol())
    raw = load_alarm_log(iter_alarm_log_files(args.raw))
    print(f"raw records: {len(raw):,}  {raw['TimeOn'].min()} to {raw['TimeOn'].max()}")

    events = build_alarm_events(raw, codes)
    print(f"in-scope events: {len(events):,} (dropped {len(raw) - len(events):,})")
    print(count_summary(events).to_string())

    args.out.mkdir(parents=True, exist_ok=True)
    events.to_parquet(args.out / "alarm_events.parquet")
    if args.keep_other:
        other_alarm_events(raw, codes).to_parquet(args.out / "alarm_other.parquet")
    print(f"wrote {args.out / 'alarm_events.parquet'}")


if __name__ == "__main__":
    main()
