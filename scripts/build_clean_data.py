"""Build the cleaned SCADA and daily-summary tables.

Needs data/interim/scada/ (python scripts/build_scada_parquet.py) and the raw tblDailySummary files.
Writes data/interim/clean/scada_<year>.parquet, daily_summary.parquet and cleaning_report.json,
and a short table to reports/tables/cleaning_report.csv.

    python scripts/build_clean_data.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from wt_predmaint.config import load_protocol
from wt_predmaint.data.clean import clean_year, read_scada_parquet
from wt_predmaint.data.daily import load_daily_summary
from wt_predmaint.data.scada import TABLES

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--interim", type=Path, default=ROOT / "data" / "interim")
    p.add_argument("--raw", type=Path, default=ROOT / "data" / "raw")
    args = p.parse_args()
    cfg = load_protocol()
    scada_dir, out = args.interim / "scada", args.interim / "clean"
    out.mkdir(parents=True, exist_ok=True)

    report: dict[str, object] = {}
    table_rows = []
    for year in range(2016, 2027):
        if not all((scada_dir / f"{t}_{year}.parquet").exists() for t in TABLES):
            print(f"{year}: skipped (missing one of {TABLES})")
            continue
        tables = {t: read_scada_parquet(scada_dir, t, year) for t in TABLES}
        df, stats = clean_year(tables, year, cfg)
        df.to_parquet(out / f"scada_{year}.parquet")
        report[str(year)] = stats
        fl = stats["flagged_rows"]
        table_rows.append(
            {
                "year": year,
                "rows": stats["rows_final"],
                "exact_duplicates": sum(t["exact_duplicates"] for t in stats["tables"].values()),
                "conflicting_duplicates": sum(
                    t["conflicting_duplicates"] for t in stats["tables"].values()
                ),
                "rows_with_masked_cells": fl["any_masked"],
                "outage_month_rows": fl["outage_month"],
                "outage_months": " ".join(stats["outage_months"]),
                "suspect_wind_rows": fl["suspect_wind"],
                "suspect_wind_turbines": " ".join(
                    str(s["station"]) for s in stats["suspect_wind_turbines"]
                ),
            }
        )
        print(
            f"{year}: {stats['rows_final']:,} rows, outage {stats['outage_months']}, "
            f"suspect wind {[s['station'] for s in stats['suspect_wind_turbines']]}"
        )

    daily, dstats = load_daily_summary(args.raw, cfg)
    daily.to_parquet(out / "daily_summary.parquet")
    report["daily_summary"] = dstats
    print("daily:", dstats)

    (out / "cleaning_report.json").write_text(
        json.dumps(report, indent=1, default=str), encoding="utf-8"
    )
    tables_dir = ROOT / "reports" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(table_rows).to_csv(tables_dir / "cleaning_report.csv", index=False)
    print(f"wrote {out} and {tables_dir / 'cleaning_report.csv'}")


if __name__ == "__main__":
    main()
