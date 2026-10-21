"""Read key columns of the raw SCADA tables (tblSCTurbine, tblSCTurTemp, tblSCTurPress).

TimeStamp marks the END of each 10-min interval. Files re-saved in Excel (e.g. 2025-01) use
month-first 'M/D/YYYY H:MM' timestamps; see parse_times in wt_predmaint.data.alarms.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from wt_predmaint.data.alarms import SITE_STATION, parse_times

TABLES = ("tblSCTurbine", "tblSCTurTemp", "tblSCTurPress")
EXTRA_COLUMNS = frozenset(
    {
        "wtc_PrWindSp_stddev",
        "wtc_GenRpm_stddev",
        "wtc_PowerRef_endvalue",
        "wtc_WTOperSt_endvalue",
    }
)


def key_columns(header: list[str]) -> list[str]:
    """Identifier columns, every *_mean column, and the few extra signals we need."""
    keep = [c for c in header if c in ("TimeStamp", "StationId")]
    keep += [c for c in header if c.endswith("_mean") or c in EXTRA_COLUMNS]
    return keep


def read_scada_file(path: Path) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0).columns.tolist()
    df = pd.read_csv(path, usecols=key_columns(header), low_memory=False)
    df["TimeStamp"] = parse_times(df["TimeStamp"].astype(str), path.name)
    df["StationId"] = df["StationId"].astype(int)
    df = df[df["StationId"] != SITE_STATION]
    value_cols = [c for c in df.columns if c not in ("TimeStamp", "StationId")]
    df[value_cols] = df[value_cols].astype("float32")
    return df


def read_scada_year(root: Path, table: str, year: int) -> pd.DataFrame:
    files = sorted((root / str(year)).glob(f"{table}_{year}_*.csv"))
    if not files:
        raise FileNotFoundError(f"no {table} files for {year} under {root}")
    return pd.concat([read_scada_file(f) for f in files], ignore_index=True)
