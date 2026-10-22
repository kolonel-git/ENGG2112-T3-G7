"""Read key columns of the raw SCADA tables (tblSCTurbine, tblSCTurTemp, tblSCTurPress).

TimeStamp marks the END of each 10-min interval and is UTC (dataset dictionary). Files re-saved in
Excel (e.g. 2025-01) use month-first 'M/D/YYYY H:MM' timestamps; see parse_times in
wt_predmaint.data.alarms.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from wt_predmaint.data.alarms import SITE_STATION, parse_times

TABLES = ("tblSCTurbine", "tblSCTurTemp", "tblSCTurPress")

# Key pressure and temperature sensors (team feature lists). Their _min and _max are kept as well
# as _mean so that peaks inside a 10-min interval are not lost (D-016).
KEY_SENSORS = frozenset(
    {
        "wtc_GearPres", "wtc_HydPress", "wtc_HubPresA", "wtc_HubPresB", "wtc_HubPresC",
        "wtc_HSGenTmp", "wtc_HSRotTmp", "wtc_IMSGenTm", "wtc_IMSRotTm", "wtc_GeOilTmp",
        "wtc_GenBeGTm", "wtc_GenBeRTm", "wtc_MainBTmp", "wtc_Gen1U1Tm", "wtc_Gen1V1Tm",
        "wtc_Gen1W1Tm", "wtc_AmbieTmp", "wtc_NacelTmp", "wtc_ConvWTmp", "wtc_TraOilTF",
        "wtc_TraRooTF", "wtc_HydOilTm", "wtc_A1ExtTmp", "wtc_A3LefTmp", "wtc_A3RigTmp",
        "wtc_A9IntTmp", "wtc_A21IntTm", "wtc_DeltaTmp", "wtc_BrkTmpGn", "wtc_GFilB1Tm",
        "wtc_GFilB2Tm", "wtc_GFilB3Tm", "wtc_ReacUTmp", "wtc_ReacVTmp", "wtc_ReacWTmp",
        "wtc_InvUpTmp", "wtc_TrafoTMa",
    }
)  # fmt: skip
EXTRA_COLUMNS = frozenset(
    {
        "wtc_PrWindSp_stddev",
        "wtc_GenRpm_stddev",
        "wtc_PowerRef_endvalue",
        "wtc_WTOperSt_endvalue",
    }
)


def key_columns(header: list[str]) -> list[str]:
    """Identifiers, every *_mean column, min/max of the key sensors, and a few extra signals."""
    wanted = {f"{s}_{stat}" for s in KEY_SENSORS for stat in ("min", "max")}
    keep = [c for c in header if c in ("TimeStamp", "StationId")]
    keep += [c for c in header if c.endswith("_mean") or c in EXTRA_COLUMNS or c in wanted]
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
