"""Cleaning layer for the raw 10-min SCADA tables.

Rules (docs/project/DECISIONS.md D-013 to D-016): de-duplicate on (StationId, TimeStamp); mask
sentinel values per table to NaN and count them per row; inner-join the three tables; assign each
row to the year of its interval START (TimeStamp is the interval end); flag, but never drop,
outage months and turbine-years with a suspect wind signal.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

KEYS = ["StationId", "TimeStamp"]
INTERVAL = pd.Timedelta(minutes=10)
FLAG_COLUMNS = ["n_masked", "outage_month", "suspect_wind"]


def interval_start(timestamp: pd.Series) -> pd.Series:
    """SCADA TimeStamp is the interval end; the interval starts 10 minutes earlier."""
    return timestamp - INTERVAL


def dedupe_keys(df: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
    """Drop duplicate (StationId, TimeStamp) rows.

    Returns the frame, the number of exact duplicate rows removed (month-boundary repeats), and
    the number of conflicting duplicates removed (same key, different values; the first is kept).
    """
    exact = df.drop_duplicates()
    out = exact.drop_duplicates(KEYS, keep="first")
    return out, len(df) - len(exact), len(exact) - len(out)


def mask_sentinels(df: pd.DataFrame, values: list[float]) -> tuple[pd.DataFrame, pd.Series]:
    """Set the given sentinel values to NaN in every value column; count masked cells per row."""
    value_cols = [c for c in df.columns if c not in KEYS]
    out = df.copy()
    if not values:
        return out, pd.Series(0, index=out.index, dtype="uint8")
    hit = out[value_cols].isin(values)
    out[value_cols] = out[value_cols].mask(hit)
    return out, hit.sum(axis=1).astype("uint8")


def join_tables(
    tables: Mapping[str, pd.DataFrame], sentinels: Mapping[str, list[float]]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Dedupe, mask and inner-join the SCADA tables; returns the joined frame and statistics."""
    stats: dict[str, Any] = {"tables": {}}
    joined: pd.DataFrame | None = None
    for name, raw in tables.items():
        df, n_exact, n_conflict = dedupe_keys(raw)
        df, masked = mask_sentinels(df, list(sentinels.get(name, [])))
        value_cols = [c for c in df.columns if c not in KEYS]
        per_col = {c: int(n) for c, n in df[value_cols].isna().sum().items() if n}
        stats["tables"][name] = {
            "rows_in": len(raw),
            "exact_duplicates": n_exact,
            "conflicting_duplicates": n_conflict,
            "rows_out": len(df),
            "masked_cells": int(masked.sum()),
        }
        stats.setdefault("nan_cells", {}).update({f"{name}.{c}": n for c, n in per_col.items()})
        df = df.assign(**{f"_nm_{name}": masked.values})
        joined = df if joined is None else joined.merge(df, on=KEYS, how="inner")
    assert joined is not None
    nm_cols = [c for c in joined.columns if c.startswith("_nm_")]
    joined["n_masked"] = joined[nm_cols].sum(axis=1).astype("uint8")
    joined = joined.drop(columns=nm_cols)
    stats["rows_joined"] = len(joined)
    return joined, stats


def add_outage_flag(
    df: pd.DataFrame, expected_turbines: int, coverage_threshold: float
) -> tuple[pd.DataFrame, list[str]]:
    """Flag rows in months whose coverage of the full 10-min grid is below the threshold."""
    month = interval_start(df["TimeStamp"]).dt.to_period("M")
    rows = df.groupby(month).size()
    expected = rows.index.to_series().dt.days_in_month * 144 * expected_turbines
    coverage = rows / expected
    bad = coverage[coverage < coverage_threshold].index
    out = df.assign(outage_month=month.isin(bad).values)
    return out, [str(m) for m in bad]


def add_suspect_wind_flag(
    df: pd.DataFrame, min_rpm: float, min_corr: float, min_rows: int
) -> tuple[pd.DataFrame, list[tuple[int, float]]]:
    """Flag turbines whose wind speed barely tracks generator speed while producing.

    Correlation of wtc_PrWindSp_mean with wtc_GenRpm_mean among producing rows (rpm >= min_rpm)
    is computed per turbine within the frame (call it per year). Turbines below min_corr with at
    least min_rows producing rows are flagged on all their rows.
    """
    producing = df[df["wtc_GenRpm_mean"] >= min_rpm]
    suspect: list[tuple[int, float]] = []
    for station, g in producing.groupby("StationId"):
        if len(g) < min_rows:
            continue
        corr = g["wtc_PrWindSp_mean"].corr(g["wtc_GenRpm_mean"])
        if pd.notna(corr) and corr < min_corr:
            suspect.append((int(station), float(corr)))
    flagged = {s for s, _ in suspect}
    return df.assign(suspect_wind=df["StationId"].isin(flagged).values), suspect


def clean_year(
    tables: Mapping[str, pd.DataFrame], year: int, cfg: dict[str, Any]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Clean one year: join, keep rows whose interval starts in `year`, add the flags."""
    c = cfg["cleaning"]
    df, stats = join_tables(tables, c["mask_sentinels"])
    df = df[interval_start(df["TimeStamp"]).dt.year == year]
    df, outage = add_outage_flag(df, c["expected_turbines"], c["outage_coverage_threshold"])
    df, suspect = add_suspect_wind_flag(
        df, c["producing_min_rpm"], c["suspect_wind_min_corr"], c["suspect_wind_min_rows"]
    )
    stats.update(
        year=year,
        rows_final=len(df),
        outage_months=outage,
        suspect_wind_turbines=[{"station": s, "corr": round(r, 3)} for s, r in suspect],
        flagged_rows={"outage_month": int(df["outage_month"].sum()),
                      "suspect_wind": int(df["suspect_wind"].sum()),
                      "any_masked": int((df["n_masked"] > 0).sum())},
    )  # fmt: skip
    return df.reset_index(drop=True), stats


def read_scada_parquet(scada_dir: Path, table: str, year: int) -> pd.DataFrame:
    return pd.read_parquet(scada_dir / f"{table}_{year}.parquet")
