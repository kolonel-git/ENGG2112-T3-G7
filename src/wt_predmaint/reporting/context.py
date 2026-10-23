"""Lazy access to the project's data and tables for figure functions."""

from __future__ import annotations

from functools import cached_property
from pathlib import Path
from typing import Any

import pandas as pd

from wt_predmaint.config import load_protocol

ROOT = Path(__file__).resolve().parents[3]
GRID_PER_DAY = 144


class Ctx:
    """Paths plus cached loaders. Figures read only what they need."""

    def __init__(self, interim: Path | None = None, tables: Path | None = None) -> None:
        self.interim = interim or ROOT / "data" / "interim"
        self.tables_dir = tables or ROOT / "reports" / "tables"
        self.cfg: dict[str, Any] = load_protocol()
        self._labels: dict[str, pd.DataFrame] = {}
        self._episodes: dict[str, pd.DataFrame] = {}

    @cached_property
    def features(self) -> pd.DataFrame:
        return pd.read_parquet(self.interim / "features" / "daily_features.parquet")

    @cached_property
    def daily(self) -> pd.DataFrame:
        return pd.read_parquet(self.interim / "clean" / "daily_summary.parquet")

    @cached_property
    def alarms(self) -> pd.DataFrame:
        return pd.read_parquet(self.interim / "alarm_events.parquet")

    def labels(self, target: str) -> pd.DataFrame:
        if target not in self._labels:
            self._labels[target] = pd.read_parquet(
                self.interim / "labels" / f"{target}_labels.parquet"
            )
        return self._labels[target]

    def episodes(self, target: str) -> pd.DataFrame:
        if target not in self._episodes:
            self._episodes[target] = pd.read_parquet(
                self.interim / "labels" / f"{target}_episodes.parquet"
            )
        return self._episodes[target]

    def table(self, name: str) -> pd.DataFrame:
        return pd.read_csv(self.tables_dir / f"{name}.csv")

    def scores(
        self, target: str, model: str, split: str = "val", variant: str = "all"
    ) -> pd.DataFrame:
        path = self.interim / "scores" / f"{target}_{variant}_{model}_{split}.parquet"
        return pd.read_parquet(path)

    @cached_property
    def coverage(self) -> pd.DataFrame:
        """Share of the full 10-minute grid present per turbine and month (interval-start month)."""
        parts = []
        for f in sorted((self.interim / "clean").glob("scada_*.parquet")):
            d = pd.read_parquet(f, columns=["StationId", "TimeStamp"])
            month = (d["TimeStamp"] - pd.Timedelta(minutes=10)).dt.to_period("M")
            parts.append(d.groupby(["StationId", month]).size().rename("rows").reset_index())
        out = pd.concat(parts, ignore_index=True)
        out["TimeStamp"] = out["TimeStamp"].astype("period[M]")
        out["share"] = (out["rows"] / (out["TimeStamp"].dt.days_in_month * GRID_PER_DAY)).clip(
            upper=1.0
        )
        return out.rename(columns={"TimeStamp": "month"})
