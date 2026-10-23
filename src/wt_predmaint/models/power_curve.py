"""B3 pooled healthy wind/RPM curve, D-031. Scores are not probabilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from wt_predmaint.data.clean import interval_start
from wt_predmaint.models.base import KEYS, require_train_only
from wt_predmaint.models.healthy import healthy_intervals

WIND = "wtc_PrWindSp_mean"
RPM = "wtc_GenRpm_mean"
COLUMNS = ["StationId", "TimeStamp", WIND, RPM, "outage_month", "suspect_wind"]


class PowerCurveProxy:
    """Baseline contract adapter with explicit optional in-memory sources for tests.

    Fit statistics use train only. Scoring is a key lookup into causal daily
    features computed with that fixed curve, independent of scoring batch/labels.
    Default disk sources are the shared interim tables; no source is modified.
    """

    name = "B3_power_curve"

    def __init__(
        self,
        target: str,
        interim: Path | None = None,
        scada: pd.DataFrame | None = None,
        labels: pd.DataFrame | None = None,
    ):
        if target not in {"a", "b"}:
            raise ValueError("target must be a or b")
        self.target = target
        self.interim = interim or Path(__file__).resolve().parents[3] / "data/interim"
        self.scada = scada
        self.labels = labels

    def _sources(self, years: list[int] | None = None):
        if self.scada is not None:
            data = self.scada[COLUMNS]
            if years is not None:
                data = data.loc[interval_start(data.TimeStamp).dt.year.isin(years)]
            yield data
        else:
            for path in sorted((self.interim / "clean").glob("scada_*.parquet")):
                year = int(path.stem.split("_")[-1])
                # Clean annual files already use interval-start years (D-013).
                if years is None or year in years:
                    yield pd.read_parquet(path, columns=COLUMNS)

    def _valid(self, scada: pd.DataFrame) -> pd.Series:
        return (
            np.isfinite(scada[WIND])
            & np.isfinite(scada[RPM])
            & scada[WIND].between(self.settings["wind_min_ms"], self.settings["wind_max_ms"])
            & scada.outage_month.eq(False)
            & scada.suspect_wind.eq(False)
        ).fillna(False)

    def expected_rpm(self, wind: np.ndarray) -> np.ndarray:
        """Interpolate supported bin centres; no extrapolation beyond learned centres."""
        return np.interp(wind, self.curve.wind, self.curve.rpm, left=np.nan, right=np.nan)

    def fit(self, train: pd.DataFrame, cfg: dict[str, Any]) -> PowerCurveProxy:
        require_train_only(train)
        if train.empty:
            raise ValueError("B3 needs labelled training rows")
        self.settings = cfg["power_curve"]
        width = float(self.settings["bin_width_ms"])
        if width <= 0 or self.settings["wind_min_ms"] >= self.settings["wind_max_ms"]:
            raise ValueError("Invalid B3 bin width or wind range")
        if self.settings["tolerance_std_multiple"] < 0:
            raise ValueError("B3 tolerance multiple must be nonnegative")
        years = cfg["splits"]["train_years"]
        parts = list(self._sources(years))
        if not parts:
            raise ValueError("No clean training SCADA for B3")
        history = pd.concat(parts, ignore_index=True)
        history = history.loc[interval_start(history.TimeStamp).dt.year.isin(years)].copy()
        lab = self.labels
        if lab is None:
            lab = pd.read_parquet(self.interim / f"labels/{self.target}_labels.parquet")
        lab = lab.loc[lab.Day.dt.year.isin(years) & lab.split.eq("train")]
        healthy = healthy_intervals(lab, train, history, cfg, self.target)
        fit = history.loc[healthy & self._valid(history), [WIND, RPM]].copy()
        if fit.empty:
            raise ValueError("No healthy working-range B3 training intervals")
        fit["bin"] = np.floor(fit[WIND] / width).astype(int)
        curve = fit.groupby("bin").agg(rpm=(RPM, "median"), count=(RPM, "size"))
        curve = curve.loc[curve["count"] >= self.settings["min_bin_intervals"]]
        curve["wind"] = (curve.index.to_numpy() + 0.5) * width
        self.curve = curve.reset_index()
        if len(self.curve) < 2:
            raise ValueError("B3 needs at least two supported wind bins")
        residual = self.expected_rpm(fit[WIND].to_numpy()) - fit[RPM].to_numpy()
        residual = residual[np.isfinite(residual)]
        self.tolerance = float(np.std(residual) * self.settings["tolerance_std_multiple"])
        self.fit_intervals = len(fit)
        del history, parts, fit
        self.cfg = cfg
        self._stats: dict[int, pd.DataFrame] = {}
        # Test data is not processed unless score() explicitly receives test rows.
        self._ensure_years(years + cfg["splits"]["val_years"])
        return self

    def _ensure_years(self, years: list[int]) -> None:
        requested = set(years) | {year - 1 for year in years}
        missing = requested - self._stats.keys()
        for year in sorted(missing):
            parts = [self.interval_daily(part) for part in self._sources([year])]
            if parts:
                self._stats[year] = pd.concat(parts, ignore_index=True)
            else:
                self._stats[year] = pd.DataFrame(
                    columns=[*KEYS, "valid_intervals", "shortfall_sum", "exceed_intervals"]
                )
        if missing:
            parts = [part for part in self._stats.values() if not part.empty]
            self.daily = self.window_daily(pd.concat(parts, ignore_index=True), self.cfg)
            self.lookup = self.daily.set_index(KEYS)["b3_score"]

    def interval_daily(self, scada: pd.DataFrame) -> pd.DataFrame:
        """Sufficient daily statistics; no labels or stop-run lookahead at score time."""
        work = scada[["StationId", "TimeStamp"]].copy()
        work["Day"] = interval_start(work.TimeStamp).dt.floor("D")
        residual = self.expected_rpm(scada[WIND].to_numpy()) - scada[RPM].to_numpy()
        valid = self._valid(scada).to_numpy() & np.isfinite(residual)
        work["valid_intervals"] = valid.astype(int)
        work["shortfall_sum"] = np.where(valid, residual, 0.0)
        work["exceed_intervals"] = (valid & (residual > self.tolerance)).astype(int)
        return work.groupby(KEYS, as_index=False)[
            ["valid_intervals", "shortfall_sum", "exceed_intervals"]
        ].sum()

    @staticmethod
    def window_daily(stats: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
        """Calendar-based trailing means, preserving missing evidence separately."""
        daily = (
            stats.groupby(KEYS)[["valid_intervals", "shortfall_sum", "exceed_intervals"]]
            .sum()
            .reset_index()
        )
        output = []
        for station, group in daily.groupby("StationId"):
            group = group.set_index("Day").sort_index()
            group = group.reindex(pd.date_range(group.index.min(), group.index.max(), name="Day"))
            count = group.valid_intervals.where(group.valid_intervals > 0)
            group["b3_daily_share"] = group.exceed_intervals / count
            group["b3_mean_shortfall"] = group.shortfall_sum / count
            roll = group.b3_daily_share.rolling(
                cfg["labels"]["feature_window_days"], min_periods=cfg["features"]["window_min_days"]
            )
            group["b3_window_share"] = roll.mean()
            group["b3_available"] = group.b3_window_share.notna()
            group["b3_score"] = group.b3_window_share.fillna(0.0)
            group["StationId"] = station
            output.append(group.reset_index())
        return pd.concat(output, ignore_index=True)

    def score(self, rows: pd.DataFrame) -> pd.Series:
        if not hasattr(self, "lookup"):
            raise RuntimeError("Fit B3 before scoring")
        if not rows.empty:
            self._ensure_years(rows.Day.dt.year.unique().tolist())
        keys = pd.MultiIndex.from_frame(rows[KEYS])
        return pd.Series(self.lookup.reindex(keys).fillna(0.0).to_numpy(), index=rows.index)


def check_b3_no_lookahead(
    model: PowerCurveProxy,
    scada: pd.DataFrame,
    cfg: dict[str, Any],
    cases: list[tuple[int, pd.Timestamp]],
) -> list[dict[str, Any]]:
    """Recompute B3 features with future intervals removed, holding train fit fixed."""
    problems = []
    columns = ["b3_daily_share", "b3_mean_shortfall", "b3_window_share", "b3_score"]
    for station in sorted({s for s, _ in cases}):
        source = scada.loc[scada.StationId == station]
        full = model.window_daily(model.interval_daily(source), cfg).set_index(KEYS)
        for st, day in cases:
            if st != station:
                continue
            cut = source.loc[source.TimeStamp <= day + pd.Timedelta(days=1)]
            past = model.window_daily(model.interval_daily(cut), cfg).set_index(KEYS)
            key = (station, day)
            if key not in full.index:
                continue
            if key not in past.index:
                bad = ["<row missing>"]
            else:
                bad = [
                    c
                    for c in columns
                    if not np.isclose(full.loc[key, c], past.loc[key, c], equal_nan=True)
                ]
            if bad:
                problems.append({"StationId": station, "Day": day, "columns": bad})
    return problems
