"""Normal-behaviour models for O1 (D-054): ridge on healthy train days, residual as the indicator.

One ridge regression per health-bearing temperature, predicting its daily producing-interval mean
from operating conditions (wind, rpm, ambient and nacelle temperature). Turbines share the slopes
and keep their own offset (a within-turbine, fixed-effects fit), so sensor offsets between turbines
do not leak into the residual. The gap is ``predicted - observed`` as in the project brief, divided
by the turbine's healthy residual spread from the training years, so a positive value means the
sensor runs cooler than expected and a negative value means hotter.

This is a feature generator, not a ``Baseline``: it reads the healthy mask (which uses labels to
choose training rows) in ``fit`` only. ``residuals`` scores a row from that row alone, and
``smooth_residuals`` looks backwards only, so nothing at day D uses data after D.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

KEYS = ["StationId", "Day"]
PREFIX, SUFFIX = "op_wtc_", "_mean"


def short_name(column: str) -> str:
    """``op_wtc_GeOilTmp_mean`` -> ``GeOilTmp``."""
    return column.removeprefix(PREFIX).removesuffix(SUFFIX)


def _finite(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    values = frame[columns].apply(pd.to_numeric, errors="coerce").astype("float64")
    return values.replace([np.inf, -np.inf], np.nan)


class NormalBehaviourModel:
    """Ridge normal-behaviour models with per-turbine offsets and residual spreads."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        s = cfg["nbm"]
        self.inputs: list[str] = list(s["inputs"])
        self.targets: list[str] = list(s["targets"])
        self.alpha = float(s["ridge_alpha"])
        self.min_rows = int(s["minimum_fit_rows"])
        self.min_turbine_rows = int(s["minimum_turbine_rows"])
        self.min_std = float(s["minimum_std"])
        self.train_years = {int(y) for y in cfg["splits"]["train_years"]}
        if not self.inputs or not self.targets:
            raise ValueError("nbm.inputs and nbm.targets must not be empty")
        if set(self.inputs) & set(self.targets):
            raise ValueError("an NBM target cannot also be one of its inputs")
        self.fitted: dict[str, dict[str, Any]] = {}

    def fit(self, frame: pd.DataFrame, healthy: pd.Series) -> NormalBehaviourModel:
        """Fit on healthy rows of train years; ``healthy`` is a mask aligned to ``frame``."""
        missing = set(KEYS + self.inputs + self.targets) - set(frame)
        if missing:
            raise ValueError(f"missing feature columns: {sorted(missing)}")
        if not healthy.index.equals(frame.index):
            raise ValueError("healthy mask must be indexed like the frame")
        chosen = frame.loc[healthy.fillna(False).astype(bool)]
        if not chosen["Day"].dt.year.isin(self.train_years).all():
            raise ValueError("NBM fit received rows outside the train years")
        x_all = _finite(chosen, self.inputs)
        self.fitted = {}
        for target in self.targets:
            y_all = _finite(chosen, [target])[target]
            ok = x_all.notna().all(axis=1) & y_all.notna()
            if int(ok.sum()) < self.min_rows:
                continue  # target stays unmodelled; residuals for it are NaN
            x, y = x_all.loc[ok], y_all.loc[ok]
            station = chosen.loc[ok, "StationId"]
            x_mean = x.groupby(station).transform("mean")
            y_mean = y.groupby(station).transform("mean")
            xc, yc = x - x_mean, y - y_mean
            scale = xc.std(ddof=0).replace(0.0, np.nan)
            usable = scale.index[scale.notna()].tolist()
            if not usable:
                continue
            ridge = Ridge(alpha=self.alpha, fit_intercept=False)
            ridge.fit((xc[usable] / scale[usable]).to_numpy(), yc.to_numpy())
            coef = pd.Series(ridge.coef_ / scale[usable].to_numpy(), index=usable)
            offsets = y.groupby(station).mean() - x[usable].groupby(station).mean().mul(coef).sum(
                axis=1
            )
            fleet_offset = float(y.mean() - (x[usable].mean() * coef).sum())
            resid = y - (x[usable] @ coef + offsets.reindex(station).to_numpy())
            fleet_std = float(resid.std(ddof=0))
            spread = resid.groupby(station).agg(["std", "count"])
            self.fitted[target] = {
                "coef": coef,
                "offsets": offsets,
                "fleet_offset": fleet_offset,
                "fleet_std": fleet_std,
                "std": {
                    st: (float(r["std"]) if r["count"] >= self.min_turbine_rows else fleet_std)
                    for st, r in spread.iterrows()
                },
                "n_fit": int(ok.sum()),
                "r2_train": float(1.0 - resid.var(ddof=0) / y.var(ddof=0)),
            }
        if not self.fitted:
            raise ValueError("no NBM target had enough healthy rows to fit")
        return self

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Predicted daily mean per modelled target; NaN where an input is missing."""
        if not self.fitted:
            raise RuntimeError("fit the model first")
        x = _finite(frame, self.inputs)
        out = pd.DataFrame(index=frame.index)
        for target, m in self.fitted.items():
            offset = frame["StationId"].map(m["offsets"]).fillna(m["fleet_offset"]).astype(float)
            pred = (x[m["coef"].index] * m["coef"]).sum(axis=1, skipna=False) + offset
            out[target] = pred.where(x[self.inputs].notna().all(axis=1))
        return out

    def residuals(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Standardised gap (predicted - observed) per turbine-day: ``nbm_res_<sensor>`` columns."""
        pred = self.predict(frame)
        observed = _finite(frame, list(pred.columns))
        out = pd.DataFrame(index=frame.index)
        for target, m in self.fitted.items():
            std = frame["StationId"].map(m["std"]).fillna(m["fleet_std"]).astype(float)
            std = std.where(std > self.min_std)
            out[f"nbm_res_{short_name(target)}"] = (pred[target] - observed[target]) / std
        return out.astype("float32")

    def parameters(self) -> pd.DataFrame:
        """One row per fitted target: rows used, train R^2, fleet residual spread, coefficients."""
        rows = []
        for target, m in self.fitted.items():
            row = {
                "sensor": short_name(target),
                "n_fit": m["n_fit"],
                "r2_train": m["r2_train"],
                "resid_std_fleet": m["fleet_std"],
            }
            row.update({f"coef_{short_name(c)}": float(v) for c, v in m["coef"].items()})
            rows.append(row)
        return pd.DataFrame(rows)


def smooth_residuals(
    residuals: pd.DataFrame, keys: pd.DataFrame, window: int, min_days: int
) -> pd.DataFrame:
    """Trailing calendar-day mean of each residual column, per turbine; backward-looking only.

    ``keys`` carries StationId and Day aligned to ``residuals``. Days with no row count as missing,
    and a mean needs at least ``min_days`` observed days in the window. Columns are renamed
    ``nbm_w<window>_<sensor>``; the result keeps ``residuals``' index.
    """
    if window < 1 or not 1 <= min_days <= window:
        raise ValueError("need window >= 1 and 1 <= min_days <= window")
    work = pd.concat([keys[KEYS], residuals.astype("float64")], axis=1)
    if work.duplicated(KEYS).any():
        raise ValueError("StationId and Day must be unique")
    pieces = []
    for _, g in work.groupby("StationId", sort=False):
        calendar = g.drop(columns="StationId").set_index("Day").sort_index()
        calendar = calendar.reindex(pd.date_range(calendar.index.min(), calendar.index.max()))
        rolled = calendar.rolling(window, min_periods=min_days).mean()
        pieces.append(rolled.reindex(g["Day"]).set_axis(g.index))
    out = pd.concat(pieces).reindex(residuals.index)
    out.columns = [c.replace("nbm_res_", f"nbm_w{window}_") for c in residuals.columns]
    return out.astype("float32")


def indicator(smoothed: pd.DataFrame) -> pd.Series:
    """Daily degradation indicator: largest absolute smoothed standardised residual over sensors."""
    return smoothed.abs().max(axis=1).astype("float32").rename("nbm_indicator")
