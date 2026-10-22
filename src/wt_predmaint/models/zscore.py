"""R1: fixed H0 healthy-training moments and maximum absolute sensor z-score."""

import numpy as np
import pandas as pd

from wt_predmaint.models.base import require_train_only


class ZScoreReference:
    name = "R1_zscore"
    requires_healthy_rows = True

    def __init__(self, target):
        if target not in {"a", "b"}:
            raise ValueError("target must be a or b")
        self.target = target

    def fit(self, train, cfg):
        require_train_only(train)
        s = cfg["r1_zscore"]
        if "healthy_fit" not in train:
            raise ValueError("R1 needs the runner's complete-history H0 training mask")
        healthy = train.loc[train.healthy_fit.eq(True)]
        self.sensors = s["sensors"]
        self.window = s["window_days"]
        self.fallback = float(s["missing_score"])
        self.parameters = []
        self.fleet = {}
        self.local = {}
        minimum, floor = s["minimum_healthy_rows"], s["minimum_std"]
        for sensor in self.sensors:
            col = "op_" + sensor
            values = pd.to_numeric(healthy[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
            mean, std = values.mean(), values.std(ddof=0)
            valid = values.count() >= minimum and np.isfinite(std) and std > floor
            self.fleet[sensor] = (mean, std) if valid else (np.nan, np.nan)
            self.local[sensor] = {}
            for station in train.StationId.unique():
                v = values.loc[healthy.StationId.eq(station)]
                mu, sd = v.mean(), v.std(ddof=0)
                local = v.count() >= minimum and np.isfinite(sd) and sd > floor
                pair = (mu, sd) if local else self.fleet[sensor]
                self.local[sensor][station] = pair
                self.parameters.append(
                    dict(
                        target=self.target,
                        StationId=station,
                        sensor=sensor,
                        healthy_count=int(v.count()),
                        mean=pair[0],
                        std=pair[1],
                        source="turbine" if local else "fleet" if valid else "disabled",
                    )
                )
        if not any(np.isfinite(v[1]) for v in self.fleet.values()):
            raise ValueError("no usable healthy sensor reference")
        return self

    def sensor_scores(self, rows):
        out = pd.DataFrame(index=rows.index)
        for sensor in self.sensors:
            pairs = [
                self.local[sensor].get(station, self.fleet[sensor]) for station in rows.StationId
            ]
            means = np.array([p[0] for p in pairs])
            stds = np.array([p[1] for p in pairs])
            col = f"w{self.window}mean_op_{sensor}"
            v = pd.to_numeric(rows.reindex(columns=[col])[col], errors="coerce").to_numpy(
                dtype=float
            )
            out[sensor] = (v - means) / stds
        return out.replace([np.inf, -np.inf], np.nan)

    def score(self, rows):
        return self.sensor_scores(rows).abs().max(axis=1).fillna(self.fallback).astype("float64")
