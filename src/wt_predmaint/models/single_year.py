"""Purged within-year training, with preprocessing fitted on training rows only."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def training_folds(data: pd.DataFrame, year: int, horizon: int):
    """August–October test months; 90-day calibration block and full-horizon gaps.

    Calibration occupies the start of the 90 days before testing; its final
    ``horizon`` days are purged. Train labels must mature before calibration.
    Test labels must mature before the end of the selected calendar year.
    """
    year_start = pd.Timestamp(year=year, month=1, day=1)
    year_end = pd.Timestamp(year=year + 1, month=1, day=1)
    data = data.loc[(data["Day"] >= year_start) & (data["Day"] < year_end)
                    & data["label"].notna()].sort_values(["StationId", "Day"]).copy()
    data["label"] = data["label"].astype(int)
    mature = data["Day"] + pd.Timedelta(days=horizon)
    for fold, month in enumerate((8, 9, 10), 1):
        start = pd.Timestamp(year=year, month=month, day=1)
        end = start + pd.DateOffset(months=1)
        calibration_start = start - pd.Timedelta(days=90)
        train = data.loc[mature < calibration_start].copy()
        calibration = data.loc[(data["Day"] >= calibration_start) & (mature < start)].copy()
        test = data.loc[(data["Day"] >= start) & (data["Day"] < end)
                        & (mature < year_end)].copy()
        yield fold, start, end, train, calibration, test


def fit_classifier(train: pd.DataFrame, candidates: list[str], seed: int,
                   max_missing: float = 0.8):
    """Drop sparse/constant columns using train only, then impute, scale and fit."""
    x = train[candidates].replace([np.inf, -np.inf], np.nan).astype(float)
    kept = [c for c in candidates if x[c].isna().mean() <= max_missing
            and x[c].nunique(dropna=True) > 1]
    if not kept:
        raise ValueError("No variable features with sufficient training coverage")
    if train["label"].nunique() != 2:
        raise ValueError("Training requires both positive and negative labels")
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(class_weight="balanced", C=1.0,
                                          max_iter=3000, random_state=seed)),
    ])
    pipeline.fit(x[kept], train["label"])
    audit = pd.DataFrame({
        "feature": candidates,
        "training_missing_share": [float(x[c].isna().mean()) for c in candidates],
        "training_unique_values": [int(x[c].nunique()) for c in candidates],
        "used": [c in kept for c in candidates],
        "coefficient": [float(pipeline[-1].coef_[0, kept.index(c)])
                        if c in kept else np.nan for c in candidates],
    })
    return pipeline, kept, audit


def risk_scores(pipeline: Pipeline, data: pd.DataFrame, columns: list[str]) -> np.ndarray:
    """Class-1 scores; balanced fitting does not guarantee probability calibration."""
    x = data[columns].replace([np.inf, -np.inf], np.nan).astype(float)
    return pipeline.predict_proba(x)[:, 1]
