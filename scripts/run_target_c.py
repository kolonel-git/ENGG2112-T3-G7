"""Fit Target C on train years; report validation yes/no predictions, leaving test untouched."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from wt_predmaint.config import load_protocol

METADATA = {"StationId", "TimeStamp", "horizon_end", "label", "horizon_observed", "split"}


def fit_and_evaluate(train: pd.DataFrame, val: pd.DataFrame) -> tuple:
    """Fit preprocessing only on train; use a fixed 0.5 decision threshold."""
    if train.empty or val.empty or train.label.nunique() != 2 or val.label.nunique() != 2:
        raise ValueError("train and validation each require both yes and no labels")
    columns = [c for c in train if c not in METADATA and train[c].notna().any()]
    if not columns:
        raise ValueError("no observed training features")
    model = make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True),
        StandardScaler(),
        LogisticRegression(
            class_weight="balanced", max_iter=2000, random_state=load_protocol()["seed"]
        ),
    )
    model.fit(train[columns], train.label.astype(int))
    probabilities = model.predict_proba(val[columns])[:, 1]
    predicted = probabilities >= 0.5
    truth = val.label.astype(int)
    metrics = {
        "train_rows": len(train),
        "validation_rows": len(val),
        "train_positive_rate": float(train.label.mean()),
        "validation_positive_rate": float(truth.mean()),
        "pr_auc_average_precision": float(average_precision_score(truth, probabilities)),
        "constant_score_average_precision": float(truth.mean()),
        "precision": float(precision_score(truth, predicted, zero_division=0)),
        "recall": float(recall_score(truth, predicted, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, predicted)),
        "confusion_matrix_rows_actual_no_yes_columns_predicted_no_yes": confusion_matrix(
            truth, predicted, labels=[0, 1]
        ).tolist(),
        "threshold": 0.5,
        "note": "Class-weighted scores are not calibrated alarm probabilities.",
    }
    predictions = val[["StationId", "TimeStamp", "label"]].copy()
    predictions["alarm_score"] = probabilities
    predictions["predicted_alarm"] = ["yes" if flag else "no" for flag in predicted]
    return model, columns, metrics, predictions


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=ROOT / "data/interim/target_c")
    p.add_argument("--out", type=Path, default=ROOT / "reports/target_c")
    args = p.parse_args()
    frames = []
    for path in sorted(args.input.glob("scada_*.parquet")):
        # Filter at read time: test data is never used for fitting or evaluation here.
        frame = pd.read_parquet(path, filters=[("split", "in", ["train", "val"])])
        frames.append(frame.loc[frame.label.notna() & frame.horizon_observed])
    if not frames:
        p.error("no Target C tables; run scripts/build_target_c.py first")
    table = pd.concat(frames, ignore_index=True)
    model, columns, metrics, predictions = fit_and_evaluate(
        table.loc[table.split == "train"], table.loc[table.split == "val"]
    )
    args.out.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": columns}, args.out / "model.joblib")
    predictions.to_csv(args.out / "validation_predictions.csv", index=False)
    (args.out / "validation_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
