"""Memory-bounded Target C validation run with reproducible uniform train sampling.

Fits on a uniform sample of eligible train prediction points and scores every eligible
validation point, in turbine/year batches. Coverage is provisional; see audit JSON.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import joblib
import numpy as np
import pandas as pd
from scripts.run_target_c import fit_and_evaluate
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from wt_predmaint.config import load_protocol

META = ["StationId", "TimeStamp", "label", "split", "horizon_observed", "horizon_end"]


def eligible(frame, split):
    return frame.loc[(frame.split == split) & frame.label.notna() & frame.horizon_observed]


def metrics(y, score, predicted):
    return {
        "rows": len(y),
        "positive_rate": float(np.mean(y)),
        "average_precision": float(average_precision_score(y, score)),
        "roc_auc": float(roc_auc_score(y, score)) if len(np.unique(y)) == 2 else None,
        "accuracy": float(accuracy_score(y, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
        "precision": float(precision_score(y, predicted, zero_division=0)),
        "recall": float(recall_score(y, predicted, zero_division=0)),
        "f1": float(f1_score(y, predicted, zero_division=0)),
        "predicted_yes_rate": float(np.mean(predicted)),
        "confusion_matrix_actual_no_yes_predicted_no_yes": confusion_matrix(
            y, predicted, labels=[0, 1]
        ).tolist(),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=ROOT / "data/interim/target_c")
    p.add_argument("--out", type=Path, default=ROOT / "reports/target_c")
    p.add_argument("--max-train-rows", type=int, default=100000)
    args = p.parse_args()
    if args.max_train_rows < 2:
        p.error("--max-train-rows must be at least 2")
    args.out.mkdir(parents=True, exist_ok=True)
    paths = sorted(args.input.glob("scada_*.parquet"))
    counts, inventory = [], []
    for path in paths:
        meta = pd.read_parquet(path, columns=META)
        n = len(eligible(meta, "train"))
        counts.append(n)
        inventory.append(
            {
                "partition": path.name,
                "rows": len(meta),
                "train_eligible": n,
                "val_eligible": len(eligible(meta, "val")),
                "unknown_labels": int(meta.label.isna().sum()),
            }
        )
    total = sum(counts)
    if not total:
        p.error("no eligible train data")
    rng = np.random.default_rng(load_protocol()["seed"])
    chosen = np.sort(rng.choice(total, min(total, args.max_train_rows), replace=False))
    pieces, offset = [], 0
    for path, count in zip(paths, counts, strict=True):
        selected = chosen[(chosen >= offset) & (chosen < offset + count)] - offset
        if len(selected):
            frame = eligible(pd.read_parquet(path), "train")
            pieces.append(frame.iloc[selected])
        offset += count
    train = pd.concat(pieces, ignore_index=True)
    del pieces
    print(f"Uniform train sample: {len(train):,} of {total:,} eligible points", flush=True)
    val_paths = [path for path, row in zip(paths, inventory, strict=True) if row["val_eligible"]]
    # A source-year file may contain only the next year's 00:00 boundary point.
    # Find a genuine two-class validation batch for the shared fitting helper.
    first_val = None
    for path in val_paths:
        candidate = eligible(pd.read_parquet(path), "val")
        if candidate.label.nunique() == 2:
            first_val = candidate
            break
    if first_val is None:
        p.error("validation requires both yes and no labels")
    start = time.perf_counter()
    model, columns, _, _ = fit_and_evaluate(train, first_val)
    print(f"Fit {len(columns)} input features in {time.perf_counter() - start:.1f}s", flush=True)
    joblib.dump({"model": model, "features": columns}, args.out / "model.joblib")
    train_rate = float(train.label.mean())
    del train, first_val
    frames = []
    for path in val_paths:
        frame = eligible(pd.read_parquet(path), "val")
        out = frame[["StationId", "TimeStamp", "label"]].copy()
        out["alarm_score"] = model.predict_proba(frame[columns])[:, 1]
        out["predicted_alarm"] = np.where(out.alarm_score >= 0.5, "yes", "no")
        out["recent_alarm_count_1h"] = frame["1h_alarm_count"].to_numpy()
        frames.append(out)
        print(f"Scored {path.name}: {len(out):,}", flush=True)
    predictions = pd.concat(frames, ignore_index=True)
    y = predictions.label.astype(int).to_numpy()
    score = predictions.alarm_score.to_numpy()
    baseline = predictions.recent_alarm_count_1h.fillna(0).to_numpy()
    results = {
        "status": "PROVISIONAL: supplied monthly alarm exports assumed complete",
        "train_eligible_rows": total,
        "train_sample_rows": len(chosen),
        "train_sample_positive_rate": train_rate,
        "feature_count": len(columns),
        "seed": load_protocol()["seed"],
        "threshold": 0.5,
        "test_years_scored": False,
        "prediction_minutes": int(
            predictions.groupby("StationId").TimeStamp.diff().dropna().dt.total_seconds().median()
            / 60
        ),
        "forecast_horizon_hours": 6,
        "train_years": load_protocol()["splits"]["train_years"],
        "validation_years": load_protocol()["splits"]["val_years"],
        "logistic": metrics(y, score, score >= 0.5),
        "recent_alarm_1h_baseline": metrics(y, baseline, baseline > 0),
        "always_yes": metrics(y, np.ones(len(y)), np.ones(len(y), dtype=bool)),
        "always_no": metrics(y, np.zeros(len(y)), np.zeros(len(y), dtype=bool)),
        "notes": [
            "Overlapping six-hour windows; point-level metrics are dependent.",
            "Class-weighted model scores are not calibrated probabilities.",
            "Training sampled for memory; all eligible validation points scored.",
            "Missing recent-alarm history is zero only for the simple baseline.",
        ],
    }
    per_year = []
    predictions["year"] = predictions.TimeStamp.dt.year
    for year, g in predictions.groupby("year"):
        per_year.append(
            {
                "year": int(year),
                **metrics(
                    g.label.astype(int).to_numpy(),
                    g.alarm_score.to_numpy(),
                    g.alarm_score.to_numpy() >= 0.5,
                ),
            }
        )
    results["logistic_by_year"] = per_year
    per_station = []
    for station, g in predictions.groupby("StationId"):
        per_station.append(
            {
                "StationId": int(station),
                **metrics(
                    g.label.astype(int).to_numpy(),
                    g.alarm_score.to_numpy(),
                    g.alarm_score.to_numpy() >= 0.5,
                ),
            }
        )
    pd.DataFrame(per_station).to_csv(args.out / "validation_by_station.csv", index=False)
    pd.DataFrame(inventory).to_csv(args.out / "partition_inventory.csv", index=False)
    predictions.to_parquet(args.out / "validation_predictions.parquet", index=False)
    # Keep notebook compatibility and a compact human-readable preview.
    predictions.head(1000).to_csv(args.out / "validation_predictions.csv", index=False)
    results["csv_preview_rows"] = min(1000, len(predictions))
    (args.out / "validation_metrics.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
