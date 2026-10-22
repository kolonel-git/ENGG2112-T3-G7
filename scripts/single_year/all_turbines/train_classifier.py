"""Train/evaluate within-year logistic regression and same-fold dummy references."""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

from wt_predmaint.config import load_protocol
from wt_predmaint.evaluation.metrics import (
    METRICS, detect_episodes, evaluate, evaluation_row, threshold_for_budget,
)
from wt_predmaint.features.daily import KEYS, feature_columns
from wt_predmaint.models.single_year import fit_classifier, risk_scores, training_folds

YEAR = 2018


def main(default_station: int | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=YEAR)
    parser.add_argument("--station", type=int, default=default_station,
                        help="Train, calibrate and test using only this turbine")
    parser.add_argument("--interim", type=Path, default=ROOT / "data/interim")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resamples", type=int, default=1000)
    parser.add_argument("--max-missing", type=float, default=0.8)
    args = parser.parse_args()
    if not 1900 <= args.year <= 9998 or args.resamples < 1 or not 0 <= args.max_missing < 1:
        parser.error("Use a valid year, positive resamples, and 0 <= max-missing < 1")
    cfg = load_protocol()
    seed = int(cfg["seed"])
    horizon = int(cfg["labels"]["exclusion_end_days"])
    features = pd.read_parquet(args.interim / "features/daily_features.parquet")
    candidates = feature_columns(features)
    features = features.loc[features["Day"].dt.year == args.year]
    if args.station is not None:
        features = features.loc[features["StationId"] == args.station]
        if features.empty:
            parser.error(f"No feature rows for turbine {args.station} in {args.year}")
    if features.duplicated(KEYS).any():
        parser.error("Feature table contains duplicate turbine-day keys")
    scope = f"turbine_{args.station}" if args.station is not None else "all_turbines"
    output = args.output_dir or ROOT / f"data/processed/single_year/{args.year}/{scope}"
    metrics, predictions, audits, detections, skipped, bounds, model_artifacts = [], [], [], [], [], [], []
    for target_index, target in enumerate(("a", "b")):
        labels = pd.read_parquet(args.interim / f"labels/{target}_labels.parquet")
        episodes = pd.read_parquet(args.interim / f"labels/{target}_episodes.parquet")
        if args.station is not None:
            labels = labels.loc[labels["StationId"] == args.station]
            episodes = episodes.loc[episodes["StationId"] == args.station]
        if labels.duplicated(KEYS).any():
            parser.error(f"Target {target}: duplicate label keys")
        data = labels.loc[labels["Day"].dt.year == args.year].merge(
            features, on=KEYS, how="left", validate="one_to_one", indicator=True
        )
        missing = (data["label"].notna() & data["_merge"].ne("both")).sum()
        if missing:
            parser.error(f"Target {target}: {missing} labelled days have no feature row; rebuild features")
        data = data.drop(columns="_merge")
        for fold, start, end, train, calibration, test in training_folds(data, args.year, horizon):
            reason = None
            if train.empty or calibration.empty or test.empty:
                reason = "empty train/calibration/test period"
            elif train["label"].nunique() < 2 or calibration["label"].nunique() < 2:
                reason = "train or calibration lacks both classes"
            if reason:
                skipped.append({"target": target.upper(), "fold": fold, "reason": reason,
                                "train_rows": len(train), "calibration_rows": len(calibration),
                                "test_rows": len(test)})
                print(f"Target {target.upper()} fold {fold}: skipped ({reason})", flush=True)
                continue
            print(f"Target {target.upper()} fold {fold}: train {len(train)}, "
                  f"calibration {len(calibration)}, test {len(test)}", flush=True)
            if test["label"].nunique() < 2:
                print("  Test period contains one class only: ranking and false-alarm "
                      "results cannot establish discrimination between classes.", flush=True)
            # Do not silently publish an unconverged model.
            with warnings.catch_warnings():
                warnings.simplefilter("error", ConvergenceWarning)
                model, columns, audit = fit_classifier(train, candidates, seed, args.max_missing)
            audits.append(audit.assign(target=target.upper(), fold=fold))
            model_artifacts.append((f"{target}_fold{fold}.joblib", {
                "pipeline": model, "features": columns, "year": args.year,
                "target": target.upper(), "fold": fold, "station": args.station,
            }))
            bounds.append({"target": target.upper(), "fold": fold,
                           **{f"{name}_{edge}": str(frame["Day"].agg(edge).date())
                              for name, frame in (("train", train), ("calibration", calibration),
                                                  ("test", test)) for edge in ("min", "max")}})
            eps = episodes.loc[(episodes["Start"] >= start) & (
                episodes["Start"] < min(end + pd.Timedelta(days=horizon),
                                       pd.Timestamp(year=args.year + 1, month=1, day=1)))]
            rng = np.random.default_rng([seed, args.year, target_index, fold])
            for name in ("logistic regression", "constant (prevalence)", "random uniform"):
                val, frame = calibration.copy(), test.copy()
                if name == "logistic regression":
                    val["score"] = risk_scores(model, val, columns)
                    frame["score"] = risk_scores(model, frame, columns)
                elif name.startswith("constant"):
                    val["score"], frame["score"] = 0.0, 0.0
                else:
                    val["score"] = rng.uniform(size=len(val))
                    frame["score"] = rng.uniform(size=len(frame))
                threshold = threshold_for_budget(
                    val["label"].to_numpy(), val["score"].to_numpy(),
                    float(cfg["evaluation"]["false_alarm_budget_per_turbine_month"]),
                    float(cfg["evaluation"]["days_per_month"]),
                )
                if name == "logistic regression":
                    model_artifacts[-1][1]["threshold"] = threshold
                if frame["StationId"].nunique() == 1:
                    result = {"model": name, "threshold": threshold,
                              **evaluate(frame, eps, threshold, cfg)}
                    result.update({f"{metric}_{edge}": np.nan
                                   for metric in METRICS for edge in ("lo", "hi")})
                else:
                    result = evaluation_row(name, frame, eps, threshold, cfg, args.resamples)
                metrics.append({"year": args.year, "target": target.upper(), "fold": fold,
                                "station": args.station,
                                "test_has_both_classes": test["label"].nunique() == 2,
                                "train_rows": len(train), "calibration_rows": len(val),
                                "n_features": len(columns) if name == "logistic regression" else 0,
                                **result})
                predictions.append(frame[[*KEYS, "label", "score"]].assign(
                    target=target.upper(), fold=fold, model=name, threshold=threshold,
                    alert=frame["score"].to_numpy() > threshold))
                det = detect_episodes(frame, eps, threshold, int(cfg["labels"]["positive_window_days"]))
                detections.append(det.assign(target=target.upper(), fold=fold, model=name))
    if not metrics:
        parser.error("No trainable folds. Check year coverage, labels, and printed skip reasons")
    output.mkdir(parents=True, exist_ok=True)
    for name, frames in (("predictions", predictions), ("features", audits),
                         ("episode_detections", detections)):
        pd.concat(frames, ignore_index=True).to_csv(output / f"{name}.csv", index=False)
    pd.DataFrame(metrics).to_csv(output / "metrics.csv", index=False)
    for name, artifact in model_artifacts:
        joblib.dump(artifact, output / name)
    manifest = {"year": args.year, "station": args.station, "scope": scope,
                "seed": seed, "protocol": cfg, "folds": bounds,
                "interval_method": "none_single_turbine" if args.station is not None
                else "turbine_bootstrap_when_multiple_test_turbines",
                "skipped_folds": skipped, "max_training_missing_share": args.max_missing,
                "resamples": args.resamples, "score_warning": "Not calibrated probabilities"}
    (output / "run.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote models, scores, feature audit and metrics to {output}")


if __name__ == "__main__":
    main()
