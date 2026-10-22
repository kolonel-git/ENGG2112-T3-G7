"""Fit on early years, choose thresholds on calibration, evaluate untouched later years."""

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
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning

from wt_predmaint.evaluation.metrics import METRICS, detect_episodes, evaluate, threshold_for_budget
from wt_predmaint.features.daily import KEYS, feature_columns
from wt_predmaint.models.single_year import fit_classifier, risk_scores
from wt_predmaint.models.baselines import sensor_rule_fit, sensor_rule_score, daily_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-missing", type=float, default=0.8)
    parser.add_argument("--targets", nargs="+", choices=["a", "b"], default=["a"])
    args = parser.parse_args()
    if not 0 <= args.max_missing < 1:
        parser.error("--max-missing must be in [0, 1)")
    prep = json.loads((args.input_dir / "preparation.json").read_text(encoding="utf-8"))
    if not set(args.targets).issubset(prep.get("targets", ["a", "b"])):
        parser.error("Requested targets were not prepared; rerun preparation for those targets")
    cfg, station = prep["protocol"], prep["station"]
    seed = int(cfg["seed"])
    features = pd.read_parquet(args.input_dir / "features.parquet")
    if set(features["StationId"].unique()) != {station} or features.duplicated(KEYS).any():
        parser.error("Expected unique feature keys for the selected turbine only")
    candidates = feature_columns(features)
    metrics, predictions, detections, audits, support, artifacts = [], [], [], [], [], []
    for target in args.targets:
        target_index = ("a", "b").index(target)
        labels = pd.read_parquet(args.input_dir / f"{target}_labels.parquet")
        episodes = pd.read_parquet(args.input_dir / f"{target}_episodes.parquet")
        if not labels["StationId"].eq(station).all() or not episodes["StationId"].eq(station).all():
            parser.error("Prepared labels/episodes contain another turbine")
        data = labels.merge(features, on=KEYS, how="left", validate="one_to_one", indicator=True)
        usable = data["label"].notna()
        for split in ("train", "val", "test"):
            mask = usable & data["split"].eq(split)
            support.append({"target": target.upper(), "split": split,
                            "labelled_rows": int(mask.sum()),
                            "missing_feature_rows": int((mask & data["_merge"].ne("both")).sum())})
        data = data.loc[usable & data["_merge"].eq("both")].sort_values(KEYS).copy()
        data["label"] = data["label"].astype(int)
        train, calibration, test = [data.loc[data["split"].eq(s)].copy()
                                     for s in ("train", "val", "test")]
        if any(frame.empty for frame in (train, calibration, test)):
            parser.error(f"Target {target.upper()}: empty training/calibration/test data; see prepared labels")
        if train["label"].nunique() != 2 or calibration["label"].nunique() != 2:
            parser.error(f"Target {target.upper()}: both classes required in train and calibration")
        print(f"Target {target.upper()}: train {len(train)}, calibration {len(calibration)}, "
              f"test {len(test)} labelled days", flush=True)
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            logistic, columns, audit = fit_classifier(train, candidates, seed, args.max_missing)
        forest = clone(logistic)
        forest.set_params(classifier=RandomForestClassifier(
            n_estimators=300, max_depth=6, min_samples_leaf=10,
            class_weight="balanced_subsample", random_state=seed, n_jobs=2))
        forest.fit(train[columns].replace([np.inf, -np.inf], np.nan), train["label"])
        audit["forest_importance"] = audit["feature"].map(
            dict(zip(columns, forest[-1].feature_importances_, strict=True)))
        rule = sensor_rule_fit(train, columns)
        audit["sensor_rule_used"] = audit["feature"].isin(rule["features"])
        audits.append(audit.assign(target=target.upper()))
        rng = np.random.default_rng([seed, station, target_index])
        scorers = {
            "logistic regression": (risk_scores(logistic, calibration, columns),
                                    risk_scores(logistic, test, columns), {"pipeline": logistic}),
            "random forest": (risk_scores(forest, calibration, columns),
                              risk_scores(forest, test, columns), {"pipeline": forest}),
            "high sensor rule": (sensor_rule_score(rule, calibration), sensor_rule_score(rule, test),
                                 {"sensor_rule": rule}),
            "constant (prevalence)": (np.zeros(len(calibration)), np.zeros(len(test)), {}),
            "random uniform": (rng.uniform(size=len(calibration)), rng.uniform(size=len(test)), {}),
        }
        for name, (cal_score, test_score, artifact) in scorers.items():
            threshold = threshold_for_budget(calibration["label"].to_numpy(), cal_score,
                float(cfg["evaluation"]["false_alarm_budget_per_turbine_month"]),
                float(cfg["evaluation"]["days_per_month"]))
            frame = test[[*KEYS, "label"]].assign(score=test_score)
            for period, subset in [("test_all", frame), *[(f"test_{year}", g)
                                  for year, g in frame.groupby(frame["Day"].dt.year)]]:
                result = evaluate(subset, episodes, threshold, cfg)
                result.update(daily_metrics(subset, threshold))
                result.update({f"{metric}_{bound}": np.nan
                               for metric in METRICS for bound in ("lo", "hi")})
                metrics.append({"target": target.upper(), "period": period, "model": name,
                                "station": station, "threshold": threshold,
                                "train_rows": len(train), "calibration_rows": len(calibration),
                                "n_features": (len(rule["features"]) if name == "high sensor rule"
                                               else len(columns) if artifact else 0),
                                "test_has_both_classes": subset["label"].nunique() == 2, **result})
            predictions.append(frame.assign(target=target.upper(), model=name, threshold=threshold,
                                             alert=test_score > threshold))
            detections.append(detect_episodes(frame, episodes, threshold,
                int(cfg["labels"]["positive_window_days"])).assign(target=target.upper(), model=name))
            if artifact:
                artifact.update(features=rule["features"] if name == "high sensor rule" else columns,
                                threshold=threshold, target=target.upper(),
                                station=station, protocol=cfg)
                artifacts.append((f"{target}_{name.replace(' ', '_')}.joblib", artifact))
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    for name, frames in (("predictions", predictions), ("episode_detections", detections),
                         ("features", audits)):
        pd.concat(frames, ignore_index=True).to_csv(output / f"{name}.csv", index=False)
    pd.DataFrame(metrics).to_csv(output / "metrics.csv", index=False)
    pd.DataFrame(support).to_csv(output / "coverage.csv", index=False)
    for name, artifact in artifacts:
        joblib.dump(artifact, output / name)
    metadata = {"station": station, "protocol": cfg, "years": prep["years"],
                "targets": args.targets, "max_missing": args.max_missing, "intervals": "omitted: only one turbine",
                "score_warning": "Scores are not calibrated probabilities"}
    (output / "run.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    selected = pd.DataFrame(metrics).query("period == 'test_all'")
    print(selected[["target", "model", "pr_auc", "recall", "median_lead_days",
                    "false_alarms_per_turbine_month"]].round(3).to_string(index=False))
    print(f"Wrote results and fitted models to {output}")


if __name__ == "__main__":
    main()
