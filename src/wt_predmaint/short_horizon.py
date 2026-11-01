"""Binary forecasts of ANY new alarm start from the preceding 120 minutes.

Independent of the repository's daily degradation protocol. Reads the team's
already aligned Data_Parquets_10min files; never changes or regenerates them.
"""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

VERSION = 1
STATS = ("mean", "std", "min", "max", "last", "change")


def load_config(path: str | Path) -> dict[str, Any]:
    """Load explicit experiment settings, independent of machine-specific paths."""
    path = Path(path).resolve()
    cfg = json.loads(path.read_text(encoding="utf-8"))
    # Support the team's original nested layout as well as this local flattening.
    repo = path.parent.parent
    if not (repo / cfg["data_root"]).is_dir():
        canonical = repo / "iteration2" / "Iteration2" / "Data_Parquets_10min"
        if canonical.is_dir():
            cfg["data_root"] = canonical.relative_to(repo).as_posix()
            cfg["report_root"] = (
                (canonical.parent / "reports" / "any_alarm_binary").relative_to(repo).as_posix()
            )
    if cfg["target"] != "any_alarm_start":
        raise ValueError("This experiment predicts any alarm start, not selected alarms.")
    if cfg["lookback_minutes"] != 120 or cfg["step_minutes"] != 10:
        raise ValueError("This workflow requires a 120-minute history and 10-minute grid.")
    if cfg["horizons_minutes"] != [10, 20, 30]:
        raise ValueError("Expected forecast horizons [10, 20, 30].")
    for name in ("max_train_rows", "max_validation_rows", "max_test_rows"):
        if cfg[name] is not None and cfg[name] < 2:
            raise ValueError(f"{name} must be None or at least 2.")
    dates = [
        pd.Timestamp(cfg[n]) for n in ("train_start", "train_end", "validation_end", "test_end")
    ]
    if dates != sorted(set(dates)):
        raise ValueError("Split dates must be strictly increasing.")
    return cfg


def discover(data_root: Path, cfg: dict[str, Any], begin: str, end: str) -> list[Path]:
    """Discover only requested years and the 21 physical turbine IDs."""
    paths = []
    for year in range(pd.Timestamp(begin).year, pd.Timestamp(end).year + 1):
        if pd.Timestamp(year, 1, 1) >= pd.Timestamp(end):
            break
        for turbine in cfg["turbine_ids"]:
            path = data_root / str(year) / f"Station_{turbine}" / "data.parquet"
            if path.is_file():
                paths.append(path)
    if not paths:
        raise FileNotFoundError(f"No requested data found in {data_root} for {begin} to {end}.")
    return paths


def sensor_columns(paths: list[Path]) -> list[str]:
    """Feature vocabulary comes from training files only, never future labels."""
    names = sorted({n for p in paths for n in pq.read_schema(p).names if n.startswith("wtc_")})
    if not names:
        raise ValueError("No wtc_ sensor columns found.")
    return names


def read_frame(path: Path, sensors: list[str]) -> pd.DataFrame:
    """Read only sensor values and two audit/label fields; no event JSON in RAM."""
    names = set(pq.read_schema(path).names)
    required = {"sensorObserved", "alarmOccurred"}
    if not required <= names:
        raise ValueError(f"Missing expected preprocessing fields: {path}")
    cols = [n for n in sensors if n in names] + sorted(required)
    frame = pd.read_parquet(path, columns=cols)
    validate_grid(frame)
    return frame


def validate_grid(frame: pd.DataFrame) -> None:
    """Do not silently sort, fill or aggregate a damaged/misaligned source grid."""
    index = frame.index
    if (
        not isinstance(index, pd.DatetimeIndex)
        or index.tz is not None
        or index.has_duplicates
        or not index.is_monotonic_increasing
    ):
        raise ValueError("Expected a sorted, unique, timezone-naive timestamp grid.")
    if len(index) < 2 or (index != index.floor("10min")).any():
        raise ValueError("Expected timestamps on a 10-minute grid.")
    if not (index[1:] - index[:-1] == pd.Timedelta(minutes=10)).all():
        raise ValueError("Gap in the regular grid: rerun the source preprocessing.")
    for name in ("sensorObserved", "alarmOccurred"):
        if frame[name].isna().any() or not frame[name].isin([True, False, 0, 1]).all():
            raise ValueError(f"{name} must be a nonmissing boolean field.")


def feature_windows(frame: pd.DataFrame, sensors: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    """At t, use exactly 12 rows [t-120min,t); never current/future readings.

    At t=12:00 the last usable row is 11:50. This conservative convention avoids
    using a partially available current interval. Missing sensors stay NaN;
    train-fitted imputers handle them. All 12 intervals need source observations.
    """
    observed = frame.sensorObserved.astype(bool)
    values = frame.reindex(columns=sensors).astype("float32")
    values = values.replace([np.inf, -np.inf], np.nan).where(observed, axis=0)
    history = values.shift(1)
    rolling = history.rolling(12, min_periods=1)
    parts = []
    for stat in STATS:
        if stat == "last":
            part = history.copy()
        elif stat == "change":
            part = history - values.shift(12)
        else:
            part = getattr(rolling, stat)()
        part.columns = [f"{name}__{stat}" for name in sensors]
        parts.append(part)
    features = pd.concat(parts, axis=1).astype("float32")
    complete = observed.shift(1, fill_value=False).rolling(12, min_periods=12).sum().eq(12)
    return features, complete


def binary_targets(frame: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    """1 iff ANY alarm starts in [t,t+h). Includes starts during active alarms.

    alarmOccurred describes starts in each half-open bin; alarmOn and the
    selectedAlarmOccurred field are not the requested target. Unknown year-end
    outcomes remain NaN and cannot be labelled 'no'.
    """
    starts = frame.alarmOccurred.astype("int8")
    targets = {}
    for horizon in horizons:
        bins = horizon // 10
        forward = starts.iloc[::-1].rolling(bins, min_periods=bins).sum().iloc[::-1]
        targets[f"target_{horizon}min"] = forward.gt(0).astype("float32").where(forward.notna())
    return pd.DataFrame(targets, index=frame.index)


def split_mask(
    index: pd.DatetimeIndex, complete: pd.Series, begin: str, end: str, max_horizon: int = 30
) -> np.ndarray:
    """Keep both the entire history and all future targets inside a split."""
    return (
        complete.to_numpy()
        & (index - pd.Timedelta(minutes=120) >= pd.Timestamp(begin))
        & (index + pd.Timedelta(minutes=max_horizon) <= pd.Timestamp(end))
    )


class UniformSample:
    """Bounded-memory uniform sample without replacement via random priorities.

    Every eligible row has equal inclusion probability. Sampling never uses
    labels, never balances validation/test, and is disclosed with population sizes.
    """

    def __init__(self, limit: int | None, seed: int) -> None:
        self.limit = limit
        self.rng = np.random.default_rng(seed)
        self.rows: pd.DataFrame | None = None
        self.seen = 0

    def add(self, rows: pd.DataFrame) -> None:
        self.seen += len(rows)
        if rows.empty:
            return
        rows = rows.copy()
        rows["_priority"] = self.rng.random(len(rows))
        if self.limit and self.rows is not None and len(self.rows) == self.limit:
            rows = rows.loc[rows._priority < self.rows._priority.max()]
        if rows.empty:
            return
        combined = rows if self.rows is None else pd.concat([self.rows, rows], ignore_index=True)
        if self.limit and len(combined) > self.limit:
            take = np.argpartition(combined._priority.to_numpy(), self.limit - 1)[: self.limit]
            combined = combined.iloc[take]
        self.rows = combined.reset_index(drop=True)

    def finish(self) -> pd.DataFrame:
        if self.rows is None or self.rows.empty:
            raise ValueError("No eligible windows: inspect coverage and split dates.")
        return (
            self.rows.drop(columns="_priority")
            .sort_values(["TimeStamp", "turbine"])
            .reset_index(drop=True)
        )


def prepare_sample(
    data_root: Path,
    cfg: dict[str, Any],
    sensors: list[str],
    begin: str,
    end: str,
    limit: int | None,
    seed: int,
    audit: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Stream turbine-years; generate features in memory without a huge cache."""
    paths = discover(data_root, cfg, begin, end)
    sample = UniformSample(limit, seed)
    summaries = []
    counts = np.zeros(len(cfg["horizons_minutes"]), dtype="int64")
    missing_sum = pd.Series(0, index=sensors, dtype="int64")
    observed_total = 0
    for number, path in enumerate(paths, 1):
        frame = read_frame(path, sensors)
        features, complete = feature_windows(frame, sensors)
        targets = binary_targets(frame, cfg["horizons_minutes"])
        keep = split_mask(frame.index, complete, begin, end)
        keep &= targets.notna().all(axis=1).to_numpy()
        eligible = int(keep.sum())
        counts += targets.loc[keep].sum().to_numpy(dtype="int64")
        rows = features.loc[keep].copy()
        rows["TimeStamp"] = frame.index[keep]
        rows["turbine"] = path.parent.name.removeprefix("Station_")
        for name in targets:
            rows[name] = targets.loc[keep, name].to_numpy(dtype="int8")
        sample.add(rows)
        if audit:
            seen = frame.sensorObserved.astype(bool)
            missing_sum += frame.reindex(columns=sensors).loc[seen].isna().sum()
            observed_total += int(seen.sum())
            summaries.append(
                {
                    "year": int(path.parent.parent.name),
                    "turbine": rows.turbine.iloc[0] if eligible else path.parent.name[8:],
                    "grid_rows": len(frame),
                    "observed_rows": int(seen.sum()),
                    "coverage": float(seen.mean()),
                    "alarm_start_bins": int(frame.alarmOccurred.sum()),
                    "eligible_windows": eligible,
                    **{
                        f"positive_{h}min": int(targets.loc[keep, f"target_{h}min"].sum())
                        for h in cfg["horizons_minutes"]
                    },
                }
            )
        if number % 10 == 0 or number == len(paths):
            print(
                f"Prepared {number}/{len(paths)} files "
                f"({begin[:4]} to {pd.Timestamp(end).year - 1})",
                flush=True,
            )
    rows = sample.finish()
    details = {
        "eligible_population": sample.seen,
        "sample_rows": len(rows),
        "uniform_sample": limit is not None,
        "population_positives": {
            str(h): int(n) for h, n in zip(cfg["horizons_minutes"], counts, strict=True)
        },
        "missing_sensor_fraction": (missing_sum / max(observed_total, 1)).to_dict()
        if audit
        else {},
    }
    return rows, pd.DataFrame(summaries), details


def best_threshold(y: np.ndarray, scores: np.ndarray) -> float:
    """Maximise validation F1. Ties prefer the higher threshold."""
    precision, recall, thresholds = precision_recall_curve(y, scores)
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-15)
    return float(thresholds[np.flatnonzero(np.isclose(f1, f1.max()))[-1]])


def metrics(y: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    prediction = scores >= threshold
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    return {
        "rows": len(y),
        "positive_rate": float(np.mean(y)),
        "threshold": threshold,
        "accuracy": accuracy_score(y, prediction),
        "balanced_accuracy": balanced_accuracy_score(y, prediction),
        "precision": precision_score(y, prediction, zero_division=0),
        "recall": recall_score(y, prediction, zero_division=0),
        "f1": f1_score(y, prediction, zero_division=0),
        "average_precision": average_precision_score(y, scores),
        "roc_auc": roc_auc_score(y, scores) if len(np.unique(y)) == 2 else float("nan"),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def models(cfg: dict[str, Any], y: np.ndarray) -> dict[str, Any]:
    """Train-only imputation/scaling; class weighting is learned from train only."""

    def impute() -> SimpleImputer:
        return SimpleImputer(strategy="median", keep_empty_features=True)

    result = {"prevalence_baseline": make_pipeline(impute(), DummyClassifier(strategy="prior"))}
    if "logistic_regression" in cfg["models"]:
        result["logistic_regression"] = make_pipeline(
            impute(),
            StandardScaler(),
            LogisticRegression(
                class_weight="balanced", max_iter=1000, tol=1e-3, random_state=cfg["seed"]
            ),
        )
    if "random_forest" in cfg["models"]:
        result["random_forest"] = make_pipeline(
            impute(),
            RandomForestClassifier(
                **cfg["random_forest"],
                class_weight="balanced_subsample",
                n_jobs=cfg["n_jobs"],
                random_state=cfg["seed"],
            ),
        )
    if "xgboost" in cfg["models"]:
        from xgboost import XGBClassifier

        ratio = float((y == 0).sum() / max((y == 1).sum(), 1))
        result["xgboost"] = make_pipeline(
            impute(),
            XGBClassifier(
                **cfg["xgboost"],
                objective="binary:logistic",
                tree_method="hist",
                eval_metric="logloss",
                scale_pos_weight=ratio,
                n_jobs=cfg["n_jobs"],
                random_state=cfg["seed"],
            ),
        )
    return result


def new_run(repo: Path, cfg: dict[str, Any]) -> Path:
    digest = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:8]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    out = repo / cfg["report_root"] / f"{stamp}_{digest}"
    out.mkdir(parents=True, exist_ok=False)
    (out / "configuration.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    return out


def development(repo: Path, cfg: dict[str, Any], out: Path) -> dict[str, Any]:
    """Prepare train/validation and draw EDA BEFORE any model fitting."""
    root = repo / cfg["data_root"]
    sensors = sensor_columns(discover(root, cfg, cfg["train_start"], cfg["train_end"]))
    train, train_audit, train_info = prepare_sample(
        root,
        cfg,
        sensors,
        cfg["train_start"],
        cfg["train_end"],
        cfg["max_train_rows"],
        cfg["seed"],
        audit=True,
    )
    valid, valid_audit, valid_info = prepare_sample(
        root,
        cfg,
        sensors,
        cfg["train_end"],
        cfg["validation_end"],
        cfg["max_validation_rows"],
        cfg["seed"] + 1,
        audit=True,
    )
    audit = pd.concat([train_audit.assign(split="train"), valid_audit.assign(split="validation")])
    audit.to_csv(out / "data_quality.csv", index=False)
    (out / "sampling.json").write_text(
        json.dumps({"train": train_info, "validation": valid_info}, indent=2)
    )
    for h in cfg["horizons_minutes"]:
        for name, data in [("train", train), ("validation", valid)]:
            if data[f"target_{h}min"].nunique() != 2:
                raise ValueError(
                    f"{name}/{h}min needs both classes. Inspect audit or increase sample size."
                )
    plot_data(repo, cfg, out, audit, train_info)
    return {
        "repo": repo,
        "train": train,
        "validation": valid,
        "sensors": sensors,
        "features": [n for n in train if n.startswith("wtc_")],
        "sampling": {"train": train_info, "validation": valid_info},
        "output": out,
    }


def plot_data(
    repo: Path, cfg: dict[str, Any], out: Path, audit: pd.DataFrame, train_info: dict[str, Any]
) -> None:
    import matplotlib.pyplot as plt

    coverage = audit.pivot(index="turbine", columns="year", values="coverage")
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), constrained_layout=True)
    im = axes[0].imshow(coverage, vmin=0, vmax=1, aspect="auto", cmap="YlGnBu")
    axes[0].set(
        xticks=range(len(coverage.columns)),
        xticklabels=coverage.columns,
        yticks=range(len(coverage.index)),
        yticklabels=coverage.index,
        title="Source sensor coverage (development years)",
        xlabel="Year",
        ylabel="Turbine",
    )
    fig.colorbar(im, ax=axes[0], label="Fraction of grid rows with readings")
    missing = pd.Series(train_info["missing_sensor_fraction"]).sort_values().tail(15)
    missing.index = missing.index.str.removeprefix("wtc_")
    missing.plot.barh(ax=axes[1], color="#367c9b")
    axes[1].set(
        title="Training sensor missingness: highest 15",
        xlabel="Missing fraction among observed rows",
    )
    fig.savefig(out / "01_coverage_missingness.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
    for h in cfg["horizons_minutes"]:
        totals = audit.groupby("year")[[f"positive_{h}min", "eligible_windows"]].sum()
        ax.plot(
            totals.index,
            totals[f"positive_{h}min"] / totals.eligible_windows,
            marker="o",
            label=f"Next {h} minutes",
        )
    ax.set(
        title="Alarm-start prevalence in all eligible development windows",
        xlabel="Year",
        ylabel="Fraction labelled yes",
    )
    ax.legend()
    fig.savefig(out / "02_alarm_prevalence.png", dpi=150)
    plt.close(fig)

    path = (
        repo
        / cfg["data_root"]
        / str(cfg["eda_year"])
        / f"Station_{cfg['eda_turbine']}"
        / "data.parquet"
    )
    if not (
        pd.Timestamp(cfg["train_start"])
        <= pd.Timestamp(cfg["eda_start"])
        < pd.Timestamp(cfg["validation_end"])
    ):
        raise ValueError("EDA date must be inside development years.")
    frame = read_frame(path, cfg["eda_features"])
    begin = pd.Timestamp(cfg["eda_start"])
    view = frame.loc[
        (frame.index >= begin) & (frame.index < begin + pd.Timedelta(days=cfg["eda_days"]))
    ]
    if view.empty:
        raise ValueError("Selected EDA date has no rows.")
    fig, axes = plt.subplots(
        len(cfg["eda_features"]) + 1, 1, figsize=(13, 8), sharex=True, constrained_layout=True
    )
    for ax, name in zip(axes, cfg["eda_features"], strict=False):
        ax.plot(view.index, view[name].where(view.sensorObserved), linewidth=0.8, color="#246b92")
        ax.set_ylabel(name.removeprefix("wtc_"), fontsize=8)
    axes[-1].step(
        view.index,
        view.alarmOccurred.astype(int),
        where="post",
        color="#c44e52",
        label="New alarm start",
    )
    axes[-1].set(ylabel="Alarm (0/1)", ylim=(-0.1, 1.2), yticks=[0, 1])
    axes[0].set_title(
        f"Turbine {cfg['eda_turbine']}: sensor readings and ANY alarm starts (gaps remain missing)"
    )
    fig.savefig(out / "03_sensors_and_alarm_starts.png", dpi=150)
    plt.close(fig)


def fit_models(cfg: dict[str, Any], prepared: dict[str, Any]) -> pd.DataFrame:
    """Fit each horizon independently; select thresholds and winners on validation."""
    out = prepared["output"]
    columns = prepared["features"]
    train, valid = prepared["train"], prepared["validation"]
    x_train, x_valid = train[columns].to_numpy(), valid[columns].to_numpy()
    rows, frozen = [], {}
    for horizon in cfg["horizons_minutes"]:
        y = train[f"target_{horizon}min"].to_numpy()
        y_valid = valid[f"target_{horizon}min"].to_numpy()
        prediction_table = valid[["TimeStamp", "turbine", f"target_{horizon}min"]].copy()
        for name, model in models(cfg, y).items():
            start = time.perf_counter()
            print(f"Fitting {name}, next {horizon} minutes...", flush=True)
            model.fit(x_train, y)
            scores = model.predict_proba(x_valid)[:, 1]
            threshold = 0.5 if name == "prevalence_baseline" else best_threshold(y_valid, scores)
            row = {
                "horizon_minutes": horizon,
                "model": name,
                "split": "validation",
                "fit_seconds": time.perf_counter() - start,
                **metrics(y_valid, scores, threshold),
            }
            rows.append(row)
            prediction_table[f"{name}_score"] = scores
            prediction_table[f"{name}_yes"] = (scores >= threshold).astype("int8")
            bundle = {
                "version": VERSION,
                "model": model,
                "model_name": name,
                "horizon_minutes": horizon,
                "threshold": threshold,
                "sensors": prepared["sensors"],
                "features": columns,
                "lookback_minutes": 120,
                "target": "any_alarm_start",
                "configuration": cfg,
            }
            joblib.dump(bundle, out / f"{horizon}min_{name}.joblib", compress=3)
            estimator = model.steps[-1][1]
            importance = getattr(estimator, "feature_importances_", None)
            if importance is None and hasattr(estimator, "coef_"):
                importance = np.abs(estimator.coef_[0])
            if importance is not None:
                pd.Series(importance, index=columns, name="importance").sort_values(
                    ascending=False
                ).to_csv(out / f"{horizon}min_{name}_feature_importance.csv", index_label="feature")
            print(f"  validation AP={row['average_precision']:.3f}, F1={row['f1']:.3f}", flush=True)
        prediction_table.to_parquet(
            out / f"validation_{horizon}min_predictions.parquet", index=False
        )
        candidates = pd.DataFrame(rows)
        candidates = candidates.loc[
            (candidates.horizon_minutes == horizon) & (candidates.model != "prevalence_baseline")
        ]
        winner = candidates.sort_values(
            ["average_precision", "f1", "model"], ascending=[False, False, True]
        ).iloc[0]
        frozen[str(horizon)] = {
            "model": winner.model,
            "threshold": float(winner.threshold),
            "selection": "highest validation average precision; F1 tie-break",
        }
        pd.DataFrame(rows).to_csv(out / "validation_metrics.csv", index=False)
    (out / "frozen_selection.json").write_text(json.dumps(frozen, indent=2))
    (out / "status.json").write_text(
        json.dumps({"status": "frozen_before_test", "version": VERSION})
    )
    plot_evaluation(out, pd.DataFrame(rows), "validation")
    plot_forecasts(prepared["repo"], cfg, out)
    return pd.DataFrame(rows)


def evaluate_test(repo: Path, cfg: dict[str, Any], prepared: dict[str, Any]) -> pd.DataFrame:
    """Open test data only AFTER all fits, thresholds and model choices are frozen."""
    out = prepared["output"]
    if not (out / "frozen_selection.json").is_file():
        raise RuntimeError("Fit and freeze models before evaluating test years.")
    if (out / "test_metrics.csv").exists():
        raise RuntimeError("Test has already been evaluated in this run. Read the saved metrics.")
    test, _, info = prepare_sample(
        repo / cfg["data_root"],
        cfg,
        prepared["sensors"],
        cfg["validation_end"],
        cfg["test_end"],
        cfg["max_test_rows"],
        cfg["seed"] + 2,
    )
    sampling = {**prepared["sampling"], "test": info}
    (out / "sampling.json").write_text(json.dumps(sampling, indent=2))
    rows, per_turbine = [], []
    x_test = test[prepared["features"]].to_numpy()
    for horizon in cfg["horizons_minutes"]:
        y = test[f"target_{horizon}min"].to_numpy()
        if len(np.unique(y)) != 2:
            raise ValueError("Test needs both classes for this comparison.")
        predictions = test[["TimeStamp", "turbine", f"target_{horizon}min"]].copy()
        for name in ["prevalence_baseline", *cfg["models"]]:
            bundle = joblib.load(out / f"{horizon}min_{name}.joblib")
            scores = bundle["model"].predict_proba(x_test)[:, 1]
            threshold = bundle["threshold"]
            rows.append(
                {
                    "horizon_minutes": horizon,
                    "model": name,
                    "split": "test",
                    **metrics(y, scores, threshold),
                }
            )
            predictions[f"{name}_score"] = scores
            predictions[f"{name}_yes"] = (scores >= threshold).astype("int8")
            for turbine in sorted(test.turbine.unique()):
                keep = test.turbine.eq(turbine).to_numpy()
                per_turbine.append(
                    {
                        "turbine": turbine,
                        "horizon_minutes": horizon,
                        "model": name,
                        **metrics(y[keep], scores[keep], threshold),
                    }
                )
        predictions.to_parquet(out / f"test_{horizon}min_predictions.parquet", index=False)
    result = pd.DataFrame(rows)
    result.to_csv(out / "test_metrics.csv", index=False)
    pd.DataFrame(per_turbine).to_csv(out / "test_per_turbine_metrics.csv", index=False)
    plot_evaluation(out, result, "test")
    (out / "status.json").write_text(json.dumps({"status": "complete", "version": VERSION}))
    return result


def plot_evaluation(out: Path, table: pd.DataFrame, split: str) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    for ax, horizon in zip(axes, [10, 20, 30], strict=True):
        frame = pd.read_parquet(out / f"{split}_{horizon}min_predictions.parquet")
        y = frame[f"target_{horizon}min"].to_numpy()
        for row in table.loc[table.horizon_minutes == horizon].itertuples():
            if row.model == "prevalence_baseline":
                continue
            precision, recall, _ = precision_recall_curve(y, frame[f"{row.model}_score"])
            ax.plot(recall, precision, label=f"{row.model} (AP {row.average_precision:.3f})")
        ax.axhline(y.mean(), linestyle="--", color="grey", label=f"Prevalence {y.mean():.3f}")
        ax.set(
            xlabel="Recall",
            ylabel="Precision",
            title=f"{split.title()}: next {horizon} min",
            xlim=(0, 1),
            ylim=(0, 1),
        )
        ax.legend(fontsize=7)
    fig.savefig(out / f"04_{split}_precision_recall.png", dpi=150)
    plt.close(fig)
    selected = json.loads((out / "frozen_selection.json").read_text())
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5), constrained_layout=True)
    for ax, horizon in zip(axes, [10, 20, 30], strict=True):
        name = selected[str(horizon)]["model"]
        row = table.loc[(table.horizon_minutes == horizon) & (table.model == name)].iloc[0]
        matrix = np.array([[row.tn, row.fp], [row.fn, row.tp]], dtype=int)
        ax.imshow(matrix, cmap="Blues")
        for (i, j), value in np.ndenumerate(matrix):
            ax.text(
                j,
                i,
                f"{value:,}",
                ha="center",
                va="center",
                color="white" if value > matrix.max() / 2 else "black",
            )
        ax.set(
            xticks=[0, 1],
            xticklabels=["No", "Yes"],
            yticks=[0, 1],
            yticklabels=["No", "Yes"],
            xlabel="Predicted",
            ylabel="Actual",
            title=f"{horizon} min: {name}",
        )
    fig.suptitle(
        f"{split.title()} — model chosen on validation AP; threshold chosen on validation F1"
    )
    fig.savefig(out / f"05_{split}_confusion_matrices.png", dpi=150)
    plt.close(fig)


def plot_forecasts(repo: Path, cfg: dict[str, Any], out: Path) -> None:
    """A fixed validation week: continuous scores alongside actual future targets."""
    import matplotlib.pyplot as plt

    selected = json.loads((out / "frozen_selection.json").read_text())
    first = joblib.load(out / f"10min_{selected['10']['model']}.joblib")
    begin = pd.Timestamp(cfg["train_end"]) + pd.Timedelta(days=31)
    finish = min(
        begin + pd.Timedelta(days=7), pd.Timestamp(cfg["validation_end"]) - pd.Timedelta(minutes=30)
    )
    if finish <= begin:
        return
    year = begin.year
    source = repo / cfg["data_root"] / str(year) / f"Station_{cfg['eda_turbine']}" / "data.parquet"
    frame = read_frame(source, first["sensors"])
    features, complete = feature_windows(frame, first["sensors"])
    targets = binary_targets(frame, cfg["horizons_minutes"])
    keep = (
        complete.to_numpy()
        & (frame.index >= begin)
        & (frame.index < finish)
        & targets.notna().all(axis=1).to_numpy()
    )
    view = pd.DataFrame(index=frame.index[keep])
    if len(view) == 0:
        return
    fig, axes = plt.subplots(3, 1, figsize=(13, 8), sharex=True, constrained_layout=True)
    for ax, horizon in zip(axes, cfg["horizons_minutes"], strict=True):
        bundle = joblib.load(out / f"{horizon}min_{selected[str(horizon)]['model']}.joblib")
        scores = bundle["model"].predict_proba(features.loc[keep, bundle["features"]].to_numpy())[
            :, 1
        ]
        target = targets.loc[keep, f"target_{horizon}min"].to_numpy()
        view[f"actual_{horizon}min"] = target.astype("int8")
        view[f"score_{horizon}min"] = scores
        view[f"predicted_yes_{horizon}min"] = (scores >= bundle["threshold"]).astype("int8")
        # Preserve missing-history gaps rather than drawing a line through them.
        full = pd.DataFrame(index=frame.loc[(frame.index >= begin) & (frame.index < finish)].index)
        full["score"] = pd.Series(scores, index=view.index)
        full["target"] = pd.Series(target, index=view.index)
        ax.plot(full.index, full.score, linewidth=0.8, color="#246b92", label="Model score")
        ax.fill_between(
            full.index,
            0,
            1,
            where=full.target.eq(1),
            step="post",
            alpha=0.2,
            color="#c44e52",
            label="Actual next-window alarm = yes",
        )
        ax.axhline(
            bundle["threshold"], color="#d48128", linestyle="--", label="Frozen yes/no threshold"
        )
        ax.set(ylim=(0, 1), ylabel=f"Next {horizon} min", title=selected[str(horizon)]["model"])
        ax.legend(fontsize=7, loc="upper right")
    fig.suptitle(
        f"Validation: turbine {cfg['eda_turbine']} — scores are not calibrated probabilities"
    )
    fig.savefig(out / "06_validation_forecast_timeline.png", dpi=150)
    plt.close(fig)
    view.to_csv(out / "validation_week_example.csv", index_label="forecast_time")


def predict_history(
    bundle: dict[str, Any], history: pd.DataFrame, forecast_time: str | pd.Timestamp
) -> dict[str, Any]:
    """Use precisely [t-120min,t) sensor rows; does not require future/label data."""
    t = pd.Timestamp(forecast_time)
    if t.tz is not None or t != t.floor("10min"):
        raise ValueError("Forecast time must be timezone-naive and on the 10-minute grid.")
    expected = pd.date_range(t - pd.Timedelta(minutes=120), periods=12, freq="10min")
    if not isinstance(history.index, pd.DatetimeIndex) or history.index.has_duplicates:
        raise ValueError("History must have a unique timestamp index.")
    prior = history.reindex(index=expected, columns=[*bundle["sensors"], "sensorObserved"]).copy()
    if (
        "sensorObserved" not in prior
        or prior.sensorObserved.isna().any()
        or not prior.sensorObserved.eq(True).all()
    ):
        raise ValueError("Need all 12 observed intervals in the preceding two hours.")
    # Append an empty row at t only to compute the same shifted feature function.
    placeholder = prior.iloc[[-1]].copy()
    placeholder.index = pd.DatetimeIndex([t])
    placeholder.loc[:, bundle["sensors"]] = np.nan
    features, complete = feature_windows(pd.concat([prior, placeholder]), bundle["sensors"])
    assert complete.iloc[-1]
    score = float(
        bundle["model"].predict_proba(features[bundle["features"]].iloc[[-1]].to_numpy())[0, 1]
    )
    return {
        "forecast_time": str(t),
        "horizon_minutes": bundle["horizon_minutes"],
        "alarm_prediction": "yes" if score >= bundle["threshold"] else "no",
        "model_score": score,
        "threshold": bundle["threshold"],
    }
