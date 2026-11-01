"""Purged chronological development and explicitly gated reserved-year evaluation."""

from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    fbeta_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

VERSION = 2
STEP = pd.Timedelta(minutes=10)
LOOKBACK = pd.Timedelta(hours=2)
HORIZON = pd.Timedelta(hours=6)
FEATURE_STATS = ("mean", "std", "min", "max")


def discover(root, years):
    """Only enumerate the requested years: development never opens reserved years."""
    paths, absent = [], []
    for year in sorted(set(years)):
        directory = Path(root) / str(year)
        found = sorted(directory.glob("Station_*/data.parquet")) if directory.is_dir() else []
        if not found:
            absent.append(year)
        paths.extend(found)
    return paths, absent


def sensor_names(paths):
    import pyarrow.parquet as pq

    return sorted(
        {name for path in paths for name in pq.read_schema(path).names if name.startswith("wtc_")}
    )


def episode_table(frame, codes, turbine):
    """Deduplicate exact event records, then merge overlapping target-code activity."""
    events = list(frame.attrs.get("carry_in_alarm_events", []))
    for text in frame.loc[frame.alarmOccurred, "alarmEvents"]:
        events.extend(json.loads(text))
    data = pd.DataFrame(events, columns=["Alarmcode", "TimeOn", "TimeOff"])
    data["Alarmcode"] = data.Alarmcode.astype("string").str.strip()
    data = data.loc[data.Alarmcode.isin(set(codes))].copy()
    for name in ["TimeOn", "TimeOff"]:
        data[name] = pd.to_datetime(data[name], errors="raise").astype("datetime64[ns]")
    if data.TimeOn.isna().any() or (data.TimeOff < data.TimeOn).any():
        raise ValueError("Missing alarm start or end before start; inspect source events.")
    before = len(data)
    data = data.drop_duplicates(["Alarmcode", "TimeOn", "TimeOff"])
    duplicates = before - len(data)
    in_year = data.TimeOn.ge(frame.index.min()) & data.TimeOn.lt(frame.index.max() + STEP)
    code_counts = {str(k): int(v) for k, v in data.loc[in_year, "Alarmcode"].value_counts().items()}
    unknown = data.TimeOff.isna() | data.TimeOff.le(data.TimeOn)
    data.loc[unknown, "TimeOff"] = data.loc[unknown, "TimeOn"].dt.floor("10min") + STEP
    episodes = []
    for event in data.sort_values(["TimeOn", "TimeOff"]).itertuples(index=False):
        if episodes and event.TimeOn < episodes[-1]["TimeOff"]:
            prior = episodes[-1]
            prior["TimeOff"] = max(prior["TimeOff"], event.TimeOff)
            prior["codes"].add(str(event.Alarmcode))
            prior["records"] += 1
        else:
            episodes.append(
                {
                    "turbine": str(turbine),
                    "TimeOn": event.TimeOn,
                    "TimeOff": event.TimeOff,
                    "codes": {str(event.Alarmcode)},
                    "records": 1,
                }
            )
    for episode in episodes:
        episode["codes"] = "; ".join(sorted(episode["codes"]))
    result = pd.DataFrame(episodes, columns=["turbine", "TimeOn", "TimeOff", "codes", "records"])
    for name in ["TimeOn", "TimeOff"]:
        result[name] = pd.to_datetime(result[name]).astype("datetime64[ns]")
    return result, {
        "exact_duplicates_removed": duplicates,
        "unknown_or_zero_durations": int(unknown.sum()),
        "code_records": code_counts,
        "target_episode_count": len(result),
    }


def prepare_one(source, cache_root, codes, sensors, rebuild=False):
    """One turbine-year at a time. No imputer or scaler is fitted here."""
    source = Path(source)
    year, turbine = int(source.parent.parent.name), source.parent.name.removeprefix("Station_")
    stat = source.stat()
    signature = {
        "version": VERSION,
        "source": str(source.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "codes": sorted(codes),
        "sensors": list(sensors),
        "horizon": str(HORIZON),
        "lookback": str(LOOKBACK),
    }
    digest = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()[:16]
    folder = Path(cache_root) / str(year) / f"Station_{turbine}" / digest
    manifest = folder / "manifest.json"
    if not rebuild and manifest.is_file():
        saved = json.loads(manifest.read_text(encoding="utf-8"))
        if (folder / "features.parquet").is_file() and (folder / "episodes.parquet").is_file():
            return {
                **saved,
                "features": str(folder / "features.parquet"),
                "episodes": str(folder / "episodes.parquet"),
            }
    frame = pd.read_parquet(source)
    grid = frame.index
    begin, end = pd.Timestamp(year, 1, 1), pd.Timestamp(year + 1, 1, 1)
    expected = pd.date_range(begin, end, freq=STEP, inclusive="left")
    if not isinstance(grid, pd.DatetimeIndex) or not grid.equals(expected):
        raise ValueError(f"Expected a complete regular year grid: {source}; rerun preprocessing.")
    observed = frame.sensorObserved.fillna(False).astype(bool)
    episodes, audit = episode_table(frame, codes, turbine)
    values = frame.reindex(columns=sensors).apply(pd.to_numeric, errors="coerce")
    values = values.replace([np.inf, -np.inf], np.nan).where(observed, axis=0)
    history_steps = int(LOOKBACK / STEP)
    rolling = values.shift(1).rolling(history_steps, min_periods=1)
    parts = []
    for name in FEATURE_STATS:
        part = getattr(rolling, name)()
        part.columns = [f"{c}__{name}" for c in sensors]
        parts.append(part)
    features = pd.concat(parts, axis=1).astype("float32")
    features.attrs = {}
    starts = np.sort(episodes.TimeOn.to_numpy(dtype="datetime64[ns]"))
    times = grid.to_numpy(dtype="datetime64[ns]")
    target = (
        np.searchsorted(starts, times + HORIZON.to_timedelta64(), side="left")
        > np.searchsorted(starts, times, side="left")
    ).astype("int8")
    changes = np.zeros(len(grid) + 1, dtype="int32")
    for event in episodes.itertuples():
        left = grid.searchsorted(event.TimeOn, side="right")
        right = grid.searchsorted(event.TimeOff, side="left")
        if right > left:
            changes[left] += 1
            changes[right] -= 1
    complete_history = (
        observed.shift(1, fill_value=False)
        .rolling(history_steps, min_periods=history_steps)
        .sum()
        .eq(history_steps)
    )
    keep = complete_history.to_numpy() & ((grid + HORIZON) <= end) & (np.cumsum(changes[:-1]) == 0)
    output = features.loc[keep].copy()
    output.insert(0, "TimeStamp", grid[keep])
    output["target"] = target[keep]
    output["turbine"] = str(turbine)
    stats = {
        "year": year,
        "turbine": str(turbine),
        "observed_sensor_rows": int(observed.sum()),
        "eligible_rows": len(output),
        "positive_windows": int(output.target.sum()),
        **audit,
        "signature": signature,
    }
    if source.stat().st_mtime_ns != stat.st_mtime_ns:
        raise RuntimeError("Source changed during preparation; rerun after it finishes updating.")
    folder.mkdir(parents=True, exist_ok=True)
    output.to_parquet(folder / "features.parquet", index=False)
    episodes.to_parquet(folder / "episodes.parquet", index=False)
    manifest.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return {
        **stats,
        "features": str(folder / "features.parquet"),
        "episodes": str(folder / "episodes.parquet"),
    }


def prepare(paths, cache_root, codes, sensors, rebuild=False):
    entries = []
    for number, path in enumerate(paths, 1):
        entry = prepare_one(path, cache_root, codes, sensors, rebuild)
        entries.append(entry)
        if number % 10 == 0 or number == len(paths):
            print(f"Prepared {number}/{len(paths)} turbine-years", flush=True)
    return entries


def iter_frames(entries, begin, end, columns=None):
    """Purging is applied with shared timestamps, independently of turbine ordering."""
    begin, end = pd.Timestamp(begin), pd.Timestamp(end)
    for entry in entries:
        if entry["eligible_rows"] == 0 or entry["year"] < begin.year or entry["year"] > end.year:
            continue
        frame = pd.read_parquet(entry["features"], columns=columns)
        mask = (frame.TimeStamp - LOOKBACK >= begin) & (frame.TimeStamp + HORIZON <= end)
        if mask.any():
            yield entry, frame.loc[mask].reset_index(drop=True)


def load_episodes(entries, begin, end):
    parts = []
    for entry in entries:
        if pd.Timestamp(begin).year <= entry["year"] <= pd.Timestamp(end).year:
            part = pd.read_parquet(entry["episodes"])
            parts.append(
                part.loc[part.TimeOn.ge(pd.Timestamp(begin)) & part.TimeOn.lt(pd.Timestamp(end))]
            )
    if not parts:
        return pd.DataFrame(columns=["turbine", "TimeOn", "TimeOff", "codes", "records"])
    return pd.concat(parts, ignore_index=True).drop_duplicates(["turbine", "TimeOn", "TimeOff"])


def training_sample(entries, begin, end, features, max_negatives=250000, seed=42):
    """Use every positive and a stratified negative sample, with inverse-probability weights."""
    stats = []
    for entry, frame in iter_frames(entries, begin, end, ["TimeStamp", "target", "turbine"]):
        positive = int(frame.target.sum())
        stats.append((entry["features"], positive, len(frame) - positive))
    positives, negatives = sum(x[1] for x in stats), sum(x[2] for x in stats)
    if positives == 0 or negatives == 0:
        raise ValueError(
            f"Training interval {begin} to {end} needs both classes: "
            f"positives={positives}, negatives={negatives}."
        )
    negative_counts = np.array([x[2] for x in stats], dtype=int)
    budget = negatives if max_negatives is None else min(int(max_negatives), negatives)
    strata = int((negative_counts > 0).sum())
    if budget < strata:
        raise ValueError(
            f"Negative budget must be at least {strata} to represent all turbine-year strata."
        )
    allocations = (negative_counts > 0).astype(int)
    capacities = negative_counts - allocations
    if capacities.sum():
        ideal = capacities * ((budget - allocations.sum()) / capacities.sum())
        extra = np.floor(ideal).astype(int)
        remainder = budget - allocations.sum() - extra.sum()
        if remainder:
            indices = np.argsort(-(ideal - extra), kind="stable")[:remainder]
            extra[indices] += 1
        allocations += extra
    allocation = {row[0]: int(n) for row, n in zip(stats, allocations, strict=True)}
    rng = np.random.default_rng(seed)
    samples, weights = [], []
    for entry, frame in iter_frames(entries, begin, end):
        pos = np.flatnonzero(frame.target.to_numpy() == 1)
        neg = np.flatnonzero(frame.target.to_numpy() == 0)
        amount = allocation[entry["features"]]
        chosen = rng.choice(neg, size=amount, replace=False) if amount < len(neg) else neg
        positions = np.r_[pos, chosen]
        samples.append(frame.iloc[positions])
        weights.append(
            np.r_[
                np.ones(len(pos)),
                np.full(len(chosen), len(neg) / len(chosen) if len(chosen) else 0),
            ]
        )
    sample = pd.concat(samples, ignore_index=True)
    inverse = np.concatenate(weights)
    label = sample.target.to_numpy(dtype="int8")
    # Explicit full-population class balancing, then normalise for regularisation stability.
    class_weights = np.where(
        label == 1,
        (positives + negatives) / (2 * positives),
        (positives + negatives) / (2 * negatives),
    )
    fit_weights = inverse * class_weights
    fit_weights /= fit_weights.mean()
    metadata = {
        "full_positive_rows": positives,
        "full_negative_rows": negatives,
        "sample_rows": len(sample),
        "sample_negative_rows": int((label == 0).sum()),
        "turbines": sorted(sample.turbine.unique().tolist()),
        "positive_prevalence": positives / (positives + negatives),
    }
    return sample[features], label, fit_weights, metadata


def fit_model(candidate, x, y, weights, stats, seed=42):
    kind = candidate["kind"]
    result = {"kind": kind, "candidate": candidate, "feature_names": list(x.columns)}
    if kind == "constant":
        result["prevalence"] = stats["positive_prevalence"]
        return result
    preprocessor = make_pipeline(
        SimpleImputer(strategy="median", keep_empty_features=True),
        VarianceThreshold(),
        StandardScaler(),
    )
    transformed = preprocessor.fit_transform(x)
    result["preprocessor"] = preprocessor
    if kind == "random":
        result["weights"] = np.random.default_rng(seed).normal(size=transformed.shape[1]) / np.sqrt(
            transformed.shape[1]
        )
        return result
    parameters = candidate.get("parameters", {})
    if kind == "logistic":
        estimator = LogisticRegression(
            solver="newton-cholesky", max_iter=100, random_state=seed, **parameters
        )
        transformed = transformed.astype("float64")
    elif kind == "lightgbm":
        estimator = LGBMClassifier(random_state=seed, n_jobs=4, verbosity=-1, **parameters)
    elif kind == "forest":
        estimator = RandomForestClassifier(random_state=seed, n_jobs=4, **parameters)
    else:
        raise ValueError(f"Unknown model kind: {kind}")
    estimator.fit(transformed, y, sample_weight=weights)
    if kind == "logistic" and np.any(estimator.n_iter_ >= estimator.max_iter):
        raise RuntimeError("Logistic regression did not converge; fix it before comparing models.")
    result["estimator"] = estimator
    return result


def predict(model, frame, batch_size=25000):
    values = []
    for offset in range(0, len(frame), batch_size):
        batch = frame.iloc[offset : offset + batch_size]
        if model["kind"] == "constant":
            scores = np.full(len(batch), model["prevalence"])
        else:
            z = model["preprocessor"].transform(batch[model["feature_names"]])
            if model["kind"] == "random":
                scores = 1 / (1 + np.exp(-np.clip(z @ model["weights"], -30, 30)))
            else:
                if model["kind"] == "logistic":
                    z = z.astype("float64")
                scores = model["estimator"].predict_proba(z)[:, 1]
        values.append(scores)
    return np.concatenate(values) if values else np.array([], dtype=float)


def score_interval(model, entries, begin, end):
    parts = []
    for _, frame in iter_frames(entries, begin, end):
        output = frame[["TimeStamp", "turbine", "target"]].copy()
        output["score"] = predict(model, frame)
        parts.append(output)
    if not parts:
        raise ValueError(f"No eligible predictions in {begin} to {end}")
    return pd.concat(parts, ignore_index=True).sort_values(["turbine", "TimeStamp"])


def event_metrics(predictions, episodes, threshold):
    eligible, caught, warnings, false_warnings, lead_hours = 0, 0, 0, 0, []
    for turbine, rows in predictions.groupby("turbine", sort=False):
        rows = rows.sort_values("TimeStamp")
        times = rows.TimeStamp.to_numpy(dtype="datetime64[ns]")
        warning_times = times[rows.score.to_numpy() >= threshold]
        events = episodes.loc[episodes.turbine.astype(str).eq(str(turbine))]
        event_times = events.TimeOn.to_numpy(dtype="datetime64[ns]")
        for onset in event_times:
            left = times.searchsorted(onset - HORIZON.to_timedelta64(), side="right")
            right = times.searchsorted(onset, side="right")
            if right > left:
                eligible += 1
                wl = warning_times.searchsorted(onset - HORIZON.to_timedelta64(), side="right")
                wr = warning_times.searchsorted(onset, side="right")
                if wr > wl:
                    caught += 1
                    lead_hours.append(float((onset - warning_times[wl]) / np.timedelta64(1, "h")))
        if len(warning_times):
            breaks = np.r_[
                0,
                np.flatnonzero(np.diff(warning_times) > STEP.to_timedelta64()) + 1,
                len(warning_times),
            ]
            for left, right in zip(breaks[:-1], breaks[1:], strict=True):
                warnings += 1
                first, last = warning_times[left], warning_times[right - 1]
                if not (
                    (event_times >= first) & (event_times < last + HORIZON.to_timedelta64())
                ).any():
                    false_warnings += 1
    months = len(predictions) * STEP / pd.Timedelta(days=30)
    return {
        "eligible_alarm_episodes": eligible,
        "detected_alarm_episodes": caught,
        "event_recall": caught / eligible if eligible else np.nan,
        "warning_episodes": warnings,
        "false_warning_episodes": false_warnings,
        "false_warning_episodes_per_30_eligible_turbine_days": false_warnings / months
        if months
        else np.nan,
        "median_detected_lead_hours": float(np.median(lead_hours)) if lead_hours else np.nan,
    }


def metrics(predictions, episodes, threshold):
    y, scores = predictions.target.to_numpy(), predictions.score.to_numpy()
    labels = scores >= threshold
    tn, fp, fn, tp = confusion_matrix(y, labels, labels=[0, 1]).ravel()
    both = len(np.unique(y)) == 2
    return {
        "rows": len(y),
        "positives": int(y.sum()),
        "prevalence": float(y.mean()),
        "AP": average_precision_score(y, scores) if both else np.nan,
        "ROC_AUC": roc_auc_score(y, scores) if both else np.nan,
        "precision": precision_score(y, labels, zero_division=0),
        "recall": recall_score(y, labels, zero_division=0),
        "F1": f1_score(y, labels, zero_division=0),
        "F2": fbeta_score(y, labels, beta=2, zero_division=0),
        "warning_fraction": float(labels.mean()),
        "MCC": matthews_corrcoef(y, labels) if both else np.nan,
        "balanced_accuracy": balanced_accuracy_score(y, labels) if both else np.nan,
        "Brier": brier_score_loss(y, scores),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "threshold": float(threshold),
        **event_metrics(predictions, episodes, threshold),
    }


def choose_threshold(
    predictions, episodes, policy="max_f1", false_warning_budget=1.0, max_warning_fraction=0.05
):
    if predictions.target.nunique() < 2:
        raise ValueError("Threshold calibration needs both classes; extend the calibration period.")
    if policy == "max_f1":
        p, r, thresholds = precision_recall_curve(predictions.target, predictions.score)
        f1 = 2 * p[:-1] * r[:-1] / np.maximum(p[:-1] + r[:-1], 1e-15)
        # Among exact ties prefer the higher threshold (fewer warnings).
        return float(thresholds[np.flatnonzero(f1 == f1.max())[-1]])
    if policy != "false_warning_budget":
        raise ValueError("Choose max_f1 or false_warning_budget")
    thresholds = np.unique(
        np.r_[
            np.quantile(predictions.score, np.linspace(0.5, 1, 65)),
            np.nextafter(predictions.score.max(), np.inf),
        ]
    )
    ranked = []
    for threshold in thresholds:
        result = metrics(predictions, episodes, threshold)
        rate = result["false_warning_episodes_per_30_eligible_turbine_days"]
        # Avoid rewarding one effectively permanent warning as a single true episode.
        if rate <= false_warning_budget and result["warning_fraction"] <= max_warning_fraction:
            ranked.append(
                (np.nan_to_num(result["event_recall"]), result["precision"], -rate, threshold)
            )
    return float(max(ranked)[-1])


def per_turbine_metrics(predictions, episodes, threshold):
    return pd.DataFrame(
        [
            {
                "turbine": turbine,
                **metrics(
                    rows, episodes.loc[episodes.turbine.astype(str).eq(str(turbine))], threshold
                ),
            }
            for turbine, rows in predictions.groupby("turbine")
        ]
    )


def run_cv(
    entries,
    features,
    candidates,
    folds,
    output,
    max_negatives=250000,
    seed=42,
    threshold_policy="max_f1",
    false_warning_budget=1.0,
    max_warning_fraction=0.05,
):
    if any(entry["year"] < 2016 or entry["year"] > 2023 for entry in entries):
        raise ValueError(
            "Development entries must contain only 2016-2023, never reserved test years."
        )
    names = [candidate["name"] for candidate in candidates]
    if len(names) != len(set(names)) or any(not re.fullmatch(r"[A-Za-z0-9_-]+", n) for n in names):
        raise ValueError(
            "Use distinct candidate names containing letters, numbers, underscores or hyphens."
        )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    results, turbine_results, audits = [], [], []
    for fold in folds:
        begin, calibration_begin, evaluation_begin, end = map(
            pd.Timestamp,
            [fold["train_begin"], fold["calibration_begin"], fold["evaluation_begin"], fold["end"]],
        )
        if not begin < calibration_begin < evaluation_begin < end <= pd.Timestamp("2024-01-01"):
            raise ValueError("CV periods must be chronological and entirely before 2024.")
        x, y, weights, audit = training_sample(
            entries, begin, calibration_begin, features, max_negatives, seed
        )
        audits.append({"fold": fold["name"], **audit})
        calibration_episodes = load_episodes(entries, calibration_begin, evaluation_begin)
        evaluation_episodes = load_episodes(entries, evaluation_begin, end)
        for candidate in candidates:
            print(
                f"{fold['name']}: fitting {candidate['name']} on {len(y):,} sampled rows",
                flush=True,
            )
            started = time.perf_counter()
            model = fit_model(candidate, x, y, weights, audit, seed)
            calibration = score_interval(model, entries, calibration_begin, evaluation_begin)
            threshold = (
                0.5
                if candidate["kind"] == "constant"
                else choose_threshold(
                    calibration,
                    calibration_episodes,
                    threshold_policy,
                    false_warning_budget,
                    max_warning_fraction,
                )
            )
            evaluation = score_interval(model, entries, evaluation_begin, end)
            if evaluation.target.nunique() < 2:
                raise ValueError(
                    f"{fold['name']}: evaluation needs both classes; "
                    "adjust folds before comparison."
                )
            result = {
                "fold": fold["name"],
                "candidate": candidate["name"],
                "seconds": time.perf_counter() - started,
                **metrics(evaluation, evaluation_episodes, threshold),
            }
            turbine = per_turbine_metrics(evaluation, evaluation_episodes, threshold)
            result["macro_AP"] = turbine.AP.mean()
            result["turbines_with_defined_AP"] = int(turbine.AP.notna().sum())
            results.append(result)
            turbine["fold"], turbine["candidate"] = fold["name"], candidate["name"]
            turbine_results.append(turbine)
            directory = output / "oof" / candidate["name"]
            directory.mkdir(parents=True, exist_ok=True)
            evaluation.to_parquet(directory / f"{fold['name']}.parquet", index=False)
            del calibration, evaluation, model
        del x, y, weights
    table = pd.DataFrame(results)
    table.to_csv(output / "cv_metrics.csv", index=False)
    pd.concat(turbine_results, ignore_index=True).to_csv(output / "cv_per_turbine.csv", index=False)
    (output / "sampling_audit.json").write_text(json.dumps(audits, indent=2), encoding="utf-8")
    summary = table.groupby("candidate").agg(
        folds=("fold", "nunique"),
        mean_AP=("AP", "mean"),
        sd_AP=("AP", "std"),
        mean_macro_AP=("macro_AP", "mean"),
        mean_F1=("F1", "mean"),
        mean_precision=("precision", "mean"),
        mean_recall=("recall", "mean"),
        mean_event_recall=("event_recall", "mean"),
        mean_false_warning_episode_rate=(
            "false_warning_episodes_per_30_eligible_turbine_days",
            "mean",
        ),
    )
    summary = summary.sort_values("mean_AP", ascending=False)
    summary.to_csv(output / "cv_summary.csv")
    return table, summary


def final_refit(entries, features, candidates, folds, cv_output, output, config):
    """Freeze development-only selection and OOF thresholds before any reserved evaluation."""
    if any(entry["year"] < 2016 or entry["year"] > 2023 for entry in entries):
        raise ValueError("Final development refit must not receive reserved-year entries.")
    cv_output, output = Path(cv_output), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(cv_output / "cv_summary.csv", index_col="candidate")
    if not (summary.folds == len(folds)).all():
        raise ValueError("Every candidate must complete every fold before final selection.")
    x, y, weights, audit = training_sample(
        entries, "2016-01-01", "2024-01-01", features, config["max_train_negatives"], config["seed"]
    )
    models, thresholds = {}, {}
    all_episodes = load_episodes(entries, "2016-01-01", "2024-01-01")
    # Keep the best development configuration per model family for final refitting.
    chosen = []
    for kind in dict.fromkeys(c["kind"] for c in candidates):
        family = [c for c in candidates if c["kind"] == kind]
        best = max(family, key=lambda c: summary.loc[c["name"], "mean_AP"])
        chosen.append(best)
    for candidate in chosen:
        oof_paths = [
            cv_output / "oof" / candidate["name"] / f"{fold['name']}.parquet" for fold in folds
        ]
        oof = pd.concat([pd.read_parquet(path) for path in oof_paths], ignore_index=True)
        thresholds[candidate["name"]] = (
            0.5
            if candidate["kind"] == "constant"
            else choose_threshold(
                oof,
                all_episodes,
                config["threshold_policy"],
                config["false_warning_budget"],
                config.get("max_warning_fraction", 0.05),
            )
        )
        print("Final development refit:", candidate["name"], flush=True)
        models[candidate["name"]] = fit_model(candidate, x, y, weights, audit, config["seed"])
    trained = [
        candidate["name"] for candidate in chosen if candidate["kind"] not in {"constant", "random"}
    ]
    champion = summary.loc[trained, "mean_AP"].idxmax()
    bundle = {
        "frozen": True,
        "version": VERSION,
        "config": config,
        "models": models,
        "thresholds": thresholds,
        "champion": champion,
        "features": features,
        "sampling_audit": audit,
        "development_sources": [e["signature"] for e in entries],
    }
    joblib.dump(bundle, output / "frozen_models.joblib")
    manifest = {
        "champion": champion,
        "thresholds": thresholds,
        "config": config,
        "threshold_source": "2021-2023 out-of-fold development predictions",
        "note": "Thresholds frozen before test; score scale can change after final refit.",
        "development_sources": bundle["development_sources"],
        "sampling_audit": audit,
    }
    (output / "frozen_configuration.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return bundle


def reserved_test(bundle, source_root, cache_root, output, enabled=False, allow_repeat=False):
    """No reserved-year discovery, preprocessing or scoring unless explicitly enabled."""
    if not enabled:
        print("Reserved test disabled: 2024 and 2025 were not opened or scored.")
        return None
    if not bundle.get("frozen"):
        raise ValueError("Freeze the final development models and thresholds first.")
    output = Path(output)
    if (output / "metrics.csv").exists() and not allow_repeat:
        raise FileExistsError(
            "Reserved results already exist. Do not repeatedly tune on them; "
            "set allow_repeat only for an intentional rerun."
        )
    config = bundle["config"]
    paths, absent = discover(source_root, [2024, 2025])
    if absent:
        raise FileNotFoundError(f"Missing reserved years: {absent}")
    trained_turbines = set(bundle["sampling_audit"]["turbines"])
    paths = [p for p in paths if p.parent.name.removeprefix("Station_") in trained_turbines]
    present = {(int(p.parent.parent.name), p.parent.name.removeprefix("Station_")) for p in paths}
    missing_pairs = [
        (year, turbine)
        for year in [2024, 2025]
        for turbine in sorted(trained_turbines)
        if (year, turbine) not in present
    ]
    if missing_pairs:
        raise FileNotFoundError(f"Reserved data missing for trained turbine-years: {missing_pairs}")
    sensors = config["sensors"]
    entries = prepare(paths, cache_root, config["alarm_codes"], sensors)
    episodes = load_episodes(entries, "2024-01-01", "2026-01-01")
    results, per_turbine = [], []
    output.mkdir(parents=True, exist_ok=True)
    for name, model in bundle["models"].items():
        predictions = score_interval(model, entries, "2024-01-01", "2026-01-01")
        threshold = bundle["thresholds"][name]
        results.append({"candidate": name, **metrics(predictions, episodes, threshold)})
        turbine = per_turbine_metrics(predictions, episodes, threshold)
        turbine["candidate"] = name
        per_turbine.append(turbine)
        predictions.to_parquet(output / f"predictions_{name}.parquet", index=False)
    table = pd.DataFrame(results)
    table.to_csv(output / "metrics.csv", index=False)
    pd.concat(per_turbine, ignore_index=True).to_csv(
        output / "per_turbine_metrics.csv", index=False
    )
    (output / "test_protocol.json").write_text(
        json.dumps(
            {
                "years": [2024, 2025],
                "champion_fixed_before_test": bundle["champion"],
                "thresholds_not_retuned": bundle["thresholds"],
                "config": config,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return table
