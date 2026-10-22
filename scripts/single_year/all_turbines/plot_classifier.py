"""Create PNG plots from saved single-year classifier scores; never refit models."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, precision_recall_curve

from wt_predmaint.evaluation.metrics import evaluate

YEAR = 2018
COLORS = {"logistic regression": "#146b9a", "constant (prevalence)": "#777777",
          "random uniform": "#d37824"}


def save(fig, folder: Path, name: str):
    fig.tight_layout()
    fig.savefig(folder / name, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main(default_station: int | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=YEAR)
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--station", type=int, default=default_station,
                        help="Turbine for single-turbine input, or timeline filter for all-turbine input")
    args = parser.parse_args()
    scope = f"turbine_{args.station}" if default_station is not None else "all_turbines"
    source = args.input_dir or ROOT / f"data/processed/single_year/{args.year}/{scope}"
    manifest = json.loads((source / "run.json").read_text(encoding="utf-8"))
    cfg = manifest["protocol"]
    recorded_scope = manifest.get("scope", "all_turbines")
    dest = args.output_dir or ROOT / f"reports/figures/single_year/{manifest['year']}/{recorded_scope}"
    if default_station is not None and manifest.get("station") != args.station:
        parser.error("The selected run was not trained solely on the requested turbine")
    predictions = pd.read_csv(source / "predictions.csv", parse_dates=["Day"])
    detections = pd.read_csv(source / "episode_detections.csv", parse_dates=["Start"])
    metrics = pd.read_csv(source / "metrics.csv")
    features = pd.read_csv(source / "features.csv")
    if args.station is not None and args.station not in predictions["StationId"].unique():
        parser.error("Requested turbine has no scored test days")
    dest.mkdir(parents=True, exist_ok=True)
    year = manifest["year"]
    for target, target_metrics in metrics.groupby("target"):
        # Summary: same test rows for all three scorers within each fold.
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        for ax, metric, title in zip(axes.flat,
                                    ("pr_auc", "recall", "median_lead_days",
                                     "false_alarms_per_turbine_month"),
                                    ("Average precision", "Episode recall", "Median lead time (days)",
                                     "False alarms / turbine-month"), strict=True):
            for offset, (model, group) in enumerate(target_metrics.groupby("model")):
                x = group["fold"].to_numpy() + (offset - 1) * 0.14
                ax.scatter(x, group[metric], label=model, color=COLORS[model])
                ax.vlines(x, group[f"{metric}_lo"], group[f"{metric}_hi"], color=COLORS[model])
            if metric == "false_alarms_per_turbine_month":
                ax.axhline(cfg["evaluation"]["false_alarm_budget_per_turbine_month"],
                           color="crimson", ls="--", label="Calibration budget")
            if metric == "pr_auc":
                reference = target_metrics.drop_duplicates("fold")
                ax.plot(reference["fold"], reference["prevalence"], "k:", label="Prevalence")
            ax.set(title=title, xlabel="Fold", xticks=sorted(target_metrics["fold"].unique()))
            ax.grid(alpha=0.2)
        axes[0, 0].legend(fontsize=8)
        level = cfg["evaluation"]["ci_level"] * 100
        detail = (f"turbine {manifest['station']}; point estimates, no bootstrap intervals"
                  if manifest.get("station") is not None else f"{level:g}% turbine bootstrap")
        fig.suptitle(f"{year}, target {target}: out-of-sample performance ({detail})")
        save(fig, dest, f"{target}_fold_comparison.png")

        target_scores = predictions.loc[predictions["target"] == target]
        for fold, scores in target_scores.groupby("fold"):
            fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))
            for model, frame in scores.groupby("model"):
                if frame["label"].sum():
                    precision, recall, _ = precision_recall_curve(frame["label"], frame["score"])
                    axes[0].step(recall, precision, where="post", color=COLORS[model], label=model)
                det = detections.loc[(detections["target"] == target)
                                     & (detections["fold"] == fold) & (detections["model"] == model)]
                episodes = det[["StationId", "Start"]]
                thresholds = np.unique(np.r_[-np.inf, np.quantile(frame["score"],
                                                                                np.linspace(0, 1, 41))])
                points = [evaluate(frame, episodes, threshold, cfg) for threshold in thresholds]
                axes[1].step([p["false_alarms_per_turbine_month"] for p in points],
                             [p["recall"] for p in points], where="post",
                             color=COLORS[model], label=model)
                chosen = evaluate(frame, episodes, float(frame["threshold"].iloc[0]), cfg)
                axes[1].scatter(chosen["false_alarms_per_turbine_month"], chosen["recall"],
                                color=COLORS[model], marker="x", s=70)
            axes[0].axhline(scores.drop_duplicates(["StationId", "Day"])["label"].mean(),
                           color="black", ls=":", label="Prevalence")
            axes[0].set(title="Precision–recall", xlabel="Daily recall", ylabel="Precision")
            axes[0].legend(fontsize=7)
            axes[1].axvline(cfg["evaluation"]["false_alarm_budget_per_turbine_month"],
                           color="crimson", ls="--")
            axes[1].set(title="Episode recall vs false alarms (× = frozen threshold)",
                        xlabel="False alarms / turbine-month", ylabel="Episode recall")
            logistic = scores.loc[scores["model"] == "logistic regression"]
            ConfusionMatrixDisplay.from_predictions(
                logistic["label"], logistic["alert"].astype(int), labels=[0, 1],
                display_labels=["Negative", "Positive"], colorbar=False, ax=axes[2], cmap="Blues")
            axes[2].set_title("Logistic: daily confusion matrix")
            support = (" — one test class only" if logistic["label"].nunique() < 2 else "")
            fig.suptitle(f"{year}, target {target}, fold {fold} — test diagnostics{support}")
            save(fig, dest, f"{target}_fold{fold}_diagnostics.png")

            coefficients = features.loc[(features["target"] == target)
                                        & (features["fold"] == fold)].dropna(subset=["coefficient"])
            top = coefficients.loc[coefficients["coefficient"].abs().nlargest(20).index]
            top = top.sort_values("coefficient")
            fig, ax = plt.subplots(figsize=(11, 7))
            ax.barh(top["feature"], top["coefficient"],
                    color=np.where(top["coefficient"] >= 0, "#b95343", "#146b9a"))
            ax.axvline(0, color="black", lw=0.8)
            ax.set(title=f"Target {target}, fold {fold}: top standardized logistic weights",
                   xlabel="Log-odds coefficient per training standard deviation (not causal)")
            save(fig, dest, f"{target}_fold{fold}_feature_weights.png")

        logistic = target_scores.loc[target_scores["model"] == "logistic regression"]
        stations = [args.station] if args.station is not None else sorted(logistic["StationId"].unique())
        for station in stations:
            g = logistic.loc[logistic["StationId"] == station]
            if g.empty:
                continue
            fold_ids = sorted(g["fold"].unique())
            fig, axes = plt.subplots(len(fold_ids), 1, figsize=(12, 3 * len(fold_ids)), squeeze=False)
            for ax, fold in zip(axes.flat, fold_ids, strict=True):
                frame = g.loc[g["fold"] == fold].sort_values("Day")
                # Reindex to expose missing/unscored days as line breaks.
                daily = frame.set_index("Day")["score"].reindex(
                    pd.date_range(frame["Day"].min(), frame["Day"].max()))
                ax.plot(daily.index, daily.values, label="Risk score")
                threshold = frame["threshold"].iloc[0]
                if np.isfinite(threshold):
                    ax.axhline(threshold, ls="--", color="orange", label="Alert threshold")
                alerts = frame.loc[frame["alert"]]
                ax.scatter(alerts["Day"], alerts["score"], color="crimson", marker="^", label="Alert")
                det = detections.loc[(detections["target"] == target)
                                     & (detections["fold"] == fold)
                                     & (detections["model"] == "logistic regression")
                                     & detections["evaluable"]
                                     & (detections["StationId"] == station)]
                for index, start in enumerate(det["Start"]):
                    ax.axvspan(start - pd.Timedelta(days=cfg["labels"]["positive_window_days"]),
                               start, color="green", alpha=0.1,
                               label="Episode warning window" if index == 0 else None)
                    ax.axvline(start, color="green", alpha=0.6,
                               label="Episode start" if index == 0 else None)
                ax.set(title=f"Fold {fold} (only eligible test days scored)",
                       ylabel="Risk score, not calibrated probability", ylim=(-0.03, 1.03))
                ax.legend(fontsize=7, loc="upper left")
            fig.suptitle(f"{year}, target {target}, turbine {station}")
            save(fig, dest, f"{target}_turbine_{station}_timeline.png")
        fig, ax = plt.subplots(figsize=(9, 4))
        for model, g in detections.loc[(detections["target"] == target)
                                       & detections["evaluable"] & detections["detected"]].groupby("model"):
            ax.hist(g["lead_days"], bins=np.arange(0.5, 31.6, 2), histtype="step",
                    label=model, color=COLORS[model])
        ax.set(title=f"Target {target}: lead times of detected episodes across folds",
               xlabel="Days before episode start", ylabel="Detected fold-episode observations")
        if ax.patches:
            ax.legend()
        save(fig, dest, f"{target}_lead_times.png")
    print(f"Saved PNG figures to {dest}")


if __name__ == "__main__":
    main()
