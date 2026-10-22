"""Plot held-out multi-year single-turbine results from saved scores only."""

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

COLORS = {"logistic regression": "#146b9a", "random forest": "#32936f",
          "high sensor rule": "#a3529b", "constant (prevalence)": "#777777",
          "random uniform": "#d37824"}


def save(fig, output, name):
    fig.tight_layout()
    fig.savefig(output / name, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    run = json.loads((args.input_dir / "run.json").read_text(encoding="utf-8"))
    cfg = run["protocol"]
    predictions = pd.read_csv(args.input_dir / "predictions.csv", parse_dates=["Day"])
    detections = pd.read_csv(args.input_dir / "episode_detections.csv", parse_dates=["Start"])
    metrics = pd.read_csv(args.input_dir / "metrics.csv")
    features = pd.read_csv(args.input_dir / "features.csv")
    for target, scores in predictions.groupby("target"):
        summary = metrics.loc[(metrics["target"] == target) & metrics["period"].eq("test_all")]
        fig, axes = plt.subplots(2, 2, figsize=(13, 9))
        for ax, name, title in zip(axes.flat,
                ("pr_auc", "recall", "median_lead_days", "false_alarms_per_turbine_month"),
                ("Average precision", "Episode recall", "Median lead time (days)",
                 "False alarms / turbine-month"), strict=True):
            ax.bar(summary["model"], summary[name], color=[COLORS[m] for m in summary["model"]])
            ax.tick_params(axis="x", labelrotation=25, labelsize=8)
            if name == "pr_auc":
                ax.axhline(summary["prevalence"].iloc[0], color="black", ls=":", label="Prevalence")
                ax.legend(fontsize=8)
            if name == "false_alarms_per_turbine_month":
                ax.axhline(cfg["evaluation"]["false_alarm_budget_per_turbine_month"],
                           color="crimson", ls="--", label="Calibration budget")
                ax.legend(fontsize=8)
            ax.set_title(title)
        fig.suptitle(f"Turbine {run['station']}, target {target}: held-out test years\n"
                     "Point estimates; no between-turbine confidence interval")
        save(fig, output, f"{target}_model_comparison.png")

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        for model, frame in scores.groupby("model"):
            if frame["label"].sum():
                precision, recall, _ = precision_recall_curve(frame["label"], frame["score"])
                axes[0].step(recall, precision, where="post", label=model, color=COLORS[model])
            ep = detections.loc[(detections["target"] == target) & detections["model"].eq(model)]
            ep = ep[["StationId", "Start"]]
            thresholds = np.unique(np.r_[-np.inf, np.quantile(frame["score"], np.linspace(0, 1, 51))])
            points = [evaluate(frame, ep, value, cfg) for value in thresholds]
            axes[1].step([p["false_alarms_per_turbine_month"] for p in points],
                         [p["recall"] for p in points], where="post", label=model, color=COLORS[model])
            chosen = summary.loc[summary["model"].eq(model)].iloc[0]
            axes[1].scatter(chosen["false_alarms_per_turbine_month"], chosen["recall"],
                            color=COLORS[model], marker="x", s=65)
        axes[0].axhline(summary["prevalence"].iloc[0], color="black", ls=":")
        axes[0].set(title="Precision–recall", xlabel="Daily recall", ylabel="Precision")
        axes[1].axvline(cfg["evaluation"]["false_alarm_budget_per_turbine_month"], color="crimson", ls="--")
        axes[1].set(title="Episode recall vs false alarms (× = frozen threshold)",
                    xlabel="False alarms / turbine-month", ylabel="Episode recall")
        axes[0].legend(fontsize=8)
        fig.suptitle(f"Target {target}: test threshold sweeps are descriptive, not tuning")
        save(fig, output, f"{target}_threshold_curves.png")

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ax, model in zip(axes, ("logistic regression", "random forest"), strict=True):
            frame = scores.loc[scores["model"].eq(model)]
            ConfusionMatrixDisplay.from_predictions(frame["label"], frame["alert"].astype(int),
                labels=[0, 1], display_labels=["Negative", "Positive"], colorbar=False, ax=ax)
            ax.set_title(model)
        fig.suptitle(f"Target {target}: daily confusion matrices at frozen thresholds")
        save(fig, output, f"{target}_confusion_matrices.png")

        audit = features.loc[features["target"].eq(target)].copy()
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        for ax, column, title in zip(axes, ("coefficient", "forest_importance"),
                                    ("Logistic standardized weights", "Forest impurity importance"), strict=True):
            used = audit.dropna(subset=[column])
            top = used.loc[used[column].abs().nlargest(15).index].sort_values(column)
            ax.barh(top["feature"], top[column], color="#146b9a")
            ax.set_title(title)
        fig.suptitle(f"Target {target}: training feature weights (not causal explanations)")
        save(fig, output, f"{target}_feature_weights.png")

        for year in cfg["splits"]["test_years"]:
            fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
            for ax, model in zip(axes, ("logistic regression", "random forest"), strict=True):
                frame = scores.loc[scores["model"].eq(model) & scores["Day"].dt.year.eq(year)]
                if frame.empty:
                    ax.set_title(f"{model}: no eligible scored days")
                    continue
                series = frame.set_index("Day")["score"].reindex(pd.date_range(f"{year}-01-01", f"{year}-12-31"))
                ax.plot(series.index, series.values, color=COLORS[model], label="Risk score")
                threshold = frame["threshold"].iloc[0]
                if np.isfinite(threshold):
                    ax.axhline(threshold, ls="--", color="orange", label="Frozen threshold")
                alerts = frame.loc[frame["alert"]]
                ax.scatter(alerts["Day"], alerts["score"], marker="^", color="crimson", s=12, label="Alert")
                ep = detections.loc[(detections["target"] == target) & detections["model"].eq(model)
                                    & detections["evaluable"]]
                for i, start in enumerate(ep.loc[ep["Start"].dt.year.eq(year), "Start"]):
                    ax.axvline(start, color="green", alpha=0.65, label="Episode start" if i == 0 else None)
                ax.set(title=model, ylim=(-0.03, 1.03), ylabel="Score, not calibrated probability")
                ax.legend(fontsize=8)
                ax.set_xlim(pd.Timestamp(f"{year}-01-01"), pd.Timestamp(f"{year}-12-31"))
            fig.suptitle(f"Turbine {run['station']}, target {target}, {year}: eligible test days only")
            save(fig, output, f"{target}_{year}_timeline.png")

        fig, ax = plt.subplots(figsize=(10, 4))
        for model, group in detections.loc[(detections["target"] == target)
                                           & detections["evaluable"] & detections["detected"]].groupby("model"):
            ax.hist(group["lead_days"], bins=np.arange(0.5, 31.6, 2), histtype="step",
                    label=model, color=COLORS[model])
        ax.set(title=f"Target {target}: lead times for detected episodes", xlabel="Days of warning",
               ylabel="Detected episodes")
        if ax.patches:
            ax.legend(fontsize=8)
        save(fig, output, f"{target}_lead_times.png")
    print(f"Saved figures to {output}")


if __name__ == "__main__":
    main()
