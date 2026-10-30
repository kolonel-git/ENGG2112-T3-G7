"""Render the local Target C validation report and precision-recall/confusion chart."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("reports/target_c"))
    args = p.parse_args()
    results = json.loads((args.out / "validation_metrics.json").read_text())
    pred = pd.read_parquet(args.out / "validation_predictions.parquet")
    y = pred.label.astype(int).to_numpy()
    model = results["logistic"]
    tn, fp, fn, tp = np.array(model["confusion_matrix_actual_no_yes_predicted_no_yes"]).ravel()
    negative_accuracy = tn / (tn + fn)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    for title, score, key in [
        ("Logistic model", pred.alarm_score, "logistic"),
        (
            "Previous-hour alarm count",
            pred.recent_alarm_count_1h.fillna(0),
            "recent_alarm_1h_baseline",
        ),
    ]:
        precision, recall, _ = precision_recall_curve(y, score)
        indices = np.unique(np.linspace(0, len(recall) - 1, min(2000, len(recall))).astype(int))
        axes[0].plot(
            recall[indices],
            precision[indices],
            label=f"{title} (AP {results[key]['average_precision']:.3f})",
        )
    axes[0].axhline(model["positive_rate"], color="grey", linestyle="--", label="Constant score")
    axes[0].scatter(
        model["recall"], model["precision"], color="black", zorder=5, label="Model threshold 0.5"
    )
    axes[0].set(
        xlabel="Recall",
        ylabel="Precision",
        title="Validation precision–recall",
        xlim=(0, 1),
        ylim=(0, 1.02),
    )
    axes[0].grid(alpha=0.2)
    axes[0].legend(fontsize=8)
    cm = np.array(model["confusion_matrix_actual_no_yes_predicted_no_yes"])
    axes[1].imshow(cm, cmap="Blues")
    for (i, j), value in np.ndenumerate(cm):
        axes[1].text(
            j,
            i,
            f"{value:,}",
            ha="center",
            va="center",
            color="white" if value > cm.max() / 2 else "black",
            fontsize=13,
        )
    axes[1].set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["No", "Yes"],
        yticklabels=["No", "Yes"],
        xlabel="Predicted",
        ylabel="Actual",
        title="Model confusion matrix (threshold 0.5)",
    )
    fig.suptitle(
        "Target C · alarm within six hours · hourly validation anchors\n"
        "2021–2022 · provisional log coverage",
        fontsize=13,
    )
    fig.savefig(args.out / "validation_summary.png", dpi=180)
    plt.close(fig)
    rows = [
        "# Target C: local validation performance",
        "",
        "**Provisional:** monthly raw alarm exports are assumed complete for represented "
        "turbines. Seven fleet-silent days in May 2021 and overlapping windows were excluded. "
        "Coverage has not been independently certified.",
        "",
        "Prediction question: at each hourly anchor, does any new turbine alarm start in "
        "the next six hours? Sensor history uses all available 10-minute observations.",
        "",
        f"Training: a reproducible uniform sample of **{results['train_sample_rows']:,}** of "
        f"**{results['train_eligible_rows']:,}** eligible anchors from 2016–2018 and 2020. "
        f"There are **{results['feature_count']}** input features. Validation: all "
        f"**{model['rows']:,}** eligible anchors from 2021–2022. Test years were not scored.",
        "",
        "**Reading:** the model shows ranking signal, but its fixed-threshold yes/no decisions "
        "need improvement. Overall accuracy is below always yes. "
        f"Only **{negative_accuracy:.1%}** of no predictions are correct, and it misses "
        f"**{1 - model['recall']:.1%}** of positive prediction windows. "
        "Average precision is a ranking metric, not percent correct.",
        "",
        "| Method | Average precision | Precision | Recall | Accuracy | Balanced accuracy |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, key in [
        ("Logistic (threshold 0.5)", "logistic"),
        ("Alarm in previous hour", "recent_alarm_1h_baseline"),
        ("Always yes", "always_yes"),
        ("Always no", "always_no"),
    ]:
        m = results[key]
        rows.append(
            f"| {name} | {m['average_precision']:.3f} | {m['precision']:.1%} | "
            f"{m['recall']:.1%} | {m['accuracy']:.1%} | {m['balanced_accuracy']:.1%} |"
        )
    lift = model["average_precision"] / model["positive_rate"]
    rows.extend(
        [
            "",
            f"Validation positive prevalence: **{model['positive_rate']:.1%}**. The model's "
            f"average precision is **{lift:.2f}×** the constant-score reference.",
            "",
            "## Performance by validation year",
            "",
            "| Year | Rows | Positive rate | Average precision | Precision | Recall |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for m in results["logistic_by_year"]:
        rows.append(
            f"| {m['year']} | {m['rows']:,} | {m['positive_rate']:.1%} | "
            f"{m['average_precision']:.3f} | {m['precision']:.1%} | {m['recall']:.1%} |"
        )
    rows.extend(
        [
            "",
            "![Validation summary](validation_summary.png)",
            "",
            "## Interpretation limits",
            "",
            "These are hourly prediction-point metrics, not event-level detection rates. "
            "Six-hour outcome windows overlap and many alarms are repeated, operational or "
            "weather related. These scores do not establish mechanical fault prediction.",
            "",
            "The threshold is a fixed 0.5 on a class-weighted logistic score, which is not "
            "a calibrated probability. No threshold was selected on these validation results. "
            "The model was fitted on a training sample, rather than every training anchor.",
            "",
            "Complete predictions are in `validation_predictions.parquet`; "
            "`validation_predictions.csv` contains only the first 1,000 rows. "
            "See `validation_by_station.csv` for turbine-level variation.",
        ]
    )
    (args.out / "PERFORMANCE.md").write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(args.out / "PERFORMANCE.md")


if __name__ == "__main__":
    main()
