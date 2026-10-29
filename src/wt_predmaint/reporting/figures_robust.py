"""Report figures 34 and 35: SHAP and robustness of the frozen LightGBM (D-062, D-025).

Validation years only. SHAP is model attribution, not causation.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.patches import Patch

from wt_predmaint.reporting.context import Ctx
from wt_predmaint.reporting.style import BLUE, MUTED, ORANGE, TARGET_COLOUR, figure

NOTE = "Validation years (2021 to 2022) only; frozen L1_lgbm (D-060). Test years are not used."


def _label(feature: str) -> str:
    for prefix in ("w7mean_", "w7std_"):
        if feature.startswith(prefix):
            kind = "7-day mean " if prefix == "w7mean_" else "7-day sd "
            feature = feature.removeprefix(prefix)
            break
    else:
        kind = ""
    feature = feature.replace("op_wtc_", "").replace("max_wtc_", "max ").replace("_mean", "")
    return kind + feature.removesuffix("_max") if " " not in feature else kind + feature


@figure(
    "F34",
    "07_explain_robustness",
    "shap_top_features",
    "SHAP: what the frozen LightGBM relies on",
    "Explainability",
    "The model leans on ambient and weather-linked inputs (DeltaTmp, ambient temperature, tower "
    "humidity and frequency), on pressure sensors (inverter, offline, hub pressure) and on days "
    "since the last fault day; component temperatures such as MainBTmp and HydOilTm rank lower. "
    "The ranking is stable across leave-one-turbine-out refits.",
    "reports/tables/shap_importance_a.csv, shap_importance_b.csv, shap_stability.csv "
    "(explain/shap_lgbm.py, scripts/run_shap.py)",
    NOTE + " Mean absolute SHAP on 4,000 validation rows. Colour shows the direction of the effect "
    "(rank correlation of the feature value with its SHAP value). This describes the model, not "
    "causes of faults, and the model scores at the random-score level (D-060).",
)
def shap_top_features(ctx: Ctx) -> Figure:
    top = int(ctx.cfg["explain"]["top_n"]) - 5
    stab = ctx.table("shap_stability")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))
    for ax, target in zip(axes, ("a", "b"), strict=True):
        t = ctx.table(f"shap_importance_{target}").head(top).iloc[::-1]
        colours = [ORANGE if d > 0 else BLUE for d in t["direction"].fillna(0)]
        ax.barh([_label(f) for f in t["feature"]], t["share"], color=colours)
        s = stab.loc[stab["target"] == target.upper()]
        ax.set_title(
            f"Target {target.upper()}: rank correlation across turbine folds "
            f"{s['spearman_vs_full'].median():.2f} (median)",
            fontsize=10,
        )
        ax.set_xlabel("share of total mean |SHAP|")
        ax.tick_params(axis="y", labelsize=8)
    handles = [
        Patch(color=ORANGE, label="higher value raises score"),
        Patch(color=BLUE, label="higher value lowers score"),
    ]
    axes[0].legend(handles=handles, fontsize=8, loc="lower right")
    fig.tight_layout()
    return fig


@figure(
    "F35",
    "07_explain_robustness",
    "robustness_dropout_loto",
    "Robustness: sensor-group dropout and leave-one-turbine-out",
    "Robustness",
    "Removing a sensor group barely moves PR-AUC (within 0.03), while recall at the fixed threshold "
    "moves only because scores shift and false alarms change; leaving a turbine out of training "
    "leaves pooled PR-AUC at or below the pooled prevalence, and per-turbine PR-AUC follows the turbine's "
    "own prevalence.",
    "reports/tables/robustness_dropout.csv, robustness_loto.csv, robustness_loto_summary.csv "
    "(evaluation/robustness.py, scripts/run_robustness.py)",
    NOTE + " Dropout sets a group's features to missing at scoring time at the intact model's "
    "threshold, so a change in recall comes with a change in false alarms (see the CSV). In the "
    "right panel the diagonal is a score no better than the turbine's prevalence; every turbine "
    "has at least 113 positive validation rows.",
)
def robustness_dropout_loto(ctx: Ctx) -> Figure:
    d = ctx.table("robustness_dropout")
    d = d.loc[d["group"] != "none"]
    loto = ctx.table("robustness_loto")
    groups = list(dict.fromkeys(d["group"]))
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
    width = 0.38
    for i, target in enumerate(("A", "B")):
        g = d.loc[d["target"] == target].set_index("group").reindex(groups)
        x = [k + (i - 0.5) * width for k in range(len(groups))]
        axes[0].bar(
            x, g["pr_auc_change"], width, color=TARGET_COLOUR[target], label=f"target {target}"
        )
    axes[0].axhline(0, color=MUTED, lw=1)
    axes[0].set_xticks(range(len(groups)), [g.replace("_", " ") for g in groups], fontsize=8)
    axes[0].set_ylabel("change in validation PR-AUC")
    axes[0].set_title("Sensor-group dropout (scoring time)", fontsize=10)
    axes[0].legend(fontsize=8)
    for target in ("A", "B"):
        t = loto.loc[loto["target"] == target]
        axes[1].scatter(t["prevalence"], t["pr_auc_left_out"], s=22, color=TARGET_COLOUR[target],
                        label=f"target {target}")  # fmt: skip
    axes[1].plot([0.2, 1], [0.2, 1], color=MUTED, lw=1, ls="--")
    axes[1].set_xlabel("turbine's validation prevalence")
    axes[1].set_ylabel("PR-AUC with the turbine left out of training")
    axes[1].set_title("Leave-one-turbine-out (one point per turbine)", fontsize=10)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    return fig
