"""Report figures 36 to 39: validation against test, pruning, label sensitivity and ranking.

F36 reads the saved baseline tables, which hold both splits of the single frozen run (D-061).
F37 to F39 are validation-years-only exploratory results (D-066, D-068, D-065); nothing in them
was scored on the test years.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from wt_predmaint.reporting.context import Ctx
from wt_predmaint.reporting.style import (
    AQUA,
    BLUE,
    INK_2,
    MAGENTA,
    MUTED,
    ORANGE,
    TARGET_COLOUR,
    VIOLET,
    figure,
)

NAMES = {
    "B1_recent_fault": "B1 recent fault",
    "B2_alarm_count": "B2 alarm count",
    "R1_zscore": "R1 z-score",
    "B3_power_curve": "B3 power curve",
    "B4_logistic": "B4 logistic",
    "R2_nbm_indicator": "R2 NBM indicator",
    "L1_lgbm": "L1 LightGBM",
    "L1_lgbm_nbm": "L1 LightGBM + NBM",
}
EXPLORATORY = "Validation years (2021 to 2022) only; exploratory; the test years are not used."


@figure(
    "F36",
    "08_exploratory",
    "validation_to_test_shift",
    "From validation to test: no ranking carried over",
    "Results",
    "Models that led on the validation years fall back to the random-score value on the test years "
    "(R1 most clearly), while the learned models gain a little on test; no test interval excludes the "
    "random-score value.",
    "reports/tables/baselines_a.csv, baselines_b.csv (scripts/run_baselines.py, frozen run D-061)",
    "Hollow marker: validation PR-AUC. Filled marker with line: test PR-AUC with its 95% turbine "
    "bootstrap interval. Dashed lines: random-score PR-AUC on validation (grey) and test "
    "(black). The test years were scored once (D-061).",
)
def validation_to_test_shift(ctx: Ctx) -> Figure:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0), sharey=True)
    models = [m for m in NAMES if m != "L1_lgbm_nbm"]
    for ax, target in zip(axes, ("A", "B"), strict=True):
        b = ctx.table(f"baselines_{target.lower()}")
        b = b.loc[b["variant"] == "all"]
        val = b.loc[b["split"] == "val"].set_index("model")
        test = b.loc[b["split"] == "test"].set_index("model")
        y = np.arange(len(models))[::-1]
        for yi, m in zip(y, models, strict=True):
            t = test.loc[m]
            ax.plot([t["pr_auc_lo"], t["pr_auc_hi"]], [yi, yi], color=TARGET_COLOUR[target], lw=1.4)
            ax.scatter(t["pr_auc"], yi, s=34, color=TARGET_COLOUR[target], zorder=3)
            ax.scatter(val.loc[m, "pr_auc"], yi, s=34, facecolor="none",
                       edgecolor=INK_2, zorder=3)  # fmt: skip
            ax.annotate("", xy=(t["pr_auc"], yi), xytext=(val.loc[m, "pr_auc"], yi),
                        arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.8})  # fmt: skip
        ax.axvline(val["random_floor_pr_auc"].iloc[0], color=MUTED, ls="--", lw=1)
        ax.axvline(test["random_floor_pr_auc"].iloc[0], color="black", ls="--", lw=1)
        ax.set_yticks(y, [NAMES[m] for m in models])
        ax.set_xlabel("PR-AUC")
        ax.set_title(f"Target {target}")
    handles = [
        Line2D([], [], marker="o", ls="", markerfacecolor="none", markeredgecolor=INK_2,
               label="validation"),
        Line2D([], [], marker="o", ls="", color=MUTED, label="test (95% interval)"),
    ]  # fmt: skip
    axes[1].legend(handles=handles, loc="upper right", fontsize=8)
    fig.tight_layout()
    return fig


@figure(
    "F37",
    "08_exploratory",
    "feature_pruning",
    "Pruning the features: fewer columns, same score",
    "Exploratory rounds",
    "Keeping 50 or 100 columns costs nothing measurable and keeping 10 or 25 does not help; the "
    "check on training years gets worse for most pruned versions.",
    "reports/tables/exploratory_selection.csv (scripts/run_exploratory.py --round 3, D-066)",
    EXPLORATORY + " Each point is one pruned configuration minus the same model with all columns "
    "(two models, two feature sets, three ranking methods); blue is target A and orange target B. "
    "Right panel: the same difference on the training-year shift check (fit to 2018, score 2020).",
)
def feature_pruning(ctx: Ctx) -> Figure:
    d = ctx.table("exploratory_selection")
    d["method"] = d["method"].fillna("none")
    base = d.loc[d["method"] == "none"].set_index(["target", "feature_set", "model"])
    p = d.loc[~d["method"].isin(["none", "drop_group"])].copy()
    for col, new in (("pr_auc", "delta"), ("year_shift_pr_auc", "shift")):
        p[new] = [
            r[col] - base.loc[(r["target"], r["feature_set"], r["model"]), col]
            for _, r in p.iterrows()
        ]
    keeps = sorted(p["keep"].unique())
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), sharey=False)
    for ax, col, title in zip(
        axes,
        ("delta", "shift"),
        ("Validation PR-AUC", "Training-year shift check (fit to 2018, score 2020)"),
        strict=True,
    ):
        for i, k in enumerate(keeps):
            for target in ("A", "B"):
                v = p.loc[(p["keep"] == k) & (p["target"] == target), col]
                ax.scatter(i + (0.12 if target == "B" else -0.12) + rng.normal(0, 0.03, len(v)), v,
                           s=16, color=TARGET_COLOUR[target], alpha=0.8,
                           label=f"target {target}" if i == 0 else None)  # fmt: skip
        ax.axhline(0, color=MUTED, lw=1)
        ax.set_xticks(range(len(keeps)), [f"top {k}" for k in keeps])
        ax.set_ylabel("PR-AUC change against all columns")
        ax.set_title(title, fontsize=10)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    return fig


@figure(
    "F38",
    "08_exploratory",
    "label_sensitivity",
    "Other label definitions: only simple rules move",
    "Exploratory rounds",
    "As labels get narrower and rarer, the ratio to the random-score floor rises for most models, but "
    "the intervals widen too: almost none of the learned models' (LightGBM, B4) ratios clear their "
    "floor, while the simple rules (the power-curve baseline B3, the recent-fault rule B1) do in many "
    "variants.",
    "reports/tables/label_sensitivity.csv (scripts/run_label_sensitivity.py, D-068)",
    EXPLORATORY
    + " Each variant is a different label definition, ordered from the broadest (left) to "
    "the rarest (right). The value is PR-AUC divided by that variant's own random-score floor, so 1 "
    "means no better than a random score, and a ratio becomes large when the floor is small, so read it "
    "with the markers. A filled marker means the 95% interval lies above the floor.",
)
def label_sensitivity(ctx: Ctx) -> Figure:
    d = ctx.table("label_sensitivity")
    d = d.loc[~d["model"].isin(["B0a_constant", "B0b_random"])].copy()
    d["lift"] = d["pr_auc"] / d["random_floor_pr_auc"]
    d["above"] = d["excludes_floor"] & (d["excess_over_floor"] > 0)
    shown = {
        "B3_power_curve": BLUE,
        "B1_recent_fault": ORANGE,
        "R1_zscore": MAGENTA,
        "L1_lgbm": AQUA,
        "B4_logistic": VIOLET,
    }
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for ax, target in zip(axes, ("A", "B"), strict=True):
        t = d.loc[d["target"] == target]
        order = (
            t[["horizon", "severity", "weather", "prevalence"]]
            .drop_duplicates()
            .sort_values("prevalence", ascending=False)
            .reset_index(drop=True)
        )
        x = {tuple(r[:3]): i for i, r in enumerate(order.itertuples(index=False))}
        for model in NAMES:
            g = t.loc[t["model"] == model]
            xs = [x[(r.horizon, r.severity, r.weather)] for r in g.itertuples()]
            ys = g["lift"].to_numpy()
            colour = shown.get(model, MUTED)
            idx = np.argsort(xs)
            ax.plot(np.array(xs)[idx], ys[idx], color=colour, lw=1.0 if model in shown else 0.6,
                    alpha=0.9 if model in shown else 0.5)  # fmt: skip
            above = g["above"].to_numpy()
            ax.scatter(np.array(xs)[above], ys[above], s=26, color=colour, zorder=3)
            ax.scatter(np.array(xs)[~above], ys[~above], s=26, facecolor="none",
                       edgecolor=colour, zorder=3, lw=0.9)  # fmt: skip
        ax.axhline(1.0, color="black", lw=1, ls="--")
        ax.set_xticks(range(len(order)), [f"{p:.2f}" for p in order["prevalence"]], rotation=90,
                      fontsize=7)  # fmt: skip
        ax.set_xlabel("label variant, by share of positive days (broad to rare)")
        ax.set_ylabel("PR-AUC divided by the variant's random-score floor")
        ax.set_title(f"Target {target}: {len(order)} variants", fontsize=10)
    handles = [Line2D([], [], color=c, label=NAMES[m]) for m, c in shown.items()]
    handles.append(Line2D([], [], color=MUTED, label="other models"))
    axes[0].legend(handles=handles, fontsize=7, loc="upper left")
    fig.tight_layout()
    return fig


@figure(
    "F39",
    "08_exploratory",
    "turbine_ranking",
    "Ranking the turbines each day: the power-curve baseline leads",
    "Exploratory rounds",
    "Among the main models, the power-curve baseline B3 puts positive turbines in the daily top three "
    "more often than chance for both targets; the z-score R1 does worse than chance.",
    "reports/tables/ranking_validation.csv (scripts/run_ranking.py, evaluation/ranking.py, D-065)",
    EXPLORATORY + " Precision among the three highest-scored turbines each day minus the share of "
    "positive turbines that day, with a 95% interval from resampling calendar months. Zero means no "
    "better than chance; 92 intervals were examined across the project, so this is a lead to confirm.",
)
def turbine_ranking(ctx: Ctx) -> Figure:
    r = ctx.table("ranking_validation")
    r = r.loc[r["k"] == 3]
    models = [m for m in r["model"].unique() if m != "L1_lgbm_nbm"]
    labels = {"B0b_random": "Random score", **NAMES}
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), sharey=True)
    for ax, target in zip(axes, ("A", "B"), strict=True):
        t = r.loc[r["target"] == target].set_index("model")
        y = np.arange(len(models))[::-1]
        for yi, m in zip(y, models, strict=True):
            row = t.loc[m]
            ax.plot([row["difference_lo"], row["difference_hi"]], [yi, yi],
                    color=TARGET_COLOUR[target], lw=1.6)  # fmt: skip
            ax.scatter(row["difference"], yi, s=34, color=TARGET_COLOUR[target], zorder=3)
        ax.axvline(0, color="black", lw=1, ls="--")
        ax.set_yticks(y, [labels[m] for m in models])
        ax.set_xlabel("top-3 precision minus chance")
        ax.set_title(f"Target {target}")
    fig.tight_layout()
    return fig
