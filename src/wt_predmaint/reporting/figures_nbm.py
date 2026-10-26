"""Report figures 32 and 33: normal-behaviour models and the O1 indicator (D-054, D-025)."""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from wt_predmaint.reporting.context import Ctx
from wt_predmaint.reporting.style import BLUE, MUTED, ORANGE, TARGET_COLOUR, figure

NOTE = (
    "Validation years (2021 to 2022) only; models are fitted on healthy train days (2016 to 2020)."
)


@figure(
    "F32",
    "06_nbm",
    "nbm_validation_shift",
    "NBM: fit quality and validation-year shift per sensor",
    "O1 degradation indicator",
    "Train R-squared is high for the generator windings and low for hydraulic oil temperature. On "
    "healthy-looking validation days the standardised residual has a non-zero mean for several "
    "sensors, so part of any indicator signal is drift between the train and validation years.",
    "reports/tables/nbm_fit.csv (models/nbm.py, scripts/build_nbm.py)",
    NOTE + " Mean z of 0 and sd of 1 would mean validation days look exactly like the training "
    "days; days are negative-labelled and flag-free.",
)
def nbm_validation_shift(ctx: Ctx) -> Figure:
    t = ctx.table("nbm_fit").sort_values("r2_train")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    y = range(len(t))
    axes[0].barh(y, t["r2_train"], color=BLUE)
    axes[0].set_xlabel("train R-squared (healthy days)")
    axes[0].set_yticks(list(y), t["sensor"])
    axes[1].barh(y, t["val_mean_z"], color=ORANGE)
    axes[1].axvline(0, color=MUTED, lw=1)
    axes[1].set_xlabel("mean standardised residual, validation (predicted - observed)")
    fig.tight_layout()
    return fig


@figure(
    "F33",
    "06_nbm",
    "nbm_indicator_vs_lead",
    "NBM indicator against time to the next episode",
    "O1 degradation indicator",
    "The mean indicator is modestly higher in each of the four windows before an episode than "
    "far from one, but it does not rise as the episode nears (15 to 30 days is highest) and the "
    "median hardly moves, so the indicator is a weak early-warning signal on its own.",
    "reports/tables/nbm_lead.csv (models/nbm.py, scripts/build_nbm.py)",
    NOTE + " Indicator is the largest absolute 7-day smoothed standardised residual over sensors; "
    "in-episode days are excluded; turbine-day counts are in the CSV.",
)
def nbm_indicator_vs_lead(ctx: Ctx) -> Figure:
    t = ctx.table("nbm_lead")
    order = list(dict.fromkeys(t["time_to_next_episode"]))
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    width = 0.38
    for i, (target, g) in enumerate(t.groupby("target")):
        g = g.set_index("time_to_next_episode").reindex(order)
        x = [k + (i - 0.5) * width for k in range(len(order))]
        ax.bar(x, g["mean_indicator"], width, color=TARGET_COLOUR[target.upper()],
               label=f"target {target.upper()} mean")  # fmt: skip
        ax.scatter(x, g["median_indicator"], color="black", s=14, zorder=3,
                   label="median" if i == 0 else None)  # fmt: skip
    ax.set_xticks(range(len(order)), order, fontsize=8)
    ax.set_ylabel("NBM indicator (z)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig
