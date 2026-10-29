"""Report figures 32 and 33: normal-behaviour models and the O1 indicator (D-057, D-055, D-025)."""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from wt_predmaint.reporting.context import Ctx
from wt_predmaint.reporting.style import BLUE, MUTED, ORANGE, TARGET_COLOUR, figure

NOTE = (
    "Validation years (2021 to 2022) only; models are fitted on healthy train days (2016 to 2020)."
)
VARIANTS = [
    ("own_turbine", "own-turbine residual", ORANGE),
    ("fleet_relative", "fleet-relative", BLUE),
]


@figure(
    "F32",
    "06_nbm",
    "nbm_validation_shift",
    "NBM: fit quality and validation-year shift per sensor",
    "O1 degradation indicator",
    "Train R-squared is high for the generator windings and low for hydraulic oil. On healthy-looking "
    "validation days the own-turbine residual has a non-zero mean for most sensors (up to about 0.9 "
    "standard deviations); subtracting the day's fleet median brings every sensor to within about 0.2.",
    "reports/tables/nbm_fit.csv, nbm_fleet_compare.csv (models/nbm.py, scripts/build_nbm.py)",
    NOTE + " Mean z of 0 would mean validation days look like the training days; days are "
    "negative-labelled and flag-free. The fleet-relative version also removes any fleet-wide change "
    "(D-055).",
)
def nbm_validation_shift(ctx: Ctx) -> Figure:
    fit = ctx.table("nbm_fit").sort_values("r2_train")
    cmp = ctx.table("nbm_fleet_compare")
    order = list(fit["sensor"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    y = list(range(len(order)))
    axes[0].barh(y, fit["r2_train"], color=MUTED)
    axes[0].set_xlabel("train R-squared (healthy days)")
    axes[0].set_yticks(y, order)
    height = 0.38
    for i, (variant, label, colour) in enumerate(VARIANTS):
        d = cmp.loc[cmp["variant"] == variant].set_index("sensor").reindex(order)
        axes[1].barh([k + (i - 0.5) * height for k in y], d["val_mean_z"], height, color=colour,
                     label=label)  # fmt: skip
    axes[1].axvline(0, color=MUTED, lw=1)
    axes[1].set_xlabel("mean standardised residual, validation (predicted - observed)")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    return fig


@figure(
    "F33",
    "06_nbm",
    "nbm_indicator_vs_lead",
    "NBM indicator against time to the next episode",
    "O1 degradation indicator",
    "With either residual the indicator is only slightly higher in the windows before an episode than "
    "far from one, it does not rise steadily as the episode nears, and the fleet-relative version "
    "shrinks the contrast further, so the indicator is a weak early-warning signal on its own.",
    "reports/tables/nbm_lead_variants.csv (models/nbm.py, scripts/build_nbm.py)",
    NOTE + " Indicator is the largest absolute 7-day smoothed standardised residual over sensors; "
    "in-episode days are excluded; turbine-day counts are in the CSV. The two panels use different "
    "scales because the fleet-relative residual has a smaller spread.",
)
def nbm_indicator_vs_lead(ctx: Ctx) -> Figure:
    t = ctx.table("nbm_lead_variants")
    order = list(dict.fromkeys(t["time_to_next_episode"]))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    width = 0.38
    for ax, (variant, label, _) in zip(axes, VARIANTS, strict=True):
        v = t.loc[t["variant"] == variant]
        for i, (target, g) in enumerate(v.groupby("target")):
            g = g.set_index("time_to_next_episode").reindex(order)
            x = [k + (i - 0.5) * width for k in range(len(order))]
            ax.bar(x, g["mean_indicator"], width, color=TARGET_COLOUR[target.upper()],
                   label=f"target {target.upper()} mean")  # fmt: skip
            ax.scatter(x, g["median_indicator"], color="black", s=14, zorder=3,
                       label="median" if i == 0 else None)  # fmt: skip
        ax.set_xticks(range(len(order)), order, fontsize=7, rotation=15)
        ax.set_title(label)
        ax.set_ylabel("NBM indicator (z)")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    return fig
