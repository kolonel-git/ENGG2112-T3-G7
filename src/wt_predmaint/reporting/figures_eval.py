"""Report figures 15 to 31: evaluation and baselines (D-025). Validation years only, never test."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from sklearn.metrics import precision_recall_curve

from wt_predmaint.evaluation.metrics import episodes_in_split, evaluate, threshold_for_budget
from wt_predmaint.evaluation.plotting import calendar_frame, plot_risk_timeline
from wt_predmaint.evaluation.splits import split_years
from wt_predmaint.reporting.context import Ctx
from wt_predmaint.reporting.style import (
    AQUA,
    BLUE,
    GRID,
    INK_2,
    MAGENTA,
    MUTED,
    ORANGE,
    VIOLET,
    figure,
)

MODELS = [
    ("R1_zscore", "R1 z-score", BLUE),
    ("B1_recent_fault", "B1 recent faults", ORANGE),
    ("B2_alarm_count", "B2 alarm counts", AQUA),
    ("B3_power_curve", "B3 power curve", VIOLET),
    ("B4_logistic", "B4 logistic", MAGENTA),
    ("B0b_random", "random score", MUTED),
]
FLOORS = {"B0a_constant": "constant score", "B0b_random": "random score"}
NAMES = {
    "R1_zscore": "R1 z-score",
    "B1_recent_fault": "B1 recent faults",
    "B1_fault_count": "B1 fault count",
    "B1_fault_recency": "B1 fault recency",
    "B2_alarm_count": "B2 alarm counts",
    "B3_power_curve": "B3 power curve",
    "B4_logistic": "B4 logistic",
    "B0a_constant": "constant score",
    "B0b_random": "random score",
}
VAL_NOTE = "Validation years (2021 to 2022) only; thresholds are chosen on validation. Test years are not scored."


def _frame(ctx: Ctx, target: str, model: str) -> pd.DataFrame:
    """Labelled validation rows with the model's score."""
    lab = ctx.labels(target)
    lab = lab.loc[(lab["split"] == "val") & lab["label"].notna(), ["StationId", "Day", "label"]]
    lab = lab.assign(label=lab["label"].astype(int))
    return lab.merge(
        ctx.scores(target, model)[["StationId", "Day", "score"]],
        on=["StationId", "Day"],
        how="inner",
    )


def _episodes(ctx: Ctx, target: str) -> pd.DataFrame:
    return episodes_in_split(ctx.episodes(target), split_years(ctx.cfg, "val"))


@figure(
    "F15",
    "04_evaluation",
    "recall_vs_false_alarm_budget",
    "Episode recall as the false-alarm budget changes",
    "Evaluation",
    "At the project's budget of 1 false alarm per turbine-month a random score detects about three "
    "quarters of episodes and every baseline is below it; only R1 is above the random line, and only at "
    "the tightest budgets (about 0.25 per turbine-month or less for A, 0.5 or less for B).",
    "data/interim/scores/*_val.parquet, labels, episodes; evaluation/metrics.py",
    VAL_NOTE + " Each point re-chooses the threshold for its budget on the same validation rows. "
    "Real scores alert in streaks on neighbouring days, so the same number of false alarms covers "
    "fewer distinct 30-day windows than scattered random alerts (D-009 uses no cooldown).",
)
def recall_vs_budget(ctx: Ctx) -> Figure:
    ev = ctx.cfg["evaluation"]
    budgets = [0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
    for ax, target in zip(axes, ("a", "b"), strict=True):
        eps = _episodes(ctx, target)
        ends: list[tuple[float, str]] = []
        for model, label, colour in MODELS:
            fr = _frame(ctx, target, model)
            rec = []
            for b in budgets:
                thr = threshold_for_budget(
                    fr["label"].to_numpy(), fr["score"].to_numpy(), b, float(ev["days_per_month"])
                )
                rec.append(evaluate(fr, eps, thr, ctx.cfg)["recall"])
            ls = "--" if model == "B0b_random" else "-"
            ax.plot(
                budgets,
                rec,
                color=colour,
                ls=ls,
                marker="o",
                ms=3.5,
                lw=1.5 if model != "B0b_random" else 1.2,
            )
            ends.append((rec[-1], label))
        placed = -1.0
        for y, text in sorted(ends):
            placed = max(y, placed + 0.045)  # keep the direct labels apart
            ax.annotate(
                text,
                (budgets[-1], y),
                xytext=(budgets[-1] + 0.06, placed),
                fontsize=8,
                color=INK_2,
                va="center",
                arrowprops={"arrowstyle": "-", "color": GRID, "lw": 0.6},
            )
        ax.axvline(float(ev["false_alarm_budget_per_turbine_month"]), color=INK_2, lw=0.9, ls=":")
        ax.set_xlim(0, 2.9)
        ax.set_ylim(0, 1)
        ax.set_xlabel("false alarms per turbine-month allowed")
        ax.set_title(f"Target {target.upper()}")
    axes[0].set_ylabel("share of episodes detected")
    axes[0].text(1.02, 0.02, "project budget", fontsize=7.5, color=INK_2)
    fig.tight_layout()
    return fig


@figure(
    "F16",
    "04_evaluation",
    "precision_recall_curves",
    "Precision-recall curves on validation",
    "Evaluation",
    "R1 and B1 hold precision of about 0.6 to 0.8 at low recall and fall to the prevalence line as "
    "recall grows; B3 and B4 sit at or below it.",
    "data/interim/scores/*_val.parquet",
    VAL_NOTE + " Day-level precision and recall, not episode recall.",
)
def pr_curves(ctx: Ctx) -> Figure:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for ax, target in zip(axes, ("a", "b"), strict=True):
        for model, label, colour in MODELS[:-1]:
            fr = _frame(ctx, target, model)
            p, r, _ = precision_recall_curve(fr["label"], fr["score"])
            ax.plot(r, p, color=colour, lw=1.3, label=label)
        prevalence = float(_frame(ctx, target, "B0b_random")["label"].mean())
        ax.axhline(prevalence, color=MUTED, ls="--", lw=1.2, label=f"prevalence {prevalence:.2f}")
        ax.set_xlabel("recall (days)")
        ax.set_ylim(0.25, 1.02)
        ax.set_title(f"Target {target.upper()}")
    axes[0].set_ylabel("precision (days)")
    axes[0].legend(loc="upper right", fontsize=7.5)
    fig.tight_layout()
    return fig


def _t0(ctx: Ctx, variant: str = "all") -> pd.DataFrame:
    t = ctx.table("t0_validation")
    return t.loc[(t["variant"] == variant) & (t["split"] == "val")]


def _forest(ctx: Ctx, metric: str, floor_col: str, xlabel: str, title: str) -> Figure:
    t = _t0(ctx)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=False)
    for ax, target in zip(axes, ("A", "B"), strict=True):
        d = t.loc[t["target"] == target].sort_values(metric).reset_index(drop=True)
        y = np.arange(len(d))
        for i, r in d.iterrows():
            floor = r["model"] in FLOORS
            ax.plot(
                [r[f"{metric}_lo"], r[f"{metric}_hi"]],
                [i, i],
                color=MUTED if floor else BLUE,
                lw=1.6,
            )
            ax.plot(r[metric], i, "o", color=MUTED if floor else BLUE, ms=6, mec="#fcfcfb", mew=1.2)
        ax.set_yticks(y, [NAMES.get(m, m) for m in d["model"]])
        ax.axvline(float(d[floor_col].iloc[0]), color=INK_2, ls="--", lw=1.0)
        ax.set_xlabel(xlabel)
        ax.set_title(f"Target {target}")
    fig.suptitle(title, x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout()
    return fig


@figure(
    "F17",
    "05_baselines",
    "pr_auc_comparison",
    "PR-AUC of every baseline against the random score",
    "Baselines",
    "R1 has the highest PR-AUC, followed by the B1 rules and B2; B3 and B4 are at or below the random "
    "score. Every interval overlaps the random score's, so differences are small relative to the noise.",
    "reports/tables/t0_validation.csv (scripts/make_baseline_table.py)",
    VAL_NOTE
    + " Point = estimate, line = 95% turbine bootstrap interval; the dashed line is the random "
    "score. B2's variant was chosen on these data, so its value is optimistic.",
)
def pr_auc_comparison(ctx: Ctx) -> Figure:
    return _forest(
        ctx,
        "pr_auc",
        "random_floor_pr_auc",
        "PR-AUC (validation)",
        "PR-AUC with turbine-bootstrap intervals",
    )


@figure(
    "F18",
    "05_baselines",
    "recall_comparison",
    "Episode recall at the false-alarm budget",
    "Baselines",
    "No baseline reaches the random score's recall at 1 false alarm per turbine-month; the random line "
    "is the number to beat.",
    "reports/tables/t0_validation.csv",
    VAL_NOTE + " Recall is the share of evaluable episodes with an alert in the 30 days before.",
)
def recall_comparison(ctx: Ctx) -> Figure:
    return _forest(
        ctx,
        "recall",
        "random_floor_recall",
        "episode recall at 1 false alarm per turbine-month",
        "Episode recall with turbine-bootstrap intervals",
    )


@figure(
    "F19",
    "05_baselines",
    "lead_time",
    "Median warning time before an episode",
    "Baselines",
    "Median warning is roughly two to four weeks for every baseline, but the random score warns about "
    "three weeks ahead too, so lead time alone does not show skill.",
    "reports/tables/t0_validation.csv",
    VAL_NOTE + " Lead time is undefined where nothing is detected.",
)
def lead_time(ctx: Ctx) -> Figure:
    t = _t0(ctx)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, target in zip(axes, ("A", "B"), strict=True):
        d = (
            t.loc[(t["target"] == target) & t["median_lead_days"].notna()]
            .sort_values("median_lead_days")
            .reset_index(drop=True)
        )
        for i, r in d.iterrows():
            floor = r["model"] in FLOORS
            ax.plot(
                [r["median_lead_days_lo"], r["median_lead_days_hi"]],
                [i, i],
                color=MUTED if floor else BLUE,
                lw=1.6,
            )
            ax.plot(
                r["median_lead_days"],
                i,
                "o",
                color=MUTED if floor else BLUE,
                ms=6,
                mec="#fcfcfb",
                mew=1.2,
            )
        ax.set_yticks(range(len(d)), [NAMES.get(m, m) for m in d["model"]])
        ax.set_xlim(0, 30)
        ax.set_xlabel("median days from first alert to the episode start")
        ax.set_title(f"Target {target}")
    fig.tight_layout()
    return fig


@figure(
    "F20",
    "05_baselines",
    "sensitivity_without_2304515",
    "Results with and without turbine 2304515",
    "Baselines",
    "Dropping the outlier turbine lowers every PR-AUC a little but leaves the ranking of baselines "
    "unchanged.",
    "reports/tables/t0_validation.csv",
    VAL_NOTE + " Both variants refit and re-evaluate without the turbine (D-019, D-043).",
)
def sensitivity(ctx: Ctx) -> Figure:
    t = ctx.table("t0_validation")
    t = t.loc[t["split"] == "val"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, target in zip(axes, ("A", "B"), strict=True):
        a = t.loc[(t["target"] == target) & (t["variant"] == "all")].set_index("model")["pr_auc"]
        b = t.loc[(t["target"] == target) & (t["variant"] == "excl_2304515")].set_index("model")[
            "pr_auc"
        ]
        order = a.sort_values().index
        for i, m in enumerate(order):
            ax.plot([b[m], a[m]], [i, i], color=MUTED, lw=1.4)
            ax.plot(
                a[m],
                i,
                "o",
                color=BLUE,
                ms=6,
                mec="#fcfcfb",
                mew=1.2,
                label="all turbines" if i == 0 else None,
            )
            ax.plot(
                b[m],
                i,
                "o",
                color=ORANGE,
                ms=6,
                mec="#fcfcfb",
                mew=1.2,
                label="without 2304515" if i == 0 else None,
            )
        ax.set_yticks(range(len(order)), [NAMES.get(m, m) for m in order])
        ax.set_xlabel("PR-AUC (validation)")
        ax.set_title(f"Target {target}")
    axes[0].legend(loc="lower right")
    fig.tight_layout()
    return fig


@figure(
    "F21",
    "05_baselines",
    "b3_power_curve",
    "B3: the learned healthy wind-speed to rotor-speed curve",
    "Baselines",
    "Healthy turbines follow a smooth rotor-speed curve against wind speed; B3 flags days with a "
    "persistent shortfall below it.",
    "reports/tables/b3_curve_{a,b}.csv (models/power_curve.py)",
    "Pooled over the fleet, fitted on healthy train intervals only (D-031); bins with fewer than 30 "
    "intervals are dropped.",
)
def b3_curve(ctx: Ctx) -> Figure:
    fig, axes = plt.subplots(
        2, 2, figsize=(11, 5.4), sharex=True, gridspec_kw={"height_ratios": [3, 1.2]}
    )
    for col, target in enumerate(("a", "b")):
        c = ctx.table(f"b3_curve_{target}")
        axes[0, col].plot(c["wind"], c["rpm"], marker="o", ms=3.5, color=BLUE)
        axes[0, col].set_title(f"Target {target.upper()} healthy rows")
        axes[1, col].bar(c["wind"], c["count"] / 1e3, width=0.45, color=BLUE)
        axes[1, col].set_xlabel("wind speed (m/s)")
    axes[0, 0].set_ylabel("median generator rpm")
    axes[1, 0].set_ylabel("intervals (thousands)")
    fig.tight_layout()
    return fig


@figure(
    "F22",
    "05_baselines",
    "b4_coefficients",
    "B4: largest standardised logistic coefficients",
    "Baselines",
    "The largest weights sit on temperature, pressure and operating-state window features; they are "
    "correlated associations, not causes.",
    "reports/tables/b4_logistic_coefficients.csv (scripts/tune_b4_logistic.py)",
    "C = 0.01, standardised inputs; correlated features can split or reverse weights (D-042).",
)
def b4_coefficients(ctx: Ctx) -> Figure:
    t = ctx.table("b4_logistic_coefficients")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))
    for ax, target in zip(axes, ("A", "B"), strict=True):
        d = t.loc[t["target"] == target]
        top = pd.concat(
            [
                d.loc[d["direction"] == "positive"].nsmallest(8, "rank"),
                d.loc[d["direction"] == "negative"].nsmallest(8, "rank"),
            ]
        ).sort_values("coefficient")
        names = [
            f.replace("wtc_", "").replace("w7mean_", "mean ").replace("w7std_", "std ")
            for f in top["feature"]
        ]
        ax.barh(
            names,
            top["coefficient"],
            color=[ORANGE if v > 0 else BLUE for v in top["coefficient"]],
            height=0.65,
        )
        ax.axvline(0, color=INK_2, lw=0.8)
        ax.tick_params(axis="y", labelsize=7)
        ax.set_xlabel("standardised coefficient (orange raises risk, blue lowers it)")
        ax.set_title(f"Target {target}")
    fig.tight_layout()
    return fig


@figure(
    "F23",
    "05_baselines",
    "b2_alarm_seasonality",
    "B2: alarm counts move with the season and with a 2022 shift",
    "Baselines",
    "Alarm-count features vary strongly by month and codes 20 and 25 drop sharply in 2022, so B2 partly "
    "tracks season and recording changes.",
    "reports/tables/b2_monthly_counts.csv",
    "Mean trailing 7-day counts over validation feature days (D-051).",
)
def b2_seasonality(ctx: Ctx) -> Figure:
    t = ctx.table("b2_monthly_counts")
    t = t.loc[t["target"] == "a"]
    month = pd.PeriodIndex(t["month"], freq="M").to_timestamp()
    panels = [
        ("alarm_total_w7", "all codes"),
        ("alarm_20_w7", "code 20 (cut-in/out count)"),
        ("alarm_tier2_w7", "tier 2 (weather-driven)"),
        ("alarm_10105_w7", "code 10105 (cable untwisting)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 5.4), sharex=True)
    for ax, (col, title) in zip(axes.ravel(), panels, strict=True):
        ax.plot(month, t[col], color=BLUE, marker="o", ms=3)
        ax.set_title(title)
    axes[1, 0].set_ylabel("mean count in trailing 7 days")
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


@figure(
    "F24",
    "05_baselines",
    "r1_healthy_reference",
    "R1: healthy temperature references per turbine",
    "Baselines",
    "Healthy mean temperatures differ by only about 2 to 6 degrees between turbines, small next to the "
    "day-to-day spread (the bars), so a per-turbine reference adds little over a fleet reference.",
    "reports/tables/r1_parameters.csv (models/zscore.py)",
    "Mean and standard deviation of the daily producing mean over healthy train days (D-052).",
)
def r1_reference(ctx: Ctx) -> Figure:
    t = ctx.table("r1_parameters")
    t = t.loc[(t["variant"] == "all") & (t["target"] == "a")]
    sensors = ["wtc_GeOilTmp_mean", "wtc_GenBeGTm_mean", "wtc_MainBTmp_mean"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), sharey=False)
    for ax, s in zip(axes, sensors, strict=True):
        d = t.loc[t["sensor"] == s].sort_values("StationId")
        ax.errorbar(
            range(len(d)),
            d["mean"],
            yerr=d["std"],
            fmt="o",
            color=BLUE,
            ecolor=MUTED,
            ms=4.5,
            capsize=2,
        )
        ax.set_xticks(range(len(d)), [str(x)[-2:] for x in d["StationId"]], fontsize=7)
        ax.set_title(s.removeprefix("wtc_").removesuffix("_mean"))
        ax.set_xlabel("turbine (23045xx)")
    axes[0].set_ylabel("healthy mean ± 1 sd (°C)")
    fig.tight_layout()
    return fig


@figure(
    "F25",
    "05_baselines",
    "score_separation",
    "Do baseline scores separate positive from negative days?",
    "Baselines",
    "Score distributions of positive and negative days overlap heavily; R1 tilts positives toward "
    "higher scores, the others barely separate them.",
    "data/interim/scores/a_all_*_val.parquet",
    VAL_NOTE
    + " Target A; percentile ranks within each model. The spike at low ranks for R1 and B3 "
    "is the fallback score given to days without usable inputs.",
)
def score_separation(ctx: Ctx) -> Figure:
    models = MODELS[:-1]
    fig, axes = plt.subplots(1, len(models), figsize=(13, 2.9), sharey=True)
    bins = np.linspace(0, 1, 21)
    for ax, (model, label, _) in zip(axes, models, strict=True):
        fr = _frame(ctx, "a", model)
        rank = fr["score"].rank(pct=True)
        ax.hist(
            rank[fr["label"] == 0],
            bins=bins,
            color=BLUE,
            alpha=0.75,
            density=True,
            label="negative",
        )
        ax.hist(
            rank[fr["label"] == 1],
            bins=bins,
            color=ORANGE,
            alpha=0.65,
            density=True,
            label="positive",
        )
        ax.set_title(label, fontsize=9)
        ax.set_xlabel("score percentile")
    axes[0].set_ylabel("density")
    axes[0].legend(fontsize=7.5)
    fig.tight_layout()
    return fig


def _timeline_factory(ctx_model: str, label: str, letter: str) -> None:
    @figure(
        f"F26{letter}",
        "05_baselines",
        f"timeline_{ctx_model}",
        f"Risk timeline for one turbine: {label}",
        "Baselines",
        "Shows the baseline's score over the validation years with episode starts and the bands that "
        "explain every blank day (D-022).",
        f"data/interim/scores/a_all_{ctx_model}_val.parquet, labels, episodes; evaluation/plotting.py",
        "Target A, first turbine in the labels, validation years only. Rule and z-score baselines "
        "give days without usable inputs a fallback score, which shows as drops in the line.",
    )
    def timeline(ctx: Ctx) -> Figure:
        station = int(ctx.labels("a")["StationId"].min())
        scores = ctx.scores("a", ctx_model)
        cal = calendar_frame(scores, ctx.labels("a"), station, "score", "2021-01-01", "2022-12-31")
        return plot_risk_timeline(
            cal,
            ctx.episodes("a"),
            station,
            ylabel=label,
            title=f"Turbine {station}, target A: {label}",
        )


for _i, (_model, _label, _) in enumerate(MODELS[:-1]):
    _timeline_factory(_model, _label, "abcde"[_i])
