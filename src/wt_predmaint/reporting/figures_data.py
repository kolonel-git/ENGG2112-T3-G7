"""Report figures 01 to 14: data, labels and features (D-025)."""

from __future__ import annotations

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from wt_predmaint.evaluation.plotting import STATUS_COLOURS, STATUS_ORDER
from wt_predmaint.features.daily import feature_columns
from wt_predmaint.reporting.context import Ctx
from wt_predmaint.reporting.style import (
    BLUE,
    GREEN,
    INK,
    INK_2,
    MAGENTA,
    MUTED,
    ORANGE,
    SEQUENTIAL,
    TARGET_COLOUR,
    VIOLET,
    YELLOW,
    figure,
)

D = np.timedelta64(1, "D")
TIER_COLOUR = {1: VIOLET, 2: MAGENTA, 3: GREEN}
STATUS_LABEL = {
    "negative": "negative (label 0, used)",
    "positive": "positive (label 1, used)",
    "in_episode": "inside an episode",
    "zone": "fault in days 31 to 60",
    "purged": "horizon crosses a split",
    "censored": "past end of data",
    "low_coverage": "poorly observed",
}


def _used_rate(ctx: Ctx, target: str) -> float:
    lab = ctx.labels(target)
    used = lab.loc[lab["label"].notna() & (lab["split"] != "none")]
    return float((used["label"] == 1).mean())


@figure(
    "F01",
    "01_data",
    "coverage_turbine_month",
    "Data coverage by turbine and month",
    "Data and cleaning",
    "Turbines are almost fully observed; the exceptions are 2019, which has no clean temperature and "
    "pressure data, and a few months when many turbines are partial at once (for example mid-2022 and "
    "September 2025).",
    "data/interim/clean/scada_*.parquet; reporting/context.py coverage",
    "Share of the full 10-minute grid after the three SCADA tables are joined; months below 0.9 are "
    "flagged as outage months (D-015).",
)
def coverage_turbine_month(ctx: Ctx) -> Figure:
    cov = ctx.coverage
    months = pd.period_range("2016-01", "2026-04", freq="M")
    grid = cov.pivot(index="StationId", columns="month", values="share").reindex(columns=months)
    fig, ax = plt.subplots(figsize=(11, 4.6))
    cmap = SEQUENTIAL.copy()
    cmap.set_bad("#d9d9d6")
    im = ax.imshow(
        grid.to_numpy(), aspect="auto", cmap=cmap, vmin=0, vmax=1, interpolation="nearest"
    )
    ax.set_yticks(range(len(grid)), [str(s)[-2:] for s in grid.index])
    ax.set_ylabel("turbine (last two digits of the id, 23045xx)")
    ticks = [i for i, m in enumerate(months) if m.month == 1]
    ax.set_xticks(ticks, [str(months[i].year) for i in ticks])
    ax.grid(False)
    ax.set_title("Most turbine-months are complete; 2019 is missing and some months are partial")
    cb = fig.colorbar(im, ax=ax, pad=0.015, fraction=0.025)
    cb.set_label("share of 10-minute grid observed")
    ax.text(
        0.995,
        -0.16,
        "grey = no clean data for that turbine and month",
        transform=ax.transAxes,
        ha="right",
        fontsize=8,
        color=INK_2,
    )
    return fig


@figure(
    "F02",
    "01_data",
    "cleaning_by_year",
    "What the cleaning layer touched, by year",
    "Data and cleaning",
    "Cleaning flags and masks a small share of rows and never drops months: at most 3.5% of rows have "
    "a masked value (peak in 2018), and flagged outage months cover 2022 and 2025 only.",
    "reports/tables/cleaning_report.csv (scripts/build_clean_data.py)",
    "Exact duplicates were removed; flags are kept in the data (D-014, D-015).",
)
def cleaning_by_year(ctx: Ctx) -> Figure:
    t = ctx.table("cleaning_report")
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    panels = [
        ("rows", t["rows"] / 1e6, "rows after joining (millions)"),
        (
            "masked",
            100 * t["rows_with_masked_cells"] / t["rows"],
            "rows with a masked sensor value (%)",
        ),
        ("outage", 100 * t["outage_month_rows"] / t["rows"], "rows in flagged outage months (%)"),
    ]
    for ax, (_, values, title) in zip(axes, panels, strict=True):
        ax.bar(t["year"].astype(str), values, color=BLUE, width=0.62)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=60)
    fig.suptitle(
        "Cleaning layer by year (2019 has no clean data)",
        x=0.01,
        ha="left",
        fontsize=11,
        fontweight="bold",
    )
    fig.tight_layout()
    return fig


@figure(
    "F03",
    "01_data",
    "alarm_window_rates",
    "Why alarm codes are not the prediction target",
    "Data and cleaning, target choice",
    "Several alarm codes fire in almost every 30-day window, so predicting them would be trivial; the "
    "fault episodes (A, B) are positive in a more balanced share of windows.",
    "data/interim/alarm_events.parquet, data/interim/labels/*_labels.parquet",
    "Share of turbine-days with at least one event of the code in the next 30 days, over all candidate "
    "days of the three splits (no exclusions applied); the target rates are over labelled rows.",
)
def alarm_window_rates(ctx: Ctx) -> Figure:
    window = int(ctx.cfg["labels"]["positive_window_days"])
    meta = {int(c["code"]): c for c in ctx.cfg["labels"]["alarm_codes"]}
    lab = ctx.labels("a")
    days = lab.loc[lab["split"] != "none", ["StationId", "Day"]]
    ev = ctx.alarms.copy()
    ev["Day"] = (ev["TimeStamp"] - pd.Timedelta(minutes=10)).dt.floor("D")
    rows = []
    for code, info in meta.items():
        hit = []
        for station, g in days.groupby("StationId"):
            d = g["Day"].to_numpy(dtype="datetime64[D]")
            e = np.sort(
                ev.loc[(ev["Alarmcode"] == code) & (ev["StationId"] == station), "Day"].to_numpy(
                    dtype="datetime64[D]"
                )
            )
            lo = np.searchsorted(e, d, side="right")
            hi = np.searchsorted(e, d + window * D, side="right")
            hit.append(hi > lo)
        rows.append((code, int(info["tier"]), float(np.concatenate(hit).mean())))
    df = pd.DataFrame(rows, columns=["code", "tier", "rate"]).sort_values("rate")
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.barh(
        [f"{c} (tier {t})" for c, t in zip(df["code"], df["tier"], strict=True)],
        100 * df["rate"],
        color=[TIER_COLOUR[t] for t in df["tier"]],
        height=0.62,
    )
    for y, r in enumerate(df["rate"]):
        ax.text(100 * r + 1, y, f"{100 * r:.1f}%", va="center", fontsize=7.5, color=INK_2)
    for target, ls in (("A", "-"), ("B", "--")):
        ax.axvline(100 * _used_rate(ctx, target), color=TARGET_COLOUR[target], ls=ls, lw=1.4)
        ax.text(
            100 * _used_rate(ctx, target),
            len(df) - 0.2,
            f"target {target}  " if target == "A" else f"  target {target}",
            color=INK_2,
            fontsize=8,
            va="bottom",
            ha="right" if target == "A" else "left",
        )
    ax.set_xlim(0, 112)
    ax.set_xlabel(f"turbine-days with the event in the next {window} days (%)")
    ax.set_title("Routine and seasonal alarm codes are positive far more often than fault episodes")
    ax.legend(
        handles=[mpatches.Patch(color=TIER_COLOUR[t], label=f"tier {t}") for t in (1, 2, 3)],
        loc="lower right",
    )
    return fig


@figure(
    "F04",
    "01_data",
    "alarm_schedule_and_season",
    "Tier 1 alarms behave like routine operation",
    "Data and cleaning, target choice",
    "Pitch-lubrication alarm 3130 recurs on a near-fixed schedule, and cable-untwisting alarm 10105 "
    "follows the season, so neither signals component wear.",
    "data/interim/alarm_events.parquet",
    "Documented codes only; gaps are between consecutive 3130 events of one turbine.",
)
def alarm_schedule_and_season(ctx: Ctx) -> Figure:
    ev = ctx.alarms
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.8))
    e = ev.loc[ev["Alarmcode"] == 3130].sort_values(["StationId", "TimeStamp"])
    gaps = e.groupby("StationId")["TimeStamp"].diff().dt.total_seconds().dropna() / 86400
    a1.hist(gaps.clip(upper=14), bins=np.arange(0, 14.25, 0.25), color=BLUE)
    med = float(gaps.median())
    a1.axvline(med, color=INK, lw=1.2)
    a1.text(med + 0.2, a1.get_ylim()[1] * 0.92, f"median {med:.1f} days", fontsize=8, color=INK_2)
    a1.set_xlabel("days between consecutive 3130 events (capped at 14)")
    a1.set_ylabel("events")
    a1.set_title("Code 3130: near-fixed schedule")
    m = ev.loc[ev["Alarmcode"] == 10105, "TimeStamp"].dt.month.value_counts().sort_index()
    a2.bar(m.index, 100 * m / m.sum(), color=BLUE, width=0.7)
    a2.set_xticks(range(1, 13), list("JFMAMJJASOND"))
    a2.set_xlabel("calendar month")
    a2.set_ylabel("share of 10105 events (%)")
    a2.set_title("Code 10105: seasonal")
    fig.tight_layout()
    return fig


@figure(
    "F05",
    "02_labels",
    "timeerr_distribution",
    "Daily fault time and the 4-hour fault-day threshold",
    "Targets and labels",
    "Most days have no fault time; a small tail passes the 4-hour threshold that defines a target A "
    "fault day.",
    "data/interim/clean/daily_summary.parquet (tblDailySummary.TimeErr)",
    "Usable days only (not partial, no impossible hours). TimeErr is undocumented by the provider; "
    "its meaning is inferred (D-007, D-008).",
)
def timeerr_distribution(ctx: Ctx) -> Figure:
    d = ctx.daily
    d = d.loc[~(d["partial_day"] | d["bad_hours"])]
    thr = float(ctx.cfg["labels"]["targets"]["daily_fault_time"]["fault_day_min_hours"])
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.hist(d["TimeErr"].clip(upper=24), bins=np.arange(0, 24.5, 0.5), color=BLUE)
    ax.set_yscale("log")
    ax.axvline(thr, color=INK, lw=1.2)
    share = 100 * float((d["TimeErr"] >= thr).mean())
    ax.text(
        thr + 0.3,
        ax.get_ylim()[1] * 0.5,
        f"fault day: {thr:g} h or more\n{share:.1f}% of usable days",
        fontsize=8,
        color=INK_2,
        va="top",
    )
    ax.set_xlabel("TimeErr per turbine-day (hours)")
    ax.set_ylabel("turbine-days (log scale)")
    ax.set_title("Fault time per day is rare and long-tailed")
    return fig


@figure(
    "F06",
    "02_labels",
    "episodes_per_turbine_year",
    "Fault episodes per turbine and year",
    "Targets and labels",
    "Every turbine has episodes in most years, and turbine 15 (2304515) has more than most. 2019 has "
    "target A episodes (from the daily summary) but none for B, which needs the missing 10-minute data.",
    "data/interim/labels/{a,b}_episodes.parquet",
    "Episodes merge fault days or stop runs less than 7 days apart (D-005, D-007, D-008).",
)
def episodes_per_turbine_year(ctx: Ctx) -> Figure:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4), sharey=True)
    for ax, target in zip(axes, ("a", "b"), strict=True):
        e = ctx.episodes(target).assign(year=lambda x: x["Start"].dt.year)
        grid = e.pivot_table(
            index="StationId", columns="year", values="Start", aggfunc="count", fill_value=0
        ).reindex(columns=range(2016, 2027), fill_value=0)
        im = ax.imshow(grid.to_numpy(), aspect="auto", cmap=SEQUENTIAL, vmin=0)
        ax.set_xticks(range(len(grid.columns)), [str(c) for c in grid.columns], rotation=60)
        ax.set_yticks(range(len(grid)), [str(s)[-2:] for s in grid.index])
        for (i, j), v in np.ndenumerate(grid.to_numpy()):
            ax.text(
                j,
                i,
                str(v),
                ha="center",
                va="center",
                fontsize=6.5,
                color="white" if v > grid.to_numpy().max() * 0.55 else INK_2,
            )
        ax.grid(False)
        ax.set_title(f"Target {target.upper()}: {len(e)} episodes")
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="episodes starting")
    axes[0].set_ylabel("turbine (last two digits, 23045xx)")
    fig.tight_layout()
    return fig


@figure(
    "F07",
    "02_labels",
    "target_agreement",
    "Do the two fault definitions find the same events?",
    "Targets and labels",
    "Most episodes found by one definition have a counterpart in the other, so A and B are not "
    "independent evidence.",
    "data/interim/labels/{a,b}_episodes.parquet",
    "Episode-level overlap: an episode matches if the other target has an episode starting within 7 "
    "days (the merge gap). Day-level agreement (F1 about 0.8, D-008) is a different measure.",
)
def target_agreement(ctx: Ctx) -> Figure:
    tol = int(ctx.cfg["labels"]["targets"]["daily_fault_time"]["episode_gap_days"]) * D
    ea, eb = ctx.episodes("a"), ctx.episodes("b")

    def matched(src: pd.DataFrame, other: pd.DataFrame) -> float:
        hits = []
        for station, g in src.groupby("StationId"):
            o = np.sort(
                other.loc[other["StationId"] == station, "Start"].to_numpy(dtype="datetime64[D]")
            )
            s = g["Start"].to_numpy(dtype="datetime64[D]")
            lo = np.searchsorted(o, s - tol, side="left")
            hi = np.searchsorted(o, s + tol, side="right")
            hits.append(hi > lo)
        return float(np.concatenate(hits).mean())

    a_in_b, b_in_a = matched(ea, eb), matched(eb, ea)
    fig, ax = plt.subplots(figsize=(7, 2.9))
    labels = [
        f"target A episodes\nwith a B episode within 7 days (n={len(ea)})",
        f"target B episodes\nwith an A episode within 7 days (n={len(eb)})",
    ]
    ax.barh(
        labels,
        [100 * a_in_b, 100 * b_in_a],
        color=[TARGET_COLOUR["A"], TARGET_COLOUR["B"]],
        height=0.55,
    )
    for y, v in enumerate([a_in_b, b_in_a]):
        ax.text(100 * v + 1, y, f"{100 * v:.0f}%", va="center", fontsize=9, color=INK_2)
    ax.set_xlim(0, 105)
    ax.set_xlabel("share of episodes (%)")
    ax.set_title("A and B largely find the same events")
    ax.invert_yaxis()
    return fig


@figure(
    "F08",
    "02_labels",
    "label_geometry",
    "How a turbine-day is labelled",
    "Targets and labels",
    "A day is positive if a fault episode starts within 30 days; days 31 to 60 are dropped so the "
    "label is not ambiguous; features use only the previous 7 days.",
    "configs/protocol.yaml (labels block); docs/project/DECISIONS.md D-005, D-019",
    "Schematic. Day 0 is the end of the prediction day; all lengths come from the protocol.",
)
def label_geometry(ctx: Ctx) -> Figure:
    lab = ctx.cfg["labels"]
    pos, excl, feat = (
        int(lab["positive_window_days"]),
        int(lab["exclusion_end_days"]),
        int(lab["feature_window_days"]),
    )
    fig, ax = plt.subplots(figsize=(11, 3.2))
    bars = [
        (-feat, feat, BLUE, f"features: the last {feat} days"),
        (0, pos, ORANGE, f"positive window (days 1 to {pos}): episode starts here, label 1"),
        (pos, excl - pos, YELLOW, f"excluded zone (days {pos + 1} to {excl}): row dropped"),
    ]
    for i, (x0, w, c, text) in enumerate(bars):
        y = 2 - i
        ax.barh(y, w, left=x0, height=0.55, color=c)
        ax.text(x0 + w + 0.8, y, text, va="center", fontsize=8.5, color=INK_2)
    ax.axvline(0, ymax=0.74, color=INK, lw=1.4)
    ax.text(
        0,
        2.85,
        "prediction time\n(end of day D)",
        ha="center",
        va="bottom",
        fontsize=8.5,
        color=INK,
    )
    ax.set_xlim(-feat - 3, excl + 48)
    ax.set_ylim(-0.6, 3.5)
    ax.set_yticks([])
    ax.set_xlabel("days from the prediction time")
    ax.grid(False)
    ax.set_title(
        "Label geometry (D-005): a row also drops if it is already in an episode or poorly observed"
    )
    return fig


@figure(
    "F09",
    "02_labels",
    "label_status_by_split",
    "Which days are kept, and why the others are dropped",
    "Targets and labels",
    "Between about two thirds and three quarters of candidate days carry a label; most of the rest "
    "fall in the 31 to 60 day zone, with smaller shares inside episodes or dropped at split boundaries.",
    "reports/tables/label_summary.csv (scripts/build_labels.py)",
    "Shares are of all candidate turbine-days in each split (D-019).",
)
def label_status_by_split(ctx: Ctx) -> Figure:
    t = ctx.table("label_summary")
    fig, axes = plt.subplots(2, 1, figsize=(10, 4.6), sharex=True)
    for ax, target in zip(axes, ("A", "B"), strict=True):
        s = t.loc[t["target"] == target].set_index("split").loc[["train", "val", "test"]]
        left = np.zeros(len(s))
        for status in STATUS_ORDER:
            col = f"n_{status}"
            if col not in s:
                continue
            share = 100 * s[col].to_numpy() / s["candidate_days"].to_numpy()
            ax.barh(
                s.index,
                share,
                left=left,
                color=STATUS_COLOURS[status],
                height=0.62,
                edgecolor=SEQUENTIAL(0.0),
                linewidth=0.8,
            )
            left += share
        ax.invert_yaxis()
        ax.set_ylabel(f"target {target}")
        ax.set_xlim(0, 100)
    axes[1].set_xlabel("share of candidate turbine-days (%)")
    axes[0].set_title("Share of candidate days by label status")
    handles = [
        mpatches.Patch(color=STATUS_COLOURS[s], label=STATUS_LABEL[s])
        for s in STATUS_ORDER
        if s in STATUS_LABEL
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8, bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    return fig


@figure(
    "F10",
    "02_labels",
    "positive_rate_year_turbine",
    "Positive rate by year and by turbine",
    "Targets and labels",
    "The positive rate is high (37% to 50%) and varies by year; one turbine is positive on almost "
    "every validation day, so results are also reported without it.",
    "data/interim/labels/{a,b}_labels.parquet",
    "Labelled rows only. A prevalence-only score therefore reaches a PR-AUC equal to the positive rate "
    "(D-019, D-020).",
)
def positive_rate_year_turbine(ctx: Ctx) -> Figure:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4), gridspec_kw={"width_ratios": [1, 1.5]})
    for target in ("a", "b"):
        lab = ctx.labels(target)
        used = lab.loc[lab["label"].notna() & (lab["split"] != "none")]
        rate = 100 * used.groupby(used["Day"].dt.year)["label"].apply(
            lambda s: float((s == 1).mean())
        )
        rate = rate.reindex(range(2016, 2027))  # leave 2019 (no clean data) as a gap in the line
        a1.plot(
            rate.index,
            rate.values,
            marker="o",
            ms=5,
            color=TARGET_COLOUR[target.upper()],
            label=f"target {target.upper()}",
        )
    a1.set_ylim(0, 100)
    a1.set_ylabel("positive rate (%)")
    a1.set_title("By year")
    a1.legend()
    lab = ctx.labels("a")
    val = lab.loc[lab["label"].notna() & (lab["split"] == "val")]
    per = (
        100
        * val.groupby("StationId")["label"].apply(lambda s: float((s == 1).mean())).sort_values()
    )
    a2.bar([str(s)[-2:] for s in per.index], per.values, color=BLUE, width=0.65)
    top = per.idxmax()
    a2.annotate(
        f"{top}: {per.max():.0f}%",
        (list(per.index).index(top), per.max()),
        xytext=(-70, -4),
        textcoords="offset points",
        fontsize=8,
        color=INK_2,
        arrowprops={"arrowstyle": "-", "color": MUTED},
    )
    a2.set_ylim(0, 105)
    a2.set_xlabel("turbine (last two digits, 23045xx)")
    a2.set_title("Target A, validation years, by turbine")
    fig.tight_layout()
    return fig


@figure(
    "F11",
    "03_features",
    "feature_groups",
    "What the 366 model features are made of",
    "Features",
    "Most features are 7-day means and spreads of daily sensor summaries; alarm counts and fault "
    "history add a small number of columns.",
    "data/interim/features/daily_features.parquet (features/daily.py)",
    "D-021.",
)
def feature_groups(ctx: Ctx) -> Figure:
    cols = feature_columns(ctx.features)
    groups = {
        "7-day std (w7std_)": sum(c.startswith("w7std_") for c in cols),
        "7-day mean (w7mean_)": sum(c.startswith("w7mean_") for c in cols),
        "daily mean while producing (op_)": sum(c.startswith("op_") for c in cols),
        "daily maximum (max_)": sum(c.startswith("max_") for c in cols),
        "alarm counts (alarm_)": sum(c.startswith("alarm_") for c in cols),
        "day state": sum(
            c
            in {"frac_producing", "frac_idle", "frac_stopped", "wind_mean", "wind_std", "rpm_mean"}
            for c in cols
        ),
        "fault history": sum(c in {"days_since_fault_day", "fault_days_30"} for c in cols),
    }
    s = pd.Series(groups).sort_values()
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.barh(s.index, s.values, color=BLUE, height=0.6)
    for y, v in enumerate(s.values):
        ax.text(v + 1.5, y, str(v), va="center", fontsize=8, color=INK_2)
    ax.set_xlabel("columns")
    ax.set_title(f"{len(cols)} model columns by group")
    return fig


@figure(
    "F12",
    "03_features",
    "sensor_availability_by_year",
    "Which sensors exist in which years",
    "Features",
    "Sensors appear and disappear between years, so some features are mostly missing; columns must be "
    "chosen on training years only.",
    "data/interim/features/daily_features.parquet",
    "Share of turbine-days with a value for the daily producing mean; 2019 has no data (D-021).",
)
def sensor_availability_by_year(ctx: Ctx) -> Figure:
    f = ctx.features
    cols = [c for c in feature_columns(f) if c.startswith("op_")]
    year = f["Day"].dt.year
    grid = f[cols].notna().groupby(year).mean().T.reindex(columns=range(2016, 2027))
    grid = grid.loc[grid.mean(axis=1, skipna=True).sort_values().index]
    fig, ax = plt.subplots(figsize=(8, 11))
    im = ax.imshow(grid.to_numpy(), aspect="auto", cmap=SEQUENTIAL, vmin=0, vmax=1)
    ax.set_yticks(
        range(len(grid)),
        [c.removeprefix("op_wtc_").removesuffix("_mean") for c in grid.index],
        fontsize=5.5,
    )
    ax.set_xticks(range(len(grid.columns)), [str(c) for c in grid.columns], rotation=60)
    ax.grid(False)
    ax.set_title("Sensor availability by year (producing-mean features)")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="share of turbine-days with a value")
    return fig


@figure(
    "F13",
    "03_features",
    "operating_state",
    "How the turbines spend their time",
    "Features",
    "Turbines are producing for roughly 40% to 50% of the time (60% in the partial year 2026), and many "
    "days are almost entirely idle or entirely producing, which is why sensor means use producing "
    "intervals only.",
    "data/interim/features/daily_features.parquet",
    "Fractions of 10-minute intervals per turbine-day (D-021).",
)
def operating_state(ctx: Ctx) -> Figure:
    f = ctx.features
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.8))
    a1.hist(f["frac_producing"].dropna(), bins=np.linspace(0, 1, 41), color=BLUE)
    a1.set_xlabel("share of the day producing (rpm at or above the producing threshold)")
    a1.set_ylabel("turbine-days")
    a1.set_title("Producing share per day")
    y = f["Day"].dt.year
    idle_not_stopped = (f["frac_idle"] - f["frac_stopped"]).clip(lower=0)
    parts = pd.DataFrame(
        {
            "producing": f["frac_producing"],
            "idle, low wind": idle_not_stopped,
            "stopped despite wind": f["frac_stopped"],
        }
    )
    parts["low rpm or no data"] = (1 - parts.sum(axis=1)).clip(lower=0)
    mean = 100 * parts.groupby(y).mean()
    left = np.zeros(len(mean))
    colours = [BLUE, "#9cc1ef", ORANGE, "#d9d9d6"]
    for col, c in zip(mean.columns, colours, strict=True):
        a2.bar(
            mean.index.astype(str),
            mean[col],
            bottom=left,
            color=c,
            label=col,
            width=0.68,
            edgecolor=SEQUENTIAL(0.0),
            linewidth=0.8,
        )
        left += mean[col].to_numpy()
    a2.set_ylabel("mean share of the day (%)")
    a2.tick_params(axis="x", rotation=60)
    a2.set_title("Mean operating state by year")
    a2.legend(loc="upper center", ncol=4, bbox_to_anchor=(0.5, -0.2), fontsize=7.5)
    fig.tight_layout()
    return fig


@figure(
    "F14",
    "03_features",
    "healthy_rows_by_year",
    "Healthy training rows by year",
    "Features, normal behaviour",
    "Between about 20% and 45% of training turbine-days qualify as healthy once episodes, a 7-day "
    "recovery period and quality flags are excluded; the share is lowest in 2018.",
    "reports/tables/healthy_rows.csv (scripts/report_healthy_rows.py)",
    "Train years only (D-030); target-specific because the episodes differ.",
)
def healthy_rows_by_year(ctx: Ctx) -> Figure:
    t = ctx.table("healthy_rows")
    t = t.loc[t["is_train_year"]]
    g = t.groupby(["target", "year"])[["healthy_days", "feature_days"]].sum()
    share = (100 * g["healthy_days"] / g["feature_days"]).unstack(0)
    fig, ax = plt.subplots(figsize=(8, 3.6))
    x = np.arange(len(share))
    for k, target in enumerate(("A", "B")):
        ax.bar(
            x + (k - 0.5) * 0.36,
            share[target],
            width=0.34,
            color=TARGET_COLOUR[target],
            label=f"target {target}",
        )
    ax.set_xticks(x, [str(y) for y in share.index])
    ax.set_ylabel("healthy turbine-days (% of feature days)")
    ax.set_title("Healthy rows available for fitting normal behaviour")
    ax.legend()
    return fig
