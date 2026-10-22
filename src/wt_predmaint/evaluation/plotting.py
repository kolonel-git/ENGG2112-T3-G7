"""Timeline plots that show why a day is blank (D-019, D-021).

A score or feature line has gaps for different reasons: the day was dropped from labelling (inside
an episode, in the 30-60 day zone, purged at a split boundary, censored, poorly observed), there is
no data at all (for example 2019), or the value itself is missing. Colour bands under the line name
the reason for every day, so a gap is never ambiguous. Scoring does not need a label: a risk score
can be drawn for dropped days, which are only left out of the metrics.
"""

from __future__ import annotations

from collections.abc import Hashable, Iterable

import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

NO_DATA = "no_data"
STATUS_ORDER = [
    "negative",
    "positive",
    "in_episode",
    "zone",
    "purged",
    "censored",
    "low_coverage",
    NO_DATA,
]
STATUS_COLOURS = {
    "negative": "#9ecae1",
    "positive": "#fd8d3c",
    "in_episode": "#a50f15",
    "zone": "#fee08b",
    "purged": "#969696",
    "censored": "#d9d9d9",
    "low_coverage": "#9e9ac8",
    NO_DATA: "#ffffff",
}
STATUS_MEANING = {
    "negative": "label 0 (used)",
    "positive": "label 1 (used)",
    "in_episode": "already in a fault episode",
    "zone": "fault starts in days 31-60",
    "purged": "horizon reaches another split",
    "censored": "horizon past the end of data",
    "low_coverage": "next 60 days poorly observed",
    NO_DATA: "no row (no data that day)",
}
SPLIT_COLOURS = {"train": "#74c476", "val": "#6baed6", "test": "#fd8d3c", "none": "#d9d9d9"}


def calendar_frame(
    values: pd.DataFrame,
    labels: pd.DataFrame,
    station: int,
    value_col: str,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """One turbine on a gap-free daily calendar: value, status and split for every day.

    ``values`` has StationId, Day and ``value_col`` (a risk score or a feature). ``labels`` is a
    label table from ``labels.horizon.build_horizon_labels``. Days with no label row get status
    ``no_data`` and split ``none``; days without a value keep NaN.
    """
    v = values.loc[values["StationId"] == station].set_index("Day")[value_col]
    lab = labels.loc[labels["StationId"] == station].set_index("Day")
    days = v.index.union(lab.index)
    if len(days) == 0:
        raise ValueError(f"no data for station {station}")
    lo = pd.Timestamp(start) if start is not None else days.min()
    hi = pd.Timestamp(end) if end is not None else days.max()
    calendar = pd.date_range(lo, hi, freq="D", name="Day")
    out = pd.DataFrame(index=calendar)
    out["value"] = v.reindex(calendar).astype(float)
    out["status"] = lab["status"].astype(str).reindex(calendar).fillna(NO_DATA)
    out["split"] = lab["split"].astype(str).reindex(calendar).fillna("none")
    return out


def _runs(series: pd.Series) -> list[tuple[Hashable, pd.Timestamp, int]]:
    """Run-length encode a daily series into (value, first day, length in days)."""
    change = (series != series.shift()).cumsum()
    runs = []
    for _, block in series.groupby(change):
        runs.append((block.iloc[0], block.index[0], len(block)))
    return runs


def _band(ax: Axes, runs: list[tuple[Hashable, pd.Timestamp, int]], colours: dict) -> None:
    for key, first, length in runs:
        colour = colours.get(key)
        if colour is None:
            continue
        ax.broken_barh(
            [(mdates.date2num(first), length)],
            (0, 1),
            facecolors=colour,
            edgecolors="#cccccc" if colour == "#ffffff" else "none",
            linewidth=0.3,
        )
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)


def plot_risk_timeline(
    cal: pd.DataFrame,
    episodes: pd.DataFrame | None = None,
    station: int | None = None,
    ylabel: str = "value",
    title: str | None = None,
    threshold: float | None = None,
) -> Figure:
    """Line plot with bands: why each day is blank, the split, and where the value is missing.

    ``cal`` comes from :func:`calendar_frame`. Episode starts, if given, are drawn as red lines.
    The line is broken wherever the value is missing; the bands say why.
    """
    fig, axes = plt.subplots(
        4,
        1,
        figsize=(12, 7),
        sharex=True,
        gridspec_kw={"height_ratios": [6, 0.6, 0.35, 0.35], "hspace": 0.12},
    )
    top, status_ax, split_ax, miss_ax = axes
    top.plot(cal.index, cal["value"], color="#08519c", linewidth=1.0)
    top.set_ylabel(ylabel)
    if threshold is not None:
        top.axhline(threshold, color="#636363", linestyle="--", linewidth=0.8, label="threshold")
    if episodes is not None and station is not None:
        starts = episodes.loc[episodes["StationId"] == station, "Start"]
        for s in starts[(starts >= cal.index.min()) & (starts <= cal.index.max())]:
            top.axvline(s, color="#cb181d", alpha=0.35, linewidth=0.8)
    top.spines[["top", "right"]].set_visible(False)
    if title:
        top.set_title(title, loc="left", fontsize=11)

    _band(status_ax, _runs(cal["status"]), STATUS_COLOURS)
    _band(split_ax, _runs(cal["split"]), SPLIT_COLOURS)
    _band(
        miss_ax,
        _runs(cal["value"].isna().map({True: "missing", False: "ok"})),
        {"missing": "#252525"},
    )
    for ax, name in ((status_ax, "day status"), (split_ax, "split"), (miss_ax, "value NaN")):
        ax.set_ylabel(name, rotation=0, ha="right", va="center", fontsize=8)
    miss_ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    miss_ax.xaxis.set_major_formatter(
        mdates.ConciseDateFormatter(miss_ax.xaxis.get_major_locator())
    )

    counts = cal["status"].value_counts()
    handles = [
        mpatches.Patch(
            facecolor=STATUS_COLOURS[s],
            edgecolor="#cccccc",
            label=f"{s}: {STATUS_MEANING[s]} ({int(counts.get(s, 0))} d)",
        )
        for s in STATUS_ORDER
        if counts.get(s, 0) > 0
    ]
    handles.append(mpatches.Patch(facecolor="#252525", label="value missing (line is broken)"))
    fig.legend(
        handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=2, fontsize=8,
        frameon=False,
    )  # fmt: skip
    fig.subplots_adjust(left=0.1, right=0.98, top=0.93, bottom=0.2)
    return fig


def status_summary(cal: pd.DataFrame) -> pd.DataFrame:
    """Days per status and per split for one calendar; handy next to the figure."""
    table = cal.groupby(["split", "status"]).size().unstack(fill_value=0)
    cols: Iterable[str] = [s for s in STATUS_ORDER if s in table.columns]
    return table[list(cols)].assign(total=table.sum(axis=1))


def blank_reasons(cal: pd.DataFrame) -> pd.Series:
    """Why the line is blank: counts of days with NaN value, by status."""
    return cal.loc[cal["value"].isna(), "status"].value_counts().reindex(STATUS_ORDER).dropna()


__all__ = [
    "NO_DATA",
    "STATUS_COLOURS",
    "STATUS_ORDER",
    "blank_reasons",
    "calendar_frame",
    "plot_risk_timeline",
    "status_summary",
]
