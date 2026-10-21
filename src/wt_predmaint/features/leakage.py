"""Look-ahead test for the feature table (D-021).

The row for (station, day D) must be computable from data that exists at the end of D. The check
recomputes the row after deleting everything later than that moment from every source and
compares it with the row from the full data. Any difference means a feature used the future.

What exists at the end of D: SCADA and alarm rows with ``TimeStamp`` up to D+1 00:00 (interval
end), and the ``tblDailySummary`` row stamped D (stamped at the start of the day it covers).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from wt_predmaint.features.daily import KEYS, assemble_features, daily_aggregates

END_OF_DAY = pd.Timedelta(days=1)


def truncate_sources(
    scada: pd.DataFrame, events: pd.DataFrame, summary: pd.DataFrame, day: pd.Timestamp
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Sources as they stood at the end of ``day``."""
    end = day + END_OF_DAY
    return (
        scada.loc[scada["TimeStamp"] <= end],
        events.loc[events["TimeStamp"] <= end],
        summary.loc[summary["TimeStamp"] <= day],
    )


def build(
    scada: pd.DataFrame, events: pd.DataFrame, summary: pd.DataFrame, cfg: dict[str, Any]
) -> pd.DataFrame:
    return assemble_features(daily_aggregates(scada, cfg), events, summary, cfg)


def check_no_lookahead(
    scada: pd.DataFrame,
    events: pd.DataFrame,
    summary: pd.DataFrame,
    cfg: dict[str, Any],
    cases: list[tuple[int, pd.Timestamp]],
) -> list[dict[str, Any]]:
    """Return one entry per (station, day) whose row changes when the future is removed.

    An empty list means every checked row uses only data available at the end of its day. Each
    station is processed on its own, which is valid because no feature mixes turbines.
    """
    problems: list[dict[str, Any]] = []
    for station in sorted({s for s, _ in cases}):
        sc = scada.loc[scada["StationId"] == station]
        ev = events.loc[events["StationId"] == station]
        su = summary.loc[summary["StationId"] == station]
        full = build(sc, ev, su, cfg).set_index(KEYS)
        for st, day in cases:
            if st != station:
                continue
            cut = build(*truncate_sources(sc, ev, su, day), cfg).set_index(KEYS)
            key = (station, day)
            if key not in full.index:
                continue
            if key not in cut.index:
                problems.append({"StationId": station, "Day": day, "columns": ["<row missing>"]})
                continue
            a, b = full.loc[key], cut.loc[key]
            shared = [c for c in full.columns if c in cut.columns]
            bad = [
                c
                for c in shared
                if not np.isclose(float(a[c]), float(b[c]), rtol=1e-5, atol=1e-6, equal_nan=True)
            ]
            missing = [c for c in full.columns if c not in cut.columns]
            if bad or missing:
                problems.append({"StationId": station, "Day": day, "columns": bad + missing})
    return problems
