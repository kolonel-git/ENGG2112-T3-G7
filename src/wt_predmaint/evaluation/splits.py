"""Splits for O2 (D-006). Time splits are fixed by year in the labels; this adds the frame helper
and the leave-one-turbine-out folds used only in the robustness stage (O3)."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pandas as pd


def labelled_split(labels: pd.DataFrame, split: str) -> pd.DataFrame:
    """Rows of one split ('train', 'val' or 'test') that carry a label (D-019)."""
    if split not in {"train", "val", "test"}:
        raise ValueError(f"unknown split {split!r}")
    out = labels.loc[(labels["split"] == split) & labels["label"].notna()].copy()
    out["label"] = out["label"].astype(int)
    return out


def split_years(cfg: dict[str, Any], split: str) -> list[int]:
    return [int(y) for y in cfg["splits"][f"{split}_years"]]


def leave_one_turbine_out(stations: list[int]) -> Iterator[tuple[list[int], int]]:
    """Yield (training turbines, held-out turbine) for each turbine."""
    ordered = sorted(set(stations))
    for held in ordered:
        yield [s for s in ordered if s != held], held
