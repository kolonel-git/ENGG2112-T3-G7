"""Common interface for O2 baselines and models (D-023, docs/BASELINE_SPLIT.md).

A baseline is a class with ``fit(train, cfg)`` and ``score(rows)``:

* ``fit`` sees train-year rows with labels only (see :func:`require_train_only`). It must not read
  validation or test data from anywhere.
* ``score`` returns one finite number per row, higher meaning riskier, indexed like ``rows``. It
  must work for every day that has features, labelled or not (D-022), must not use the ``label``
  or ``status`` columns, and must score a row without looking at the other rows in the batch.

``attach_labels`` builds the table both methods work on; ``check_score_contract`` is a test helper
that every baseline's unit tests should call.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np
import pandas as pd

from wt_predmaint.labels.horizon import split_of_year

KEYS = ["StationId", "Day"]
LABEL_COLUMNS = ["status", "label", "split"]
NO_DATA = "no_data"


@runtime_checkable
class Baseline(Protocol):
    """What the runner needs from a baseline."""

    name: str  # short id used in tables, for example "B1_recent_fault"
    target: str  # "a" or "b"

    def fit(self, train: pd.DataFrame, cfg: dict[str, Any]) -> Baseline: ...

    def score(self, rows: pd.DataFrame) -> pd.Series: ...


def attach_labels(
    features: pd.DataFrame, labels: pd.DataFrame, cfg: dict[str, Any]
) -> pd.DataFrame:
    """Feature table with ``status``, ``label`` and ``split`` joined on (StationId, Day).

    Every day with features is kept. Days without a label row get status ``no_data``, an empty
    label and a split taken from the year (D-006), so they can still be scored (D-022).
    """
    keep = labels[[*KEYS, *LABEL_COLUMNS]]
    table = features.merge(keep, on=KEYS, how="left")
    year_split = split_of_year(cfg)
    from_year = table["Day"].dt.year.map(year_split).fillna("none")
    table["split"] = table["split"].astype("object").where(table["split"].notna(), from_year)
    table["status"] = table["status"].astype("object").where(table["status"].notna(), NO_DATA)
    return table


def labelled(table: pd.DataFrame, split: str) -> pd.DataFrame:
    """Rows of one split with a label: the only rows used for fitting and for metrics."""
    out = table.loc[(table["split"] == split) & table["label"].notna()].copy()
    out["label"] = out["label"].astype(int)
    return out


def scoring_rows(table: pd.DataFrame, split: str) -> pd.DataFrame:
    """Every day of one split that has features, labelled or not (for plots and risk curves)."""
    return table.loc[table["split"] == split].copy()


def require_train_only(train: pd.DataFrame) -> None:
    """Call at the start of ``fit``: raises if any row is not a labelled train row."""
    if not (train["split"] == "train").all():
        raise ValueError("fit received rows outside the train split")
    if train["label"].isna().any():
        raise ValueError("fit received rows without a label")


def check_scores(scores: pd.Series, rows: pd.DataFrame) -> None:
    """Raise if scores are not one finite number per row, indexed like ``rows``."""
    if not isinstance(scores, pd.Series):
        raise TypeError("score must return a pandas Series")
    if not scores.index.equals(rows.index):
        raise ValueError("scores must be indexed like the rows they score")
    if not np.isfinite(scores.to_numpy(dtype=float)).all():
        raise ValueError("scores must be finite (no NaN or inf); use a documented fallback value")


def check_score_contract(baseline: Baseline, rows: pd.DataFrame) -> list[str]:
    """Problems with a fitted baseline's ``score``; an empty list means it keeps the contract.

    Checks, on the rows given:
    1. one finite number per row, indexed like the rows;
    2. scores do not change when the ``label`` and ``status`` columns are replaced;
    3. scores do not depend on the other rows in the batch (a half scored alone matches).
    """
    problems: list[str] = []
    base = baseline.score(rows)
    try:
        check_scores(base, rows)
    except (TypeError, ValueError) as exc:
        return [str(exc)]

    blind = rows.copy()
    blind["label"] = 1 - blind["label"].fillna(0).astype(int)
    blind["status"] = "zone"
    if not np.allclose(baseline.score(blind).to_numpy(), base.to_numpy(), equal_nan=True):
        problems.append("score depends on the label or status columns")

    half = rows.iloc[: max(1, len(rows) // 2)]
    if not np.allclose(baseline.score(half).to_numpy(), base.loc[half.index].to_numpy()):
        problems.append("score of a row depends on the other rows in the batch")
    return problems
