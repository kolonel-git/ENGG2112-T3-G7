"""Train-only feature ranking and pruning around any baseline (D-066).

``SelectedBaseline`` ranks the candidate columns on the labelled train rows it is fitted on, keeps
the top k (or removes one named group), and fits the wrapped model on those columns only. The
ranking is recomputed inside every ``fit``, so a year-shift check or a refit never sees a column
choice made with other data. Scoring is the wrapped model's own.

Rankings: ``gain_cv`` (LightGBM gain share averaged over leave-one-train-year-out fits),
``year_stable_auc`` (columns whose single-column AUC lies on the same side of 0.5 in every train
year, ranked by the weakest year) and ``gain_cv_decorr`` (``gain_cv`` order that skips columns
highly correlated with one already kept).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from wt_predmaint.evaluation.robustness import group_columns
from wt_predmaint.models.base import Baseline, require_train_only
from wt_predmaint.models.lgbm import LightGBMBaseline, lgbm_feature_columns
from wt_predmaint.models.logistic import DAY_STATE_COLUMNS, FAULT_HISTORY_COLUMNS

METHODS = ("gain_cv", "year_stable_auc", "gain_cv_decorr")
MIN_YEAR_ROWS = 200
_CACHE: dict[tuple[Any, ...], pd.Series] = {}


def _fingerprint(train: pd.DataFrame, columns: list[str]) -> tuple[int, int, int]:
    sample = train[["StationId", "Day", *columns]].iloc[::37]
    return (len(train), int(train["label"].sum()), int(pd.util.hash_pandas_object(sample).sum()))


def gain_cv_ranking(train: pd.DataFrame, target: str, cfg: dict[str, Any]) -> pd.Series:
    """Mean LightGBM gain share per column over leave-one-train-year-out fits, largest first."""
    candidates = lgbm_feature_columns(train, cfg, use_nbm=False)
    key = ("gain_cv", target, *_fingerprint(train, candidates))
    if key not in _CACHE:
        years = sorted(train["Day"].dt.year.unique())
        if len(years) < 2:
            raise ValueError("gain_cv needs at least two train years")
        shares = []
        for year in years:
            fold = train.loc[train["Day"].dt.year != year]
            model = LightGBMBaseline(target, name="X_rank").fit(fold, cfg)
            table = model.importance_table().set_index("feature")["share"]
            shares.append(table)
        mean = pd.concat(shares, axis=1).fillna(0.0).mean(axis=1)
        _CACHE[key] = mean.sort_values(ascending=False)
    return _CACHE[key]


def year_stable_auc_ranking(train: pd.DataFrame, columns: list[str]) -> pd.Series:
    """Weakest-year signed distance of single-column AUC from 0.5, qualifying columns only.

    A column qualifies if every train year has enough rows with both classes and the AUC is on
    the same side of 0.5 in all of them. The value is the smallest of those distances, so a high
    value means the column separates the classes in the same direction in every year.
    """
    key = ("auc", *_fingerprint(train, columns))
    if key in _CACHE:
        return _CACHE[key]
    years = train["Day"].dt.year.to_numpy()
    y = train["label"].to_numpy()
    out: dict[str, float] = {}
    for column in columns:
        x = pd.to_numeric(train[column], errors="coerce").to_numpy(dtype=float)
        dist = []
        for year in np.unique(years):
            m = (years == year) & np.isfinite(x)
            if m.sum() < MIN_YEAR_ROWS or len(np.unique(y[m])) < 2 or np.ptp(x[m]) == 0:
                dist = []
                break
            dist.append(roc_auc_score(y[m], x[m]) - 0.5)
        if not dist:
            continue
        d = np.asarray(dist)
        if (d > 0).all() or (d < 0).all():
            out[column] = float(np.abs(d).min())
    result = pd.Series(out, dtype=float).sort_values(ascending=False)
    _CACHE[key] = result
    return result


def decorrelate(
    order: list[str], train: pd.DataFrame, threshold: float, rows: int, seed: int
) -> list[str]:
    """Greedy pass over ``order`` that skips a column too correlated with one already kept."""
    sample = train[order]
    if len(sample) > rows:
        sample = sample.sample(rows, random_state=seed)
    ranks = sample.apply(pd.to_numeric, errors="coerce").rank()
    corr = ranks.corr(min_periods=100).abs()
    kept: list[str] = []
    for column in order:
        if all(not corr.loc[column, k] > threshold for k in kept):
            kept.append(column)
    return kept


def ranked_columns(train: pd.DataFrame, target: str, method: str, cfg: dict[str, Any]) -> list[str]:
    """Candidate columns best first for ``method``, from labelled train rows only."""
    require_train_only(train)
    if method not in METHODS:
        raise ValueError(f"unknown ranking method {method!r}")
    r3 = cfg["exploratory_round3"]
    if method == "year_stable_auc":
        usable = lgbm_feature_columns(train, cfg, use_nbm=False)
        return list(year_stable_auc_ranking(train, usable).index)
    order = list(gain_cv_ranking(train, target, cfg).index)
    if method == "gain_cv":
        return order
    return decorrelate(
        order, train, float(r3["decorrelate_above"]), int(r3["rank_sample_rows"]), int(cfg["seed"])
    )


def family_columns(columns: list[str], cfg: dict[str, Any]) -> dict[str, list[str]]:
    """Columns of each removable group: the sensor groups, alarm counts, fault history, day state.

    ``day_state`` is the six daily state columns only; ``wind_rpm`` also removes every wind, rpm
    and operating-state summary, so the two overlap on purpose.
    """
    groups = group_columns(columns, cfg)
    groups["alarms"] = [c for c in columns if c.startswith("alarm_")]
    groups["fault_history"] = [c for c in columns if c in FAULT_HISTORY_COLUMNS]
    groups["day_state"] = [c for c in columns if c in DAY_STATE_COLUMNS]
    return groups


class SelectedBaseline:
    """Fit ``factory()`` on a pruned column set chosen from the rows given to ``fit``."""

    def __init__(
        self,
        factory: Callable[[], Baseline],
        target: str,
        method: str | None = None,
        keep: int | None = None,
        drop_group: str | None = None,
        name: str = "X_selected",
    ) -> None:
        if (method is None) == (drop_group is None):
            raise ValueError("give exactly one of method and drop_group")
        if method is not None and (keep is None or keep < 1):
            raise ValueError("method needs keep >= 1")
        self.factory, self.target = factory, target
        self.method, self.keep, self.drop_group, self.name = method, keep, drop_group, name

    def fit(self, train: pd.DataFrame, cfg: dict[str, Any]) -> SelectedBaseline:
        require_train_only(train)
        candidates = lgbm_feature_columns(train, cfg, use_nbm=False)
        if self.drop_group is not None:
            groups = family_columns(candidates, cfg)
            if self.drop_group not in groups or not groups[self.drop_group]:
                raise ValueError(f"group {self.drop_group!r} has no columns")
            gone = set(groups[self.drop_group])
            chosen = [c for c in candidates if c not in gone]
        else:
            assert self.method is not None and self.keep is not None
            chosen = ranked_columns(train, self.target, self.method, cfg)[: self.keep]
            if not chosen:
                raise ValueError(f"{self.method} kept no columns")
        self.columns_ = chosen
        self.inner_ = self.factory().fit(
            train.drop(columns=[c for c in candidates if c not in set(chosen)]), cfg
        )
        return self

    def score(self, rows: pd.DataFrame) -> pd.Series:
        if not hasattr(self, "inner_"):
            raise RuntimeError("fit must be called before score")
        return self.inner_.score(rows)
