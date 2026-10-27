"""Runner: fit baselines on train, choose thresholds on validation, evaluate (D-020, D-024).

For each baseline the runner

1. checks the score contract on a sample of validation days (``models.base``);
2. fits on labelled train rows only;
3. scores every day of the split that has features;
4. picks the alert threshold on validation to meet the false alarm budget (D-009);
5. evaluates validation (and test only when ``frozen=True``) with ``evaluation_row``.

The constant and random floors are always run, and every row carries the random floor's PR-AUC and
recall for the same target, split and variant, so a result is never read without it (D-020).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import pandas as pd

from wt_predmaint.evaluation.metrics import (
    episodes_in_split,
    evaluation_row,
    threshold_for_budget,
)
from wt_predmaint.evaluation.splits import split_years
from wt_predmaint.models.base import (
    Baseline,
    check_score_contract,
    check_scores,
    labelled,
    scoring_rows,
)
from wt_predmaint.models.dummy import PrevalenceBaseline, RandomBaseline

RANDOM_FLOOR = "B0b_random"
CONTRACT_SAMPLE = 2000
FLOOR_COLUMNS = {"pr_auc": "random_floor_pr_auc", "recall": "random_floor_recall"}


class ContractError(RuntimeError):
    """A baseline broke the score contract (models/base.py)."""


@dataclass
class RunResult:
    table: pd.DataFrame  # one row per (variant, baseline, split)
    scores: dict[
        tuple[str, str, str], pd.DataFrame
    ]  # (variant, baseline, split) -> StationId, Day, score


def variant_name(exclude_stations: Iterable[int]) -> str:
    ids = sorted(int(s) for s in exclude_stations)
    return "all" if not ids else "excl_" + "_".join(str(s) for s in ids)


def with_floors(baselines: list[Baseline], target: str) -> list[Baseline]:
    """The two floors first, then the given baselines (a floor already listed is not repeated)."""
    names = {b.name for b in baselines}
    floors: list[Baseline] = [
        f for f in (PrevalenceBaseline(target), RandomBaseline(target)) if f.name not in names
    ]
    return [*floors, *baselines]


def _add_floor_columns(table: pd.DataFrame) -> pd.DataFrame:
    keys = ["variant", "target", "split"]
    floor = table.loc[table["model"] == RANDOM_FLOOR, [*keys, *FLOOR_COLUMNS]]
    floor = floor.rename(columns=FLOOR_COLUMNS)
    out = table.merge(floor, on=keys, how="left", validate="many_to_one", indicator=True)
    if not out["_merge"].eq("both").all():
        raise ValueError("every result needs its matching random floor")
    out = out.drop(columns="_merge")
    out["recall_minus_random_floor"] = out["recall"] - out["random_floor_recall"]
    return out


def run_baselines(
    baselines: list[Baseline],
    table: pd.DataFrame,
    episodes: pd.DataFrame,
    cfg: dict[str, Any],
    target: str,
    splits: tuple[str, ...] = ("val",),
    frozen: bool = False,
    exclude_stations: Iterable[int] = (),
    resamples: int | None = None,
    check_contract: bool = True,
) -> RunResult:
    """Run baselines for one target; see the module docstring.

    ``table`` comes from ``models.base.attach_labels``; ``episodes`` from the matching target.
    Scoring ``test`` needs ``frozen=True``: test years are scored once per frozen configuration.
    ``exclude_stations`` drops turbines from fitting and evaluation (a sensitivity variant).
    """
    unknown = set(splits) - {"val", "test"}
    if unknown:
        raise ValueError(f"unknown splits {sorted(unknown)}; choose from val and test")
    if "test" in splits and not frozen:
        raise ValueError("test years are scored once per frozen configuration: pass frozen=True")
    if len({b.name for b in baselines}) != len(baselines):
        raise ValueError("baseline names must be unique")
    for b in baselines:
        if b.target != target:
            raise ValueError(f"baseline {b.name} is for target {b.target!r}, not {target!r}")

    excluded = sorted(int(s) for s in exclude_stations)
    variant = variant_name(excluded)
    if excluded:
        table = table.loc[~table["StationId"].isin(excluded)]
        episodes = episodes.loc[~episodes["StationId"].isin(excluded)]
    ev = cfg["evaluation"]
    train = labelled(table, "train")
    if any(getattr(b, "requires_healthy_rows", False) for b in baselines):
        from wt_predmaint.models.healthy import healthy_days

        mask = healthy_days(
            table[["StationId", "Day", "label", "status", "split"]], table, cfg, target
        )
        healthy_train = mask.reindex(train.index).fillna(False)
    val_all = scoring_rows(table, "val")
    sample = val_all.sample(min(CONTRACT_SAMPLE, len(val_all)), random_state=int(cfg["seed"]))

    rows: list[dict[str, Any]] = []
    scores: dict[tuple[str, str, str], pd.DataFrame] = {}
    for baseline in with_floors(baselines, target):
        fitting = train.copy()
        if getattr(baseline, "requires_healthy_rows", False):
            fitting["healthy_fit"] = healthy_train
        fitted = baseline.fit(fitting, cfg)
        if check_contract:
            problems = check_score_contract(fitted, sample)
            if problems:
                raise ContractError(f"{baseline.name}: " + "; ".join(problems))

        frames: dict[str, pd.DataFrame] = {}
        for split in ("val", *[s for s in splits if s != "val"]):
            all_days = scoring_rows(table, split)
            day_scores = fitted.score(all_days)
            check_scores(day_scores, all_days)
            scores[(variant, baseline.name, split)] = all_days[["StationId", "Day"]].assign(
                score=day_scores.to_numpy()
            )
            frame = labelled(table, split)
            frames[split] = frame.assign(score=day_scores.reindex(frame.index).to_numpy())

        threshold = threshold_for_budget(
            frames["val"]["label"].to_numpy(),
            frames["val"]["score"].to_numpy(),
            float(ev["false_alarm_budget_per_turbine_month"]),
            float(ev["days_per_month"]),
        )
        for split in splits:
            eps = episodes_in_split(episodes, split_years(cfg, split))
            row = evaluation_row(baseline.name, frames[split], eps, threshold, cfg, resamples)
            rows.append({"variant": variant, "target": target.upper(), "split": split, **row})

    result = pd.DataFrame(rows)
    return RunResult(_add_floor_columns(result), scores)
