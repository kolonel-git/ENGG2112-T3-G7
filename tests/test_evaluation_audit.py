"""Saved-result audits catch altered thresholds, metrics, intervals and missing floors."""

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from wt_predmaint.config import load_protocol
from wt_predmaint.evaluation.audit import audit_results
from wt_predmaint.evaluation.metrics import evaluation_row, threshold_for_budget


def saved_results():
    cfg = deepcopy(load_protocol())
    cfg["evaluation"]["bootstrap_resamples"] = 5
    frames, rows = {}, []
    eps = pd.DataFrame({"StationId": [1, 1], "Start": pd.to_datetime([
        "2021-02-15", "2023-02-15",
    ])})
    for model in ["model", "B0b_random"]:
        for split, year in [("val", 2021), ("test", 2023)]:
            days = pd.date_range(f"{year}-01-01", periods=60)
            frame = pd.DataFrame({
                "StationId": 1, "Day": days, "split": split,
                "label": ((days >= f"{year}-01-16") & (days < f"{year}-02-15")).astype(int),
                "score": np.arange(60, dtype=float) + (100 if split == "test" else 0),
            })
            frames[("all", "A", model, split)] = frame
        val = frames[("all", "A", model, "val")]
        threshold = threshold_for_budget(val.label.to_numpy(), val.score.to_numpy(), 1, 30.4375)
        for split in ["val", "test"]:
            frame = frames[("all", "A", model, split)]
            split_eps = eps.loc[eps.Start.dt.year.eq(frame.Day.dt.year.iloc[0])]
            row = evaluation_row(model, frame, split_eps, threshold, cfg)
            rows.append({"variant": "all", "target": "A", "split": split, **row})
    table = pd.DataFrame(rows)
    table["random_floor_pr_auc"] = table.pr_auc
    table["random_floor_recall"] = table.recall
    table["recall_minus_random_floor"] = 0.0
    return table, frames, {"A": eps}, cfg


def test_saved_test_table_uses_validation_threshold_even_when_test_exceeds_budget():
    table, frames, eps, cfg = saved_results()
    test = table.loc[table.split.eq("test")]
    assert test.false_alarms_per_turbine_month.gt(1).all()
    assert audit_results(test, frames, eps, cfg).status.eq("verified").all()


@pytest.mark.parametrize("column", ["threshold", "n_detected", "recall_lo", "random_floor_recall"])
def test_altered_results_are_rejected(column):
    table, frames, eps, cfg = saved_results()
    table.loc[0, column] += 1
    with pytest.raises(ValueError, match=column):
        audit_results(table, frames, eps, cfg)


def test_missing_random_floor_and_duplicate_result_keys_are_rejected():
    table, frames, eps, cfg = saved_results()
    with pytest.raises(ValueError, match="random floor"):
        audit_results(table.loc[table.model.ne("B0b_random")], frames, eps, cfg)
    with pytest.raises(ValueError, match="unique"):
        audit_results(pd.concat([table, table.iloc[:1]]), frames, eps, cfg)


def test_one_ulp_threshold_change_is_rejected_because_strict_ties_matter():
    table, frames, eps, cfg = saved_results()
    table.loc[0, "threshold"] = np.nextafter(table.loc[0, "threshold"], -np.inf)
    with pytest.raises(ValueError, match="threshold"):
        audit_results(table, frames, eps, cfg)


def test_scores_from_the_wrong_year_are_rejected():
    table, frames, eps, cfg = saved_results()
    key = ("all", "A", "model", "val")
    frames[key]["Day"] += pd.DateOffset(years=2)
    with pytest.raises(ValueError, match="outside val years"):
        audit_results(table, frames, eps, cfg)
