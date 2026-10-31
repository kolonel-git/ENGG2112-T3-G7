"""Budget sweeps keep validation isolation, strict ties and accounting identities."""

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from wt_predmaint.config import load_protocol
from wt_predmaint.evaluation.budgets import add_random_comparison, budget_sweep


def inputs():
    frame = pd.DataFrame(
        {
            "StationId": 1,
            "Day": pd.date_range("2021-01-01", periods=6),
            "split": "val",
            "label": [0, 0, 1, 1, 0, 0],
            "score": [1.0, 1.0, 3.0, 3.0, 2.0, 2.0],
        }
    )
    episodes = pd.DataFrame({"StationId": [1], "Start": [pd.Timestamp("2021-01-20")]})
    cfg = deepcopy(load_protocol())
    cfg["budget_analysis"]["budgets"] = [0.0, 5.0, 11.0, 30.0]
    return frame, episodes, cfg


def test_budget_signal_ties_and_accounting():
    frame, eps, cfg = inputs()
    before = frame.copy(deep=True)
    result = budget_sweep(frame, eps, cfg)
    assert result.false_alarms_per_turbine_month.le(result.budget + 1e-12).all()
    assert result.recall.eq(1).all()
    assert result.false_alert_days.tolist() == [0, 0, 2, 4]
    assert (result.n_detected + result.missed_episodes).equals(result.n_episodes)
    assert result.labelled_alert_days.is_monotonic_increasing
    pd.testing.assert_frame_equal(frame, before)
    tied = budget_sweep(frame.assign(score=1.0), eps, cfg)
    assert tied.iloc[0].n_detected == 0
    assert tied.iloc[-1].n_detected == 1


@pytest.mark.parametrize("change", ["split", "date", "missing", "duplicate", "infinite"])
def test_invalid_or_nonvalidation_inputs_rejected(change):
    frame, eps, cfg = inputs()
    if change == "split":
        frame["split"] = "test"
    elif change == "date":
        frame["Day"] += pd.DateOffset(years=2)
    elif change == "missing":
        frame["label"] = np.nan
    elif change == "duplicate":
        frame = pd.concat([frame, frame.iloc[:1]])
    else:
        frame["score"] = np.inf
    with pytest.raises(ValueError):
        budget_sweep(frame, eps, cfg)


def test_random_floor_matches_target_variant_and_budget():
    frame, eps, cfg = inputs()
    scores = budget_sweep(frame, eps, cfg).assign(target="A", variant="all")
    table = pd.concat([scores.assign(model="model"), scores.assign(model="B0b_random")])
    result = add_random_comparison(table)
    assert result.recall_minus_random_floor.eq(0).all()
    with pytest.raises(ValueError):
        add_random_comparison(table.loc[table.model.eq("model")])
    with pytest.raises(pd.errors.MergeError):
        add_random_comparison(pd.concat([table, table.loc[table.model.eq("B0b_random")]]))
