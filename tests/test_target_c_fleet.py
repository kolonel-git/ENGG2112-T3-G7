"""Tests for the Target C fleet labels, episodes and metrics (D-071)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score

spec = importlib.util.spec_from_file_location(
    "tcf", Path(__file__).resolve().parents[1] / "scripts" / "target_c_fleet_run.py")
tcf = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(tcf)


def ts(s: str) -> np.datetime64:
    return np.datetime64(s, "ns")


def test_labels_horizon_active_and_block() -> None:
    t = np.array([ts(f"2021-01-01T{h:02d}:00") for h in range(8)])
    starts = np.array([ts("2021-01-01T03:00"), ts("2021-01-01T05:30")])
    on = np.array([ts("2021-01-01T03:00")])
    off = np.array([ts("2021-01-01T04:10")])
    bs = np.full(len(t), ts("2021-01-01T00:00"))
    be = np.full(len(t), ts("2021-01-01T07:00"))
    y = tcf.label_anchors(t, starts, on, off, 1, bs, be)
    # 02:00 -> start at 03:00 is in (02:00, 03:00]; 03:00 and 04:00 active; 05:00 -> 05:30;
    # 06:00 nothing in (06:00, 07:00]; 07:00 horizon leaves the block
    assert y.tolist() == [0, 0, 1, -1, -1, 1, 0, -1]


def test_episode_merge() -> None:
    s = np.array([ts("2021-01-01T00:00"), ts("2021-01-01T03:00"), ts("2021-01-01T10:00")])
    e = tcf.episode_starts(s, 6)
    assert np.array_equal(e, s[[0, 2]])


def test_ap_matches_sklearn_with_ties() -> None:
    rng = np.random.default_rng(0)
    score = rng.integers(0, 20, 500).astype(float)
    y = (rng.random(500) < 0.2).astype(int)
    boot = tcf.APBootstrap(score, y, np.zeros(500, dtype=int))
    assert abs(boot.value() - average_precision_score(y, score)) < 1e-12
    # integer multiplicities equal duplicated rows
    turb = rng.integers(0, 3, 500)
    boot = tcf.APBootstrap(score, y, turb)
    mult = np.array([2, 0, 1])
    rows = np.concatenate([np.flatnonzero(turb == 0)] * 2 + [np.flatnonzero(turb == 2)])
    assert abs(boot.value(mult) - average_precision_score(y[rows], score[rows])) < 1e-12


def test_threshold_respects_budget() -> None:
    score = np.arange(10, dtype=float)
    y = np.array([0] * 8 + [1, 1])
    thr = tcf.threshold_at_budget(score, y, n_labelled=1461, budget=1.0, hours_per_month=730.5)
    assert ((score > thr) & (y == 0)).sum() == 2


def test_episode_detection() -> None:
    t = np.array([ts(f"2021-01-01T{h:02d}:00") for h in range(6)])
    alert = np.array([0, 0, 1, 0, 0, 0], dtype=bool)
    lab = np.ones(6, dtype=bool)
    eps = [(0, 1, 4, ts("2021-01-01T04:00")), (0, 4, 6, ts("2021-01-01T06:00"))]
    m = tcf.episode_metrics(alert, lab, t, eps, 1)
    assert m["det"].tolist() == [1] and m["ev"].tolist() == [2] and m["lead"] == 2.0
