"""Baselines the runner knows by name (D-024).

To add a baseline: write a class that follows ``models.base.Baseline``, then add one line to
``REGISTRY`` mapping a short id to a function that builds it for a target ("a" or "b"). The floors
(B0a, B0b) are always run by the runner and are not listed here.
"""

from __future__ import annotations

from collections.abc import Callable

from wt_predmaint.models.base import Baseline
from wt_predmaint.models.rules import RecentFaultRule

REGISTRY: dict[str, Callable[[str], Baseline]] = {
    "B1_recent_fault": lambda target: RecentFaultRule(target),
    "B1_fault_count": lambda target: RecentFaultRule(target, "count"),
    "B1_fault_recency": lambda target: RecentFaultRule(target, "recency"),
    # "B2_alarm_count": lambda target: AlarmCountRule(target),      # Farhan
    # "B3_power_curve": lambda target: PowerCurveProxy(target),     # Parvez
    # "B4_logistic": lambda target: LogisticBaseline(target),       # Revael
}


def build(names: list[str] | None, target: str) -> list[Baseline]:
    """Baselines for ``names`` (all registered ones if None), in the order given."""
    chosen = list(REGISTRY) if names is None else names
    missing = [n for n in chosen if n not in REGISTRY]
    if missing:
        raise KeyError(f"unknown baselines {missing}; registered: {sorted(REGISTRY)}")
    return [REGISTRY[n](target) for n in chosen]
