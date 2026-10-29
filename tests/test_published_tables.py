"""Published-table review rejects a corrupted export or changed test threshold."""

import runpy
import shutil
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
check_tables = runpy.run_path(str(ROOT / "scripts/check_published_tables.py"))["check_tables"]


@pytest.mark.parametrize("corruption", ["none", "export", "threshold", "floor"])
def test_published_tables(tmp_path, corruption):
    for name in ["baselines_a.csv", "baselines_b.csv", "test_results.csv", "t0_validation.csv",
                 "alert_budget_validation.csv"]:
        shutil.copyfile(ROOT / "reports/tables" / name, tmp_path / name)
    if corruption == "none":
        assert len(check_tables(tmp_path)) == 48
        return
    name = "test_results.csv" if corruption == "export" else "baselines_a.csv"
    table = pd.read_csv(tmp_path / name, float_precision="round_trip")
    column = {"export": "n_detected", "threshold": "threshold", "floor": "random_floor_recall"}[
        corruption
    ]
    idx = table.index[table.split.eq("test")][0] if "split" in table else 0
    table.loc[idx, column] += 1
    table.to_csv(tmp_path / name, index=False)
    with pytest.raises((AssertionError, ValueError)):
        check_tables(tmp_path)
