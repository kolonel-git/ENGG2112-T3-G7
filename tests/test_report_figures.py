from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from wt_predmaint.reporting import (  # noqa: F401  (registers figures)
    figures_data,
    figures_eval,
    figures_explore,
    figures_nbm,
    figures_robust,
)
from wt_predmaint.reporting.context import Ctx
from wt_predmaint.reporting.style import REGISTRY, apply_style, save

# These tests need no data: they check the registry and one schematic figure.


def test_registry_ids_and_files_are_unique_and_documented() -> None:
    ids = [s.fig_id for s in REGISTRY]
    paths = [s.relpath for s in REGISTRY]
    assert len(ids) == len(set(ids)) and len(paths) == len(set(paths))
    for s in REGISTRY:
        assert s.func is not None
        assert s.title and s.message and s.sources and s.section
        assert s.folder[:2].isdigit() and s.filename.startswith(s.fig_id)


def test_every_figure_folder_is_one_of_the_report_sections() -> None:
    folders = {s.folder for s in REGISTRY}
    assert folders == {
        "01_data",
        "02_labels",
        "03_features",
        "04_evaluation",
        "05_baselines",
        "06_nbm",
        "07_explain_robustness",
        "08_exploratory",
    }


def test_label_geometry_figure_uses_the_protocol_windows(tmp_path: Path) -> None:
    apply_style()
    spec = next(s for s in REGISTRY if s.name == "label_geometry")
    assert spec.func is not None
    fig = spec.func(Ctx())
    texts = " ".join(t.get_text() for t in fig.axes[0].texts)
    assert "days 1 to 30" in texts and "days 31 to 60" in texts and "last 7 days" in texts
    out = tmp_path / spec.relpath
    save(fig, out)
    assert out.exists() and out.stat().st_size > 1000


def test_f27_includes_new_models_from_the_budget_table(tmp_path: Path) -> None:
    table = pd.DataFrame([
        dict(variant="all", target=t, model=m, budget=b, false_alert_days=10 * b,
             missed_episodes=20 - b)
        for t in ["A", "B"] for m in ["B0b_random", "M1_lightgbm"] for b in [0.1, 1.0]
    ])
    table.to_csv(tmp_path / "alert_budget_validation.csv", index=False)
    fig = figures_eval.alert_burden(Ctx(tables=tmp_path))
    assert "M1_lightgbm" in [t.get_text() for t in fig.legends[0].texts]
    assert all(len(ax.lines) == 2 for ax in fig.axes)
    plt.close(fig)
