"""Write the LaTeX tables of the final report from the saved result tables (D-063).

Reads ``reports/tables/*.csv`` and writes ``tabular`` bodies to ``reports/report/tables/`` which
``reports/report/main.tex`` includes. Nothing is fitted or scored here; every number is copied from
a table written by another script, so the report's numbers trace to scripts and the protocol.

    python scripts/write_report_tables.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "reports" / "tables"
OUT = ROOT / "reports" / "report" / "tables"

NAMES = {
    "B0a_constant": "Constant score",
    "B0b_random": "Random score",
    "B1_recent_fault": "B1 recent fault",
    "B1_fault_count": "B1 fault count",
    "B1_fault_recency": "B1 fault recency",
    "B2_alarm_count": "B2 alarm count",
    "R1_zscore": "R1 z-score",
    "B3_power_curve": "B3 power curve",
    "B4_logistic": "B4 logistic",
    "R2_nbm_indicator": "R2 NBM indicator",
    "L1_lgbm": "L1 LightGBM",
    "L1_lgbm_nbm": "L1 LightGBM + NBM",
}
GROUPS = {
    "pressure": "Pressure",
    "drivetrain_temperature": "Drivetrain temperature",
    "power_electronics": "Power electronics",
    "wind_rpm": "Wind and rpm",
}


def tex(text: str) -> str:
    return str(text).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def ci(v: float, lo: float, hi: float, digits: int = 3) -> str:
    return f"{v:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"


def write(name: str, lines: list[str]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_tables() -> None:
    t = pd.read_csv(TABLES / "test_results.csv")
    t = t.loc[t["model"] != "B0a_constant"]
    for target in ("A", "B"):
        d = t.loc[t["target"] == target]
        lines = [
            r"\begin{tabular}{@{}lccc@{}}",
            r"\toprule",
            r"Model & PR-AUC [95\% interval] & Episode recall [95\% interval] "
            r"& False alarms per turbine-month \\",
            r"\midrule",
        ]
        for r in d.itertuples(index=False):
            lines.append(
                f"{tex(NAMES[r.model])} & {ci(r.pr_auc, r.pr_auc_lo, r.pr_auc_hi)} & "
                f"{ci(r.recall, r.recall_lo, r.recall_hi)} & "
                f"{r.false_alarms_per_turbine_month:.2f} \\\\"
            )
        lines += [r"\bottomrule", r"\end{tabular}"]
        write(f"test_{target.lower()}.tex", lines)


def valtest_table() -> None:
    rows = []
    for target in ("a", "b"):
        b = pd.read_csv(TABLES / f"baselines_{target}.csv")
        b = b.loc[b["variant"] == "all"]
        for model in NAMES:
            if model == "B0a_constant":
                continue
            v = b.loc[(b["model"] == model) & (b["split"] == "val")].iloc[0]
            s = b.loc[(b["model"] == model) & (b["split"] == "test")].iloc[0]
            rows.append((target.upper(), model, v.pr_auc, s.pr_auc))
    lines = [
        r"\begin{tabular}{@{}lcccc@{}}",
        r"\toprule",
        r" & \multicolumn{2}{c}{Target A} & \multicolumn{2}{c}{Target B} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
        r"Model & Validation & Test & Validation & Test \\",
        r"\midrule",
    ]
    for model in NAMES:
        if model == "B0a_constant":
            continue
        a = next(r for r in rows if r[0] == "A" and r[1] == model)
        b = next(r for r in rows if r[0] == "B" and r[1] == model)
        lines.append(
            f"{tex(NAMES[model])} & {a[2]:.3f} & {a[3]:.3f} & {b[2]:.3f} & {b[3]:.3f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("valtest.tex", lines)


def label_table() -> None:
    s = pd.read_csv(TABLES / "label_summary.csv")
    lines = [
        r"\begin{tabular}{@{}llrrrr@{}}",
        r"\toprule",
        r"Target & Split & Labelled days & Positive days & Positive share & Episodes \\",
        r"\midrule",
    ]
    for r in s.itertuples(index=False):
        lines.append(
            f"{r.target} & {r.split} & {r.labelled_rows:,} & {r.positives:,} & "
            f"{100 * r.positive_rate:.1f}\\% & {r.episodes} \\\\".replace(",", r"{,}")
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("labels.tex", lines)


def robustness_tables() -> None:
    d = pd.read_csv(TABLES / "robustness_dropout.csv")
    lines = [
        r"\begin{tabular}{@{}llcccc@{}}",
        r"\toprule",
        r"Target & Sensor group removed & PR-AUC & Change & Recall & False alarms \\",
        r"\midrule",
    ]
    for r in d.itertuples(index=False):
        name = "None (intact model)" if r.group == "none" else GROUPS[r.group]
        change = "" if r.group == "none" else f"{r.pr_auc_change:+.3f}"
        lines.append(
            f"{r.target} & {name} & {r.pr_auc:.3f} & {change} & {r.recall:.3f} & "
            f"{r.false_alarms_per_turbine_month:.2f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("dropout.tex", lines)

    s = pd.read_csv(TABLES / "robustness_loto_summary.csv")
    lines = [
        r"\begin{tabular}{@{}lcccc@{}}",
        r"\toprule",
        r"Target & Left out & Full fit & Prevalence & Fewest positive days \\",
        r"\midrule",
    ]
    for r in s.itertuples(index=False):
        lines.append(
            f"{r.target} & {r.pooled_pr_auc_left_out:.3f} & {r.pooled_pr_auc_full_fit:.3f} & "
            f"{r.pooled_prevalence:.3f} & {r.min_positive_rows_per_turbine} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("loto.tex", lines)


def exploratory_tables() -> None:
    """Validation PR-AUC and train-year year-shift PR-AUC of the exploratory grid (D-064)."""
    d = pd.read_csv(TABLES / "exploratory_validation.csv")
    sets = ["raw", "fleet", "fleet_season"]
    head = [r"Raw", r"Fleet", r"Fleet+season"]
    d["label"] = d["model"] + " (" + d["setting"] + ")"
    for column, name in (
        ("pr_auc", "expl_val.tex"),
        ("year_shift_pr_auc", "expl_shift.tex"),
    ):
        lines = [
            r"\begin{tabular}{@{}lcccccc@{}}",
            r"\toprule",
            r" & \multicolumn{3}{c}{Target A} & \multicolumn{3}{c}{Target B} \\",
            r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
            "Model (setting) & " + " & ".join(head * 2) + r" \\",
            r"\midrule",
        ]
        for label in dict.fromkeys(d["label"]):
            cells = []
            for target in ("A", "B"):
                for fs in sets:
                    v = d.loc[
                        (d["label"] == label) & (d["target"] == target) & (d["feature_set"] == fs),
                        column,
                    ]
                    cells.append(f"{v.iloc[0]:.3f}")
            lines.append(tex(label) + " & " + " & ".join(cells) + r" \\")
        lines += [r"\bottomrule", r"\end{tabular}"]
        write(name, lines)


def round2_tables() -> None:
    """Self-baseline PR-AUC and the turbine-ranking tables (D-065)."""
    first = pd.read_csv(TABLES / "exploratory_validation.csv")
    second = pd.read_csv(TABLES / "exploratory_self_baseline.csv")
    d = pd.concat([first.loc[first["feature_set"] == "raw"], second], ignore_index=True)
    d["label"] = d["model"] + " (" + d["setting"] + ")"
    sets = ["raw", "self", "self_fleet"]
    head = ["Raw", "Self", "Self+fleet"]
    lines = [
        r"\begin{tabular}{@{}lcccccc@{}}",
        r"\toprule",
        r" & \multicolumn{3}{c}{Target A} & \multicolumn{3}{c}{Target B} \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
        "Model (setting) & " + " & ".join(head * 2) + r" \\",
        r"\midrule",
    ]
    for label in dict.fromkeys(d["label"]):
        cells = []
        for target in ("A", "B"):
            for fs in sets:
                v = d.loc[
                    (d["label"] == label) & (d["target"] == target) & (d["feature_set"] == fs),
                    "pr_auc",
                ]
                cells.append(f"{v.iloc[0]:.3f}")
        lines.append(tex(label) + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("expl2_val.tex", lines)

    r = pd.read_csv(TABLES / "ranking_validation.csv")
    lines = [
        r"\begin{tabular}{@{}lcccc@{}}",
        r"\toprule",
        r" & \multicolumn{2}{c}{Target A} & \multicolumn{2}{c}{Target B} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
        r"Model & Top 3 & Top 5 & Top 3 & Top 5 \\",
        r"\midrule",
    ]
    for model in dict.fromkeys(r["model"]):
        cells = []
        for target in ("A", "B"):
            for k in (3, 5):
                v = r.loc[(r["model"] == model) & (r["target"] == target) & (r["k"] == k)].iloc[0]
                cells.append(
                    f"{v.difference:+.3f} [{v.difference_lo:+.3f}, {v.difference_hi:+.3f}]"
                )
        lines.append(tex(NAMES[model]) + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("ranking.tex", lines)


def main() -> None:
    test_tables()
    valtest_table()
    label_table()
    robustness_tables()
    exploratory_tables()
    round2_tables()
    print(f"wrote {sorted(p.name for p in OUT.glob('*.tex'))} to {OUT}")


if __name__ == "__main__":
    main()
