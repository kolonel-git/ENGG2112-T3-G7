# Documentation

Start with [general setup and usage](general/USAGE.md), then choose the workflow
that matches your experiment. Run terminal commands from the repository root.

| Folder | Contents |
|---|---|
| [general/](general/USAGE.md) | Installation, dependencies, IDE setup, troubleshooting, tests, and [Excel exports](general/EXCEL_EXPORTS.md). |
| [single_year/](single_year/README.md) | Running within-year experiments and interpreting their [metrics](single_year/EVALUATION.md). Default year: 2018. |
| [multi_year/](multi_year/WORKFLOW.md) | Shared data pipeline and evaluation using configured training, validation, and test years. |
| project/ | [Decision log](project/DECISIONS.md), [data traps](project/DATA_TRAPS.md), [roadmap](project/roadmap.md), and [modelling plan](project/modelling_plan.md). |
| reference/ | Original alarm-code and processed-data descriptions. |

Current all-turbine baseline work: [H0 masks](HEALTHY_ROWS.md),
[B3 implementation and validation](POWER_CURVE.md), and the
[integrated notebook](../notebooks/02_healthy_row_diagnostics_and_baseline_evaluation.ipynb).
Progress and interpretation updates are recorded in the project logbooks above
(2026-10-22, D-030/D-031).

Install dependencies with:

```powershell
python -m pip install -r docs/general/requirements.txt
```

Single-year evaluation uses chronological folds across the same turbines. The
main multi-year evaluation uses held-out years. There is currently no dedicated
held-out-turbine runner; see [script scopes](../scripts/README.md).
