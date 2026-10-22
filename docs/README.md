# Documentation

Prefer an interactive walkthrough? Open [the target-A fault-prediction notebook](../notebooks/01_single_turbine_fault_prediction.ipynb)
for processing, training, validation and evaluation in one place.

Start with [general setup and usage](general/USAGE.md), then choose the workflow
that matches your experiment. Run terminal commands from the repository root.

| Folder | Contents |
|---|---|
| [general/](general/USAGE.md) | Installation, dependencies, IDE setup, troubleshooting, tests, and [Excel exports](general/EXCEL_EXPORTS.md). |
| [single_year/](single_year/README.md) | [One-turbine experiments](single_year/SINGLE_TURBINE.md), [pooled classifier training and plots](single_year/CLASSIFIER.md), dummy experiments, and [metrics](single_year/EVALUATION.md). Default year: 2018. |
| [multi_year/](multi_year/WORKFLOW.md) | Original shared pipeline and the [one-command single-turbine 2018–2025 experiment](multi_year/SINGLE_TURBINE.md). |
| project/ | [Decision log](project/DECISIONS.md), [data traps](project/DATA_TRAPS.md), [roadmap](project/roadmap.md), and [modelling plan](project/modelling_plan.md). |
| reference/ | Original alarm-code and processed-data descriptions. |

Install dependencies with:

```powershell
python -m pip install -r docs/general/requirements.txt
```

Single-year evaluation uses chronological folds across the same turbines. The
main multi-year evaluation uses held-out years. There is currently no dedicated
held-out-turbine runner; see [script scopes](../scripts/README.md).
