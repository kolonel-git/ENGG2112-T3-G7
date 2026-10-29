# Pull request queue, in merge order

Last updated 2026-10-29. `gh` is not installed, so each PR is opened in the browser from the compare
link; nobody pushes to `main`. Repository: `kolonel-git/ENGG2112-T3-G7`. `main` is at `232aa6b`
(Farhan's PR #15).

## To merge, in this order

| # | Branch | Into | State | Notes |
|---|--------|------|-------|-------|
| 1 | `kanav/nbm` | `main` | Pushed; ready to open | NBM and indicator (D-057), R2 (D-058), LightGBM and its frozen settings (D-056, D-060), the single test run (D-061), SHAP and robustness (D-062), the rewritten report, README and stakeholder summary (D-063), the team checklist. Already contains `main` (Farhan's PR #15). Merge first: the other branches build on its decision IDs and tables. |
| 2 | `kanav/exploratory-models` | `main` | Pushed; ready after #1 | Exploratory validation-only runs (D-064), stacked on #1, so merge after #1. Touches `protocol.yaml`, `DECISIONS.md`, the report and README. |
| 2b | `kanav/label-sensitivity` | `main` | Pushed; ready after #2 | Label sensitivity round (D-068), stacked on #2 (needs its protocol and decision-log changes), so merge after #2. Adds `labels/variants.py`, `scripts/run_label_sensitivity.py`, tests, report, README and summary text. |
| 3 | `farhan-evaluation-cost-review` | `main` | Pushed by Farhan; needs an update before merging | Review and sign-off of the harness, `EVALUATION_AND_COST.md`, a saved-results auditor. After #1 lands it conflicts in `docs/DECISIONS.md`, `docs/roadmap.md` and `tests/test_report_figures.py`, and its **D-055 collides** with ours: renumber to D-069 (next free ID after D-068). Rerun `scripts/check_evaluation_results.py` and regenerate the evaluation section so it includes the LightGBM, R2 and test results. Trial merge of its code with ours passed ruff and 203 tests. |
| 4 | `revael/features-labels-report` | `main` | Pushed by Revael; ready after #1 | Docs only (sign-off and report section for labels and features). One trivial conflict in `docs/roadmap.md`. |

## Not yet created (needed before submission)

- **Parvez:** cleaning-layer sign-off; dataset version (D-004); repository licence and citation text
  (README TODOs); try to fetch the AeroUp dates and data dictionary from Zenodo.
- **Cyrus:** ethics section and report integration. The red items in `reports/report/main.tex` list
  what still needs confirming.
- **Final report integration PR:** after the sections above arrive, one PR that merges them into
  `reports/report/main.tex`.

## Do not merge

`Parvez/main`, `main_P` and `Data_Formatting_P` (Excel-workbook pipelines, and a docs reorganisation
that would move `docs/DECISIONS.md` and `docs/roadmap.md`), `revael/p0-b4-baseline` and `Farhan` (already in
`main`), and `test`.

## Compare links

- #1: https://github.com/kolonel-git/ENGG2112-T3-G7/compare/main...kanav/nbm?expand=1
- #2: https://github.com/kolonel-git/ENGG2112-T3-G7/compare/main...kanav/exploratory-models?expand=1
  (open after #1 is merged)
- #2b: https://github.com/kolonel-git/ENGG2112-T3-G7/compare/main...kanav/label-sensitivity?expand=1
  (open after #2 is merged)
- #3: https://github.com/kolonel-git/ENGG2112-T3-G7/compare/main...farhan-evaluation-cost-review?expand=1
- #4: https://github.com/kolonel-git/ENGG2112-T3-G7/compare/main...revael/features-labels-report?expand=1
