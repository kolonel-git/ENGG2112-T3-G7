# Farhan: evaluation review, 2026-10-29 (D-069)

Reviewed main `6535030` after merging the repaired Kanav stack and Revael's report.
Kanav's D-055 remains unchanged; this review's former D-055 is now D-069.

## Sign-off and scope

Farhan (with Codex) reviewed the D-020/D-024 harness: train-only fitting,
validation-selected strict-greater-than thresholds, episode warning-window endpoints,
first-alert lead time, fixed-threshold turbine bootstrap and matching random floors.
Input guards and synthetic tests are retained. The official budget stays at one
false-alert day per turbine-month of labelled exposure.

`python scripts/check_published_tables.py` passes for all **48 runner rows**
(24 validation and 24 test). The 24 test export rows match the saved runner tables;
the full-fleet validation rows match T0. Checks include validation/test threshold
identity, count ratios, matching floors, interval ordering and the validation budget.
Export comparisons allow 1e-12 absolute rounding error because the existing report
writer reads CSVs with pandas' default parser; threshold carryover within the runner
tables is checked exactly. See `reports/tables/published_table_audit.csv` and its
input-hash manifest. This is table consistency, not independent metric reproduction.

The current committed validation sweep includes R2 and both frozen LightGBM models.
F27 is regenerated from that table and includes all supplied nonconstant models.
The generated `reports/EVALUATION_AND_COST.md` now separates validation and frozen
test results, repeats the random floor beside recall and reports realised test rates.
R1/R2 test rates exceed the budget and their recall is not recall at the budget.
Interval inclusion of a random point estimate is not a paired significance test.
Costs mean labelled alert-day burden and missed inferred episodes; visits, prevented
failures and currency benefits cannot be estimated from these data.

## Remaining original-score audit

Local `data/interim/scores/` contains the older validation cache, no test files and
no R2/LightGBM scores. The historical `evaluation_audit.csv/json` verified 36 older
validation rows against main `232aa6b`; it does **not** certify current T0.
The current sweep and bootstrap intervals cannot be independently reproduced here.

Kanav needs to supply the original per-day validation and test score parquet files,
the retained D-060/D-061 protocol and the matching feature-day keys, labels and episodes.
Keep the full-fleet and exclusion-variant validation caches for every T0 model.
Do not refit, rescore or retune test to replace missing files.

After transfer, run the saved-score auditor (which fits/scores no model):

```powershell
.venv/Scripts/python.exe scripts/check_evaluation_results.py
.venv/Scripts/python.exe scripts/check_evaluation_results.py --table reports/tables/baselines_a.csv --out reports/tables/test_evaluation_audit_a.csv
.venv/Scripts/python.exe scripts/check_evaluation_results.py --table reports/tables/baselines_b.csv --out reports/tables/test_evaluation_audit_b.csv
.venv/Scripts/python.exe scripts/analyse_alert_budgets.py
```

Then regenerate F27 and the evaluation section. Use `--protocol` with the retained
frozen protocol if the working protocol differs. Do not claim full numerical sign-off
until these checks pass. The auditor does not certify that a model was run only once.

## Reproduce the available checks

```powershell
.venv/Scripts/python.exe scripts/check_published_tables.py
.venv/Scripts/python.exe scripts/write_evaluation_section.py
.venv/Scripts/python.exe scripts/make_report_figures.py --only F27
.venv/Scripts/python.exe -m ruff check .
.venv/Scripts/python.exe -m pytest -q
```

PDF compilation was attempted with the built-in compiler but failed on the existing
`graphicx` pdftex driver under XeTeX (`pdfshellescape` undefined). No PDF build is
certified by this review; compile the integrated report with the intended TeX engine.

Final validation: **257 tests pass**, Ruff and `git diff --check` pass. The full
suite required unrestricted Windows temporary-file access for six Excel-export tests;
the sandbox run had permission errors. There are 18 non-failing upstream deprecation/SHAP
warnings. Four new published-table tests include deliberate corruptions.
