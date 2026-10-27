# Farhan: evaluation review and report handoff

Reviewed 2026-10-27 to 2026-10-28 against main revision `232aa6b`, pulled before starting.
Scope: `evaluation/metrics.py`, `splits.py`, `runner.py`, D-020 and D-024,
the validation alert-burden analysis and the evaluation/cost report section.

**Farhan review sign-off, 2026-10-28 (with Codex):** the harness implements the
recorded D-020/D-024 definitions. The saved validation results reproduce from their
scores, labels, episodes and protocol, including bootstrap intervals and random
floors. This signs off the evaluation implementation and current validation
tables; G3 model freeze and final test results are pending.

## Review findings and changes

| Check | Finding |
|---|---|
| Fitting and splits | The runner passes only labelled train rows to `fit`. Validation and test days are selected separately. Changing synthetic test features/labels leaves validation results and thresholds unchanged. `labelled_split` excludes unlabelled rows; leave-one-turbine-out holds each unique station out once. |
| Threshold and ties | The threshold uses validation negatives and exposure from all labelled validation days. Alerts use strict `score > threshold`. A tied block is never split; unused budget is visible in the realised rate. The official budget remains 1 false alarm per turbine-month. |
| Episode accounting | Episodes are assigned by start year. Only episodes with at least one labelled scored day in the preceding 30 days enter recall. Day -30 is included and the start day excluded. Lead time uses the first alert, among detected episodes only. |
| Intervals | 1,000 seeded turbine resamples, percentile 95% intervals, threshold fixed across draws. These do not include model/threshold selection uncertainty. |
| Random floor | The runner always adds floors and attaches the matching target/fleet/split random comparison. Duplicate model names and invalid floor joins now fail explicitly. |
| Input checks | Evaluation now rejects duplicate turbine-day keys, missing keys and infinite scores. Threshold selection rejects misaligned arrays, nonbinary labels, nonfinite scores, negative/nonfinite budgets and nonpositive exposure conversion. These checks do not alter valid results. |
| Test safeguard | `frozen=True` is required to score test. It is a caller declaration, not a persistent once-only lock or proof that G3 was approved. Kanav must retain the frozen protocol, decision and run artefacts. Farhan's auditor only reads saved scores and does not run a model. |
| Burden / F27 | The sweep already accepts every model in T0. F27 previously had a fixed model list; it now draws every supplied model except the constant floor, including future LightGBM models. The generated readout lists all models at the official budget with their random floors. |

The old random-stream test-floor examples in D-020 and DATA_TRAPS are historical,
superseded by the row-hashed floor in D-023/D-024. They are not final test results
for the current runner. The report uses the audited T0 validation floors instead.
B3's accepted data-loading deviation and required pre-G3 feature migration remain
Kanav/Parvez's dependency (D-031); this review does not close that work.

## Reproduced evidence

- Final checks: `python -m ruff check .` and `git diff --check` pass;
  `python -m pytest -q` passes **175 tests**. The existing logistic-regression
  penalty deprecation emits ten non-failing warnings. Added cases cover invalid
  inputs, warning-window endpoints, test isolation, duplicate model names,
  corrupted saved metrics/intervals/floors and single-ULP threshold changes.
- `python scripts/check_evaluation_results.py` verifies **36** saved rows across
  both targets, both fleet variants and nine models/floors per target/variant.
  It checks every point estimate, threshold and bootstrap bound, then verifies
  the matching floor and recall difference. Outputs:
  [audit](../reports/tables/evaluation_audit.csv) and
  [input hashes](../reports/tables/evaluation_audit.json).
- `python scripts/analyse_alert_budgets.py` reproduces **288** validation
  model/target/fleet/budget comparisons and checks the official-budget results
  against T0 before writing them. Inputs are hashed in
  [the burden manifest](../reports/tables/alert_budget_inputs.json).
- `python scripts/make_report_figures.py --only F27` regenerates F27 from that
  table. Its visual check covers both targets, fleet variants, all eight
  nonconstant model/reference curves and the official-budget diamonds.
- `python scripts/write_evaluation_section.py` writes the integration-ready
  [evaluation and cost section](../reports/EVALUATION_AND_COST.md). Numeric tables
  are generated directly from T0 and the sweep; prose distinguishes validation
  estimates, operational burden and unverified economic benefit.

## Dependencies still open

| Assignment | Status and next action |
|---|---|
| Harness review and sign-off | Complete, with evidence above and D-020/D-024 sign-off lines. |
| Evaluation/cost writing and number traceability | Current validation section complete; generated from scripts. Final test paragraph depends on G3. |
| Extend tables/F27 to frozen LightGBM | Pipeline prepared and tested with a synthetic new model. Actual LightGBM implementation, selected settings and G3 decision are absent on the pulled main revision. No substitute model or settings were invented. |
| Check the single test-results table | Auditor implemented and tested on synthetic saved test scores, including test rates above budget. No final test table or current-runner test score files were supplied; actual test verification remains pending Kanav's one frozen run. |

Once Kanav merges the models and records the settings freeze, reproduce validation
without a frozen flag:

```powershell
.venv/Scripts/python.exe scripts/make_baseline_table.py
.venv/Scripts/python.exe scripts/check_evaluation_results.py
.venv/Scripts/python.exe scripts/analyse_alert_budgets.py
.venv/Scripts/python.exe scripts/make_report_figures.py --only F27
.venv/Scripts/python.exe scripts/write_evaluation_section.py
```

After Kanav's single G3 test run, check its runner-format table using its saved
validation and test scores. For example, if the consolidated table is named
`reports/tables/frozen_test.csv`:

```powershell
.venv/Scripts/python.exe scripts/check_evaluation_results.py --table reports/tables/frozen_test.csv --out reports/tables/test_evaluation_audit.csv
```

The supplied table needs the runner's result keys, point estimates, interval
columns and random-floor columns. Scores must cover every feature day for each
included target/model/fleet/split. Validation scores are required to verify frozen
thresholds even when the supplied table contains test rows only. Use `--protocol`
with the retained frozen protocol if the working config has since changed. This
command verifies consistency; it does not refit, score, retune, or certify that the
test was run only once. Add verified test results separately from validation and
retain realised false-alarm rates even when they exceed 1.
