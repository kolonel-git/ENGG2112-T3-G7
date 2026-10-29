# Team checklist for the final sprint

Last updated 2026-10-27. Submission is assumed to be Sunday 2026-11-01 (not yet confirmed with the
coordinator). Plan, gates and cut order: `docs/project/roadmap.md`. Tick a box only for your own items;
nobody signs off for someone else. Sign-offs go under "Sign-offs" at the end, in a pull request.

## Everyone

- [ ] Confirm the submission date and the marking rubric with the unit coordinator, and tell the group.
- [ ] Pull the latest `main` before starting anything.
- [ ] Do not move or rename files under `docs/` before submission (scripts and decisions refer to those paths).
- [ ] Never commit data (`data/` is git-ignored). Never invent results or references; every number in
      the report comes from a script and `configs/protocol.yaml`.
- [ ] One branch per task (`owner/short-topic`), small pull requests, one reviewer, no direct pushes to `main`.
- [ ] Fri: read your own report section and check every number traces to a script.

## Parvez: data and cleaning

- [ ] Tue/Wed: 30-minute review of the cleaning layer (`data/clean.py`, `data/daily.py`,
      `scripts/build_clean_data.py`, D-013 to D-017): sentinel masks per table, de-duplication,
      outage-month and suspect-wind flags, counts in `reports/tables/cleaning_report.csv`. Add your sign-off line.
- [ ] Tue: confirm the dataset version (D-004) with whoever downloaded the data; log the answer.
- [ ] Tue: choose the repository licence (the README carries a TODO) and record it.
- [ ] Tue: try to fetch `Hill_of_Towie_AeroUp_install_dates.csv` and the data dictionary from Zenodo.
      If the AeroUp dates are not in the repo by Wednesday, the upgrade analysis is cut and the report says so.
- [ ] Wed to Thu: write the data and cleaning section of the report (use `docs/project/DATA_TRAPS.md`).
- [ ] Decide, with Kanav, whether the four `suspect_wind` turbine-years are excluded in any model (open decision from D-015).

## Revael: features and labels

- [ ] Tue/Wed: 30-minute review of labels and features (`labels/`, `features/daily.py`,
      `scripts/build_labels.py`, `scripts/build_features.py`, D-019, D-021): window rules (30-day
      positive, 30 to 60-day zone, purge, censoring) and no look-ahead. Add your sign-off line.
- [ ] Wed to Thu: write the labels and features section from `docs/FEATURES_AND_LABELS.md` and the
      figures in `reports/figures/02_labels` and `03_features`.
- [ ] Thu: check the logistic baseline (B4, your P0 and B4 work) is described correctly in the report,
      including its validation PR-AUC against the random floor.

## Farhan: evaluation and cost

- [x] Tue/Wed: 30-minute review of the harness (`evaluation/metrics.py`, `splits.py`, `runner.py`,
      D-020, D-024): threshold chosen on validation only, event-level recall and lead time, turbine
      bootstrap, random-score floor next to every recall. Add your sign-off line.
- [ ] Wed evening: once Kanav freezes the LightGBM settings, extend the alert-burden tables and
      figure F27 (`scripts/analyse_alert_budgets.py`, D-054) to the LightGBM models; R2 is already in.
- [ ] Thu: check the single test-year results table against the harness (positive counts, intervals, floor).
- [x] Wed to Fri: write the evaluation and cost section. Keep the official budget at 1 false alarm per
      turbine-month (D-009) and show the random floor beside recall.

## Cyrus: ethics and report integration

- [ ] Tue/Wed: write the ethics section: cost of false alarms, risk to maintenance decisions, the
      CC-BY-4.0 data licence and citation, and limits of the open dataset (no measured power, no
      maintenance records, alarm descriptions for only 12 codes).
- [ ] Tue: set up the report outline, abstract and introduction draft. The claim to defend is what the
      data show: on the test years no model is distinguishable from the random-score floor, R1's
      validation lead did not hold, the validation threshold does not carry to a fixed false-alarm
      rate, and the NBM indicator is weak and drifts between train and validation years
      (`reports/TEST_RESULTS.md`, D-061).
- [ ] Wed to Fri: merge each section into the draft as it arrives.
- [ ] Fri night (G4): a complete draft that everyone has read.
- [ ] Sat: final proofread, citations (only sources actually checked), limitations section, submit (G5).
- [ ] Throughout: make sure every figure and table comes from a script and states its caveats.

## Kanav: modelling

- [ ] Wed: write `scripts/tune_lgbm.py`, run the 6-setting validation grid for targets A and B, freeze the
      winners in `configs/protocol.yaml` with a decision entry (G3). Report it as it is if nothing beats R1.
- [ ] Wed: add a figure and table comparing L1 and R2 with R1 and the random floor (D-025).
- [ ] Thu: one `--frozen` run on the test years for the frozen configurations (the only one), results
      table with positive counts, intervals and the random floor. Then SHAP on the LightGBM model
      (validation rows only).
- [ ] Fri: robustness on validation years only: leave-one-turbine-out and sensor-group dropout (pressure,
      drivetrain temperature, power electronics, wind and rpm), one table with positive counts (G4).
- [ ] Fri: write the models, O1, O2, SHAP and robustness sections.
- [ ] Sat: `ruff check .` and `pytest`, regenerate figures and tables from scripts, freeze the repo.
- [ ] Standing: keep `docs/project/DECISIONS.md` and `docs/project/roadmap.md` current; change `protocol.yaml` only with its entry.

## Cut order if a day slips

SHAP rank stability, then the 7-day-window sensitivity, random sensor dropout, the cost figure (keep
the table), and robustness except sensor-group dropout. Already cut: autoencoder, upgrade analysis
(unless the AeroUp dates arrive), lagged NBM inputs, a LightGBM NBM.

## Checkpoints

- Tue night: D-055 written, NBM frozen, G1 reviews signed off.
- Wed night (G3): LightGBM settings frozen, validation table done.
- Thu night: test results table exists.
- Fri night (G4): complete report draft, even if robustness or SHAP is thin.
- Sat (G5): fixes, final checks, submit. Sunday 2026-11-01 is the buffer.

## Sign-offs

- Farhan, 2026-10-29 (with Codex): harness review and evaluation/cost section complete;
  48 published runner rows pass table-consistency checks. F27 includes frozen R2/L1.
  Full reproduction from original R2/L1 validation and all test scores remains pending
  transfer from Kanav; see `docs/EVALUATION_REVIEW.md` (D-069).
