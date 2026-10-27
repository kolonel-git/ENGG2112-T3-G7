# Team checklist for the final sprint

Last updated 2026-10-27. Submission is assumed to be Sunday 2026-11-01 (not yet confirmed with the
coordinator). Each person records their sign-off by adding a line under "Sign-offs" in a pull
request; nobody signs for someone else. Plan and gates: `docs/roadmap.md`.

## Reviews that close G1 (30 minutes each, by Wednesday 2026-10-28)

| Reviewer | Reviews | What to check | Sign-off |
|----------|---------|---------------|----------|
| Parvez | Cleaning layer: `data/clean.py`, `data/daily.py`, `scripts/build_clean_data.py` (D-013 to D-017) | Sentinel masks per table, de-duplication, outage-month and suspect-wind flags, the counts in `reports/tables/cleaning_report.csv` | TODO |
| Revael | Labels and features: `labels/`, `features/daily.py`, `scripts/build_labels.py`, `scripts/build_features.py` (D-019, D-021) | Window rules (30-day positive, 30 to 60-day zone, purge, censoring), no look-ahead in features | TODO |
| Farhan | Evaluation harness: `evaluation/metrics.py`, `splits.py`, `runner.py` (D-020, D-024) | Threshold chosen on validation only, event-level recall and lead time, turbine bootstrap, random-score floor next to every recall | TODO |

## Open items by owner

- **Parvez:** confirm the dataset version with whoever downloaded the data (D-004); choose the licence
  for the repository (the README carries a TODO); try to fetch `Hill_of_Towie_AeroUp_install_dates.csv`
  and the data dictionary from Zenodo. If the AeroUp dates are not in the repo by Wednesday, the
  upgrade analysis is out of the report and the report says so.
- **Farhan:** extend the alert-burden tables and F27 to the LightGBM models once their settings are
  frozen (Wednesday evening); keep the official budget at 1 false alarm per turbine-month (D-009).
- **Cyrus:** ethics section (false-alarm cost, maintenance-decision risk, CC-BY-4.0 data licence and
  citation, limits of the open dataset) and the report outline; integrate sections as they arrive.
- **Revael:** report sections for labels and features from `docs/FEATURES_AND_LABELS.md` and the
  existing figures in `reports/figures/02_labels` and `03_features`.
- **Kanav:** LightGBM grid and freeze (Wednesday), one frozen test run (Thursday), SHAP and
  robustness (Thursday and Friday).
- **Everyone:** confirm the submission date and the marking rubric with the unit coordinator today,
  and tell the group. Please do not move or rename anything under `docs/` before submission
  (several scripts and decisions refer to those paths).

## Sign-offs

(none recorded yet)
