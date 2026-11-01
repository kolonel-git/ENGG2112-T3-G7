# Short-horizon any-alarm experiment decisions — 2026-11-01

These decisions implement the user's binary alarm-start request independently of the older daily degradation protocol and icing-only experiments. Settings live in `configs/short_horizon_alarm.json`; reusable code is in `src/wt_predmaint/short_horizon.py`.

- SH-001: User confirmed any alarm code starts, not a selected group. Retain windows where an alarm is already active; label any new start in a half-open cumulative forecast horizon.
- SH-002: Use the supplied processed ten-minute data without rebuilding it from the locally cleaned Excel workbooks. Validate grid and source-observation flags before training. The source labels' raw-data completeness is inherited, not re-established.
- SH-003: Exactly 12 preceding rows, excluding the decision-time row; sensor-only summaries. Horizons are 10, 20, 30 minutes. Each is trained independently.
- SH-004: Default all 21 physical turbines; chronological training before 2023, validation 2023, test 2024–2025. Purge entire history and maximum forecast at split boundaries. These years were previously inspected by the team.
- SH-005: Initial reproducible uniform row caps: 120k train, 50k validation/test. Do not rebalance evaluation. Save population/sample counts and state that results concern sampled overlapping windows for known turbines.
- SH-006: Compare logistic regression, random forest, XGBoost and constant-prevalence baseline. Fit transformations/weights on train only; choose thresholds on validation F1 and models on validation AP; freeze everything before testing.
- SH-007: Generate EDA before training. Save separate new experiment directories, leaving datasets and old notebooks intact. Include prediction/review notebooks for reuse.
