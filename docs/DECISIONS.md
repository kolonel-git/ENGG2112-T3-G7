# Decision log

One entry per decision, newest at the bottom of "Decisions". Change `configs/protocol.yaml` only together with an entry here.
Status is **decided** or **open** (value to be fixed during the work and logged here when it is).

## Template
### D-000: <short title>
- Date: YYYY-MM-DD
- Owner: <name>
- Status: decided / open
- Context: <why a decision is needed>
- Options considered: <a / b / c>
- Decision: <what was chosen>
- Consequences: <what changes, what is now fixed>

## Decisions

### D-001: In-scope alarm codes
- Date: 2026-10-20
- Owner: Kanav
- Status: decided
- Context: 42 alarm codes in the 2018 snippet; most are not relevant to the project goals.
- Decision: only codes 20, 25, 102, 1005, 3130, 8000, 8210, 8230, 8234, 8235, 8236, 10105 are in scope (list and descriptions in `configs/protocol.yaml`). All other codes are dropped from labels.
- Consequences: the 12 codes are the publicly described ones (`docs/reference/Alarm_Codes.txt`). Their role changed in D-003.

### D-002: Tiered alarm model
- Date: 2026-10-21
- Owner: Kanav
- Status: decided, role superseded by D-003
- Decision: Tier 1 = 3130, 10105; Tier 2 = 1005, 8000, 8210, 8230, 8234, 8235, 8236 (weather-driven stops); Tier 3 = 20, 25, 102 (non-stopping). Tiers live in `configs/protocol.yaml`.
- Consequences: alarm labels must come from the raw `tblAlarmLog`, because the processed workbooks keep ~0.2% of alarms (1,331 of 706,572 events for Mar to Dec 2018). Tier 1 was first intended as the O2 target; D-003 replaces that.

### D-003: O2 target is fault-time episodes plus SCADA-derived stops; alarm codes become features
- Date: 2026-10-21
- Owner: Kanav (recommended by the assistant, approved by the user)
- Status: decided
- Context: raw alarm logs 2016 to 2026-04 show Tier 1 events behave like routine operations. 3130 (pitch lubrication): median gap 5.0 days, CV 0.29, 672 to 721 events per turbine over 10 years, 8 to 9% of events in every month, ~3 min long; a 30-day window is 99.8% positive. 10105 (untwisting cables): ~9 min long, 56% of events in hours when 5 or more turbines log it, 2.5% of events in January vs 16.5% in July. Tier 2 codes are weather-driven; 8210, 8234, 8235, 8236 never occur; 20 and 25 are cut-in/cut-out counts. Interpretation of 3130 and 10105 is an inference from the data, to confirm with the dataset authors.
- Options considered: (A) fault-time episodes from `tblDailySummary.TimeErr`; (B) unplanned stops derived from 10-min SCADA, checked against alarm co-occurrence; (C) Tier 2 weather stops as the target; (D) keep Tier 1 and predict deviations from the schedule.
- Decision: targets are A and B, both trained and reported. Alarm codes of all tiers are used as features (counts per window, time since last event, gap deviation from the 5-day schedule for 3130). Tier 2 weather stops may be reported as a benchmark only. The earlier plan to use Tier 1 as the primary target is dropped.
- Evidence for A: `TimeErr` above 4 h on 2% to 7% of turbine-days; runs of such days give 5.8 episode starts per turbine-year, median 32 days apart (p10 4, p90 167), and 33% of 30-day windows positive. Computed on `tblDailySummary`, 2016 to 2026-04.
- Consequences: `TimeErr` meaning is not verified (data dictionary not in the repo), so B cross-checks A and the dictionary must be fetched (open item below). Thresholds for A and B are set in D-007 and D-008.

### D-004: Raw tables are the data source; dataset version provisional
- Date: 2026-10-21
- Owner: Kanav
- Status: decided (source), open (dataset version)
- Context: processed workbooks match raw in the years they cover but omit 2018-01 and 02 (175,640 rows), 2022-01 to 2022-04-04 (269,294 rows), and all of 2016, 2017, 2019 and 2026. They also lose the alarms and a few columns (yaw, pitch, operating state).
- Decision: all modelling data comes from `data/raw` via `src/wt_predmaint/data/`. Tables: `tblSCTurbine`, `tblSCTurTemp`, `tblSCTurPress`, `tblAlarmLog`, `tblDailySummary`. Alarms never from the `Alarmcode`/`AlarmEvents` columns of the workbooks. Dataset version stays v2.1.0 (Zenodo record 22662930) provisionally, because the data runs to 2026-04, which v1.0.0 does not cover.
- Consequences: raw SCADA from 2016 to 2026-04 (2019 only `tblSCTurbine`, Jan to May and Jul). Version to be confirmed with the teammate who downloaded the data.

### D-005: Label geometry
- Date: 2026-10-21
- Owner: Kanav
- Status: decided
- Context: the brief gives a 30-day positive window, a 30 to 60-day exclusion and a 7-day feature window without defining how they combine.
- Options considered: exclusion after a previous event vs exclusion in the gap before the positive window.
- Decision: prediction time t is the end of a turbine-day. Features use data up to t (7-day window). Label 1 if a fault episode starts in (t, t+30 days]. Rows with an episode start in (t+30, t+60 days] and none in (t, t+30] are dropped (ambiguous zone). Rows where the turbine is already in a fault episode at t are dropped, so the model predicts new episodes. All other rows are label 0.
- Consequences: set in `configs/protocol.yaml` (`positive_window_days: 30`, `exclusion_end_days: 60`, `feature_window_days: 7`). Replaces the alternative readings in `modelling_plan.md` 7.1.

### D-006: Time-based split; leave-one-turbine-out for robustness only
- Date: 2026-10-21
- Owner: Kanav
- Status: decided
- Context: raw SCADA covers 2016 to 2026-04; 2019 has no temperature or pressure data; 2022-08, 2022-09 and 2025-09 are outage months.
- Decision: train 2016, 2017, 2018, 2020; validation 2021, 2022; test 2023, 2024, 2025, 2026 (to 2026-04-30). No shuffling. Scalers, imputers, normal-behaviour models, thresholds and hyperparameters use train and validation years only; the test years are scored once per frozen configuration. Leave-one-turbine-out is used only in the robustness stage (`robustness.leave_one_turbine_out: false` for main runs).
- Consequences: positive counts per split must be reported; if too few, the split is revisited and logged here. Cross-year drift (hydraulic pressure steps, bearing temperature drift) is a known risk, covered by the year-shift stress test.

### D-007: Target A parameters
- Date: 2026-10-22
- Owner: Kanav
- Status: decided
- Decision: a fault day is a turbine-day with `TimeErr` of at least 4 hours (usable days only: not `partial_day`, not `bad_hours`); an episode starts on the first fault day after at least 7 fault-free days (`targets.daily_fault_time` in `configs/protocol.yaml`).
- Evidence (train and validation years only, 44,550 usable turbine-days): the threshold sweep gave 4 h with a 7-day gap = 4.3 episode starts per turbine-year, 31.6% of 30-day windows positive, 355 starts in train and 217 in validation. Alternatives: 2 h gave 40.6% positive (5.8 per turbine-year); 8 h gave 19.5% (2.5 per turbine-year, 205 and 144 starts); gaps of 3 and 14 days changed counts by about 15% either way.
- Rationale: a balanced positive rate with enough events in both splits, and consistent with the SCADA agreement check in D-008 (done at 4 h).

### D-008: Target B parameters
- Date: 2026-10-22
- Owner: Kanav
- Status: decided
- Decision: a stop interval is a 10-min row with generator rpm below 200 and primary wind speed (`wtc_PrWindSp_mean`, D-017) of at least 4.0 m/s, outside outage months. A stop is a run of at least 18 consecutive intervals (3 hours). An episode starts at the first stop after at least 7 stop-free days (`targets.scada_stops`).
- Evidence (train and validation years only): daily stop hours (rpm < 200, wind >= 4 m/s) correlate 0.857 with `TimeErr` and 0.778 with `TimeExternalErr`; stricter settings (wind >= 6 m/s) correlate 0.75 or lower. Day-level agreement with target A (`TimeErr` >= 4 h): daily stop hours >= 4 gave F1 0.854 (precision 0.84, recall 0.87); with the run rule, 18 intervals gave the best F1 (0.786; 12 gave 0.754, 24 gave 0.781, 36 gave 0.681). The rule gives about 12.6 qualifying runs and 5.1 episodes (7-day gap) per turbine-year.
- Consequences: this is a proxy for unplanned stops because `tblSCTurFlag` is not available (D-018). The agreement with `TimeErr` is also the first independent support that `TimeErr` measures real stop/error time, which the dictionary does not state. A and B are not independent labels: they overlap at day level (F1 about 0.8), and results for both must be read together.

### D-009: Evaluation definitions
- Date: 2026-10-21
- Owner: Kanav
- Status: decided
- Decision: metrics are PR-AUC, recall at a budget of 1 false alarm per turbine-month, and median lead time. An alert is a turbine-day with score above the threshold. A false alarm is an alert on a day whose label is 0 (not in the positive window and not in the dropped zones). The threshold is chosen on validation to meet the budget and frozen for test. Recall and lead time are event-level: an episode is detected if any alert falls in the 30 days before its start; lead time is the days from the first such alert to the start. Confidence intervals by bootstrap over turbines.
- Consequences: budget is `evaluation.false_alarm_budget_per_turbine_month: 1`. A cooldown between alerts is not used; if alert streaks distort the count, revisit and log.

### D-010: Random seed
- Date: 2026-10-21
- Owner: Kanav
- Status: decided
- Decision: `seed: 42` in `configs/protocol.yaml`; every random operation reads it.

### D-011: Power-curve baseline uses generator rpm instead of power
- Date: 2026-10-21
- Owner: Kanav
- Status: decided, provisional (see D-018: measured power exists in `tblSCTurGrid`, which is not yet in `data/raw`)
- Context: no 10-min measured active power exists; `PowerRef_endvalue` is a setpoint and `tblDailySummary` has only daily energy and mean power.
- Decision: the power-curve threshold baseline compares generator rpm to a wind-speed curve fitted on healthy train rows, and flags windows where the shortfall persists. This is a documented deviation from the brief.

### D-012: Repository housekeeping and planning branch
- Date: 2026-10-21
- Owner: Kanav
- Status: decided
- Decision: the teammate-supplied reference files (`Alarm_Codes.csv`, `Alarm_Codes.txt`, `Processed_Data_Description.txt`) live in `docs/reference/`. The planning work is closed through a pull request from `kanav/modelling-plan`, reviewed by one teammate before merge; training work starts on a new branch.

### D-013: Timestamp convention and parsing
- Date: 2026-10-21
- Owner: Kanav
- Status: decided
- Context: `wtc_CurTime_endvalue` lies 1 to 2 s before `TimeStamp` in 99% of rows (4 months checked), so `TimeStamp` marks the end of the 10-min interval. Files `tblAlarmLog` 2025_01 and 2026_01 and `tblSCTurbine`/`tblSCTurPress` 2025_01 were re-saved with month-first `M/D/YYYY H:MM` timestamps (seconds lost); month boundaries contain date-only values and duplicate rows.
- Decision: alarm events join to SCADA with `ceil` to 10 min. `parse_times` accepts ISO (with or without time), `M/D/YYYY H:MM` or `M/D/YYYY` and raises on anything else. SCADA keys are de-duplicated on (StationId, TimeStamp).
- Confirmed 2026-10-21 by the dataset dictionary (`data/raw/metadata/Hill_of_Towie_data_dictionary.md`, downloaded with the user's approval, MD5 verified): all timestamps are UTC and the 10-min `TimeStamp` is the end of the period.
- Daily rows are start-of-day labelled: a `tblDailySummary` row stamped D covers [D 00:00, D+1 day). Verified against the 10-min data (daily `NacelleWind` vs mean `wtc_PrWindSp` over the 144 intervals ending in (D, D+1 day]: correlation 0.9996, median difference 0.000; the other alignment gives 0.56). A prediction made at the end of day D uses data to D+1 00:00.
- Re-saved files (2025-01 SCADA and alarms, 2026-01 alarms and daily summary) also have floats rounded to about 10 significant digits (for example 1981.2889404296875 became 1981.28894); the effect is negligible.

### D-014: Sentinel handling
- Date: 2026-10-21
- Owner: Kanav
- Status: decided; implemented and refined by D-015 (per-table value lists, one `n_masked` count instead of per-sensor flags)
- Decision: values `200`, `-50`, `-100` in sensor columns are masked to NaN per column, with a boolean flag column per affected sensor; LightGBM uses NaN natively, linear models use train-fit imputation. Outage months (2022-08, 2022-09, 2025-09) are flagged and excluded from healthy-row fits.

### D-015: Cleaning rules for raw 10-min SCADA
- Date: 2026-10-21
- Owner: Kanav
- Status: decided
- Context: sentinels, duplicates, outage months and a few suspect wind sensors, found in the raw exploration (`DATA_TRAPS.md`). Checked in the data before choosing rules: a constant `200` on a single hub-pressure channel (min = max = 200, other channels normal, 105 rows with one of three channels, 13 with two) is a stuck-sensor placeholder, while `-100` hits all three hub channels together (630 rows in 2022); `200` in yaw and nacelle position sits in a uniform density of about 34,000 rows per degree (a real angle), and `PowerRef = 200` rows have normal rpm (a real curtailment setpoint).
- Decision: implemented in `src/wt_predmaint/data/clean.py`, parameters in `configs/protocol.yaml` under `cleaning`.
  - De-duplicate on (StationId, TimeStamp); the first copy is kept (all SCADA duplicates are exact month-boundary repeats).
  - Mask to NaN, per table and in `_mean`, `_min` and `_max` columns: `tblSCTurTemp` {200, -50}; `tblSCTurPress` {-100, 200}; `tblSCTurbine` none. Rows get `n_masked` (cells masked in that row). NaN itself marks the masked value, so no per-sensor flag columns (this simplifies D-014).
  - Inner-join the three tables on (StationId, TimeStamp).
  - A row belongs to the year of its interval START (TimeStamp minus 10 min), so the 00:00 stamp of 1 January closes the previous year.
  - Flag, never drop: `outage_month` (month with under 90% of the full 21-turbine grid) and `suspect_wind` (turbine-year with corr(wind speed, generator rpm) under 0.55 among rows with rpm of at least 1000, at least 1000 such rows).
- Result on the raw data (`reports/tables/cleaning_report.csv`): outage months 2022-08, 2022-09, 2025-09; suspect-wind turbine-years 2304511 (2022, 2025), 2304514 (2022), 2304520 (2023): exactly the four found by hand. 10,143,040 rows over 2016 to 2026-04 (2019 has no temperature or pressure tables and is excluded, D-006).
- Consequences: thresholds are config values, not code. A transient `-100` inside a 10-min `_min` column (about 1% of `HydPress_min`) means the mean is slightly pulled down; at roughly 0.5 bar this is below sensor noise and is left as is.

### D-016: Columns kept in the clean 10-min table
- Date: 2026-10-21
- Owner: Kanav (option chosen by the user)
- Status: decided
- Options considered: means only; means plus min/max for key sensors; all columns.
- Decision: every `_mean` column, `_min` and `_max` for the key pressure and temperature sensors (`KEY_SENSORS` in `data/scada.py`), plus wind-speed and generator-rpm `_stddev`, `PowerRef_endvalue` and `WTOperSt_endvalue`. 141 to 151 columns per year, of which 74 are min/max. About 620 to 660 MB per year in memory.
- Consequences: peak bearing and winding temperatures per interval are available as features. `scripts/build_scada_parquet.py` rebuilt all 31 table-year files with these columns.

### D-017: Wind speed comes from the primary channel
- Date: 2026-10-21
- Owner: Kanav
- Status: decided
- Context: the dataset dictionary says the secondary anemometer signal `wtc_SecAnemo` has a step change on each turbine's correction switch-on date (2017-08-01 for T14 and T17; 2018-04-10/11 for the rest), the primary channel is an ultrasonic anemometer and the controller's active channel is `wtc_AcWindSp`. `wtc_PriAnemo` is constant throughout.
- Decision: use `wtc_PrWindSp` (primary) for wind speed in cleaning checks, normal-behaviour inputs and the rpm-vs-wind curve. `wtc_SeWindSp` is allowed as a backup feature; `wtc_SecAnemo`, `wtc_PriAnemo` and `wtc_TetAnFrq` are not used as features.

### D-018: Missing source tables for power and error time: proceed with proxies
- Date: 2026-10-22
- Owner: Kanav (decided by the user: "skip it for now and use proxies")
- Status: decided (revisit if the tables are obtained)
- Context: the dictionary lists `wtc_ActPower` (active power, kW) in `tblSCTurGrid` and `wtc_ScTurSto_timeon` (time turbine error active within the period, s), `wtc_ScInOper_timeon` and `wtc_ScReToOp_timeon` in `tblSCTurFlag`. Neither table is in `data/raw`. Whether they are inside the Zenodo year zips is unverified (the yearly zips are about 1.4 GB, while the five tables we have are about 3.4 GB uncompressed per year).
- Options considered: (1) the user supplies the tables; (2) extend the download script to read only those files from the zips with HTTP range requests; (3) proceed with proxies.
- Decision: option 3. Power-curve baseline uses the rpm-vs-wind curve (D-011). Target B uses the SCADA stop rule of D-008. `TimeErr` meaning rests on its name and on its agreement with target B (D-008).
- Consequences: limitations to state in the report: no measured power, no 10-min error-time signal, `TimeErr` not documented. Later upgrade path: with `tblSCTurFlag`, target B can be redefined from `wtc_ScTurSto_timeon`; with `tblSCTurGrid`, the power-curve baseline and an underperformance feature become possible.

### D-019: Label construction details
- Date: 2026-10-22
- Owner: Kanav
- Status: decided
- Context: D-005 fixed the window geometry; building the labels required choices about episode spans, data gaps, the end of the data and the split boundaries.
- Decision: implemented in `src/wt_predmaint/labels/` (`episodes.py`, `horizon.py`), run by `scripts/build_labels.py`.
  - Candidate rows: each usable turbine-day of `tblDailySummary` for target A (not `partial_day`, not `bad_hours`); each observed day for target B (at least 115 of 144 intervals outside outage months, `min_observed_intervals_per_day`).
  - An episode spans its first to last fault day (A) or stop day (B). A row is `in_episode` if its day lies in that span (dropped).
  - A row is `positive` if an episode starts in days D+1 to D+30. A row with no positive but a start in D+31 to D+60 is in the `zone` (dropped).
  - A row is `purged` if D+60 falls in a different split from D (dropped, even if it looks positive), so no label uses events from another split. The last 60 days of 2020 and of 2022 are therefore dropped.
  - A row with no positive is `censored` if D+60 is beyond the last observed day, and `low_coverage` if fewer than 80% (`labels.min_horizon_coverage`) of the days D+1 to D+60 were observed. Both are dropped.
  - Precedence: in_episode, purged, positive, zone, censored, low_coverage, negative. The label is 1 for positive, 0 for negative, missing otherwise.
  - 2019 days stay in the label files with split `none` and are not used (no 2019 temperature or pressure data).
- Results (`reports/tables/label_summary.csv`):
  - Target A: train 22,309 labelled rows, 8,315 positive (37.3%), 355 episodes; validation 9,690 rows, 4,558 positive (47.0%), 217 episodes; test 17,951 rows, 8,099 positive (45.1%), 336 episodes.
  - Target B: train 20,911 rows, 10,001 positive (47.8%), 434 episodes; validation 8,827 rows, 4,346 positive (49.2%), 209 episodes; test 17,281 rows, 8,725 positive (50.5%), 382 episodes.
  - All 21 turbines have positives in every split.
- Verification: 3,000 random labelled rows per target recomputed by brute force from the episode table, 0 disagreements; target A episode counts match the D-007 sweep (355 and 217); 13 unit tests on synthetic data cover gap semantics, run rules, every status, purging and censoring.
- Consequences and caveats:
  - Positive rates are 37% to 50% after the exclusions, not the 32% of the raw sweep, so a prevalence-only model scores a PR-AUC equal to that rate. PR-AUC must be read against it.
  - Consecutive days share most of their 30-day window, so rows are strongly autocorrelated. Intervals are bootstrapped over turbines (D-009), not over rows.
  - Episodes are short: median length 1 day, 90th percentile 6 to 7 days; median spacing between episode starts is 48 to 51 days per turbine.
  - Turbine 2304515 is positive on 97% of its validation days (15 to 16 episodes in 2021 to 2022, found by both targets). Report results with and without it.
  - Positive rate by year varies (target A: 0.31 in 2016 to 0.63 in 2022).

### D-020: Evaluation harness details and the random-score floor
- Date: 2026-10-22
- Owner: Kanav
- Status: decided (budget question settled 2026-10-22: keep the brief's budget)
- Context: D-009 fixed the metrics. Building them needed choices about the false alarm denominator, undetectable episodes, the threshold rule and the confidence intervals.
- Decision: implemented in `src/wt_predmaint/evaluation/metrics.py` and `splits.py`.
  - Alert: `score > threshold` (strict, so tied scores are all in or all out). Rows must be labelled (0 or 1); unlabelled rows (D-019) are rejected.
  - Exposure: one turbine-month is `evaluation.days_per_month` = 30.4375 scored turbine-days (all labelled rows, positives and negatives).
  - Threshold: the lowest threshold with at most `floor(budget * turbine-months)` false alarms on validation; frozen for test. The realised test false alarm rate is reported next to the budget, because it can differ.
  - Episode recall: an episode counts only if at least one scored day lies in its 30-day window (the purge and the in-episode rule can remove all of them). Episodes are assigned to a split by the year of their start. The number of evaluable episodes is reported.
  - Lead time: days from the first alert in the window to the episode start.
  - Intervals: 1000 bootstrap resamples of turbines (`bootstrap_resamples`), percentile 95% interval (`ci_level`), seeded from `seed`. The threshold stays fixed across draws.
  - Splitters: `labelled_split` for the year splits, `leave_one_turbine_out` for the robustness stage only (D-006).
- Results: `scripts/run_dummy_baselines.py` writes `reports/tables/baselines_dummy.csv`. A constant score gives PR-AUC equal to the prevalence (0.45 to 0.51) and recall 0. A random uniform score at the budget gives, on test, recall 0.79 (A) and 0.84 (B) with median lead time 22 days; on validation 0.81 and 0.79.
- Finding: episode recall at 1 false alarm per turbine-month is easy to reach by chance. About 53% of scored days are negative, so a random alert rate near 6% of days meets the budget, and a 30-day window then contains at least one alert about 80% of the time. Random-score recall on test for target A: 0.79 at a budget of 1, 0.51 at 0.5, 0.34 at 0.25, 0.16 at 0.1, 0.06 at 0.05 (target B: 0.84, 0.65, 0.44, 0.21, 0.13). A model's recall at budget 1 therefore has about 0.2 of headroom above chance.
- Resolution (Kanav, 2026-10-22): stay with the original project brief. The headline metric stays recall at 1 false alarm per turbine-month, no change to scope. Every results table carries the random-score floor next to it so the headroom is visible. No extra budgets are added.

### D-021: Daily features and the look-ahead test
- Date: 2026-10-22
- Owner: Kanav
- Status: decided; the two `features:` values are provisional choices, not tuned
- Context: the model needs one feature row per labelled turbine-day, built only from data available at the end of that day (D-005, D-013).
- Decision: implemented in `src/wt_predmaint/features/daily.py`, run by `scripts/build_features.py`.
  - Day assignment: a 10-min interval belongs to the day of its start (`TimeStamp` minus 10 minutes), so day D holds `TimeStamp` in (D 00:00, D+1 00:00]. Alarms use the same rule after the ceil join (D-013). The `tblDailySummary` row stamped D is used for day D.
  - Daily sensor features: mean of every 10-min `_mean` column over producing intervals only (generator rpm at least `cleaning.producing_min_rpm`), set to missing if fewer than `features.min_producing_intervals` (12) such intervals; daily maximum of every `_max` column. This partly removes the effect of idle time on temperatures. It does not remove the effect of load or wind; the normal-behaviour model is meant to do that.
  - State features: interval count, fractions producing, idle and stopped, wind and rpm mean and spread, share of intervals with a masked value, outage-month and suspect-wind flags.
  - Window features: trailing 7-day mean and standard deviation (`labels.feature_window_days`) of the sensor and state features, on a gap-free calendar so missing days are not bridged; needs at least `features.window_min_days` (4) observed days.
  - Alarm features: counts per documented code, per tier and in total over the same 7 days. Days without alarms count as zero, including days without SCADA coverage.
  - Fault-history features from `tblDailySummary` (target A rule, usable days only): days since the latest fault day up to and including D, and fault days in the last 30 days. They feed the recent-fault baseline. Not available as 10-min signals for target B.
  - Not model inputs: keys, the three data-quality flags, interval counts and masked share (`feature_columns`).
- Look-ahead test: `features/leakage.py` deletes everything after the end of day D from every source (SCADA and alarms up to D+1 00:00, summary rows up to D), rebuilds, and requires the row for D to be identical. Unit tests include a deliberately leaky feature (tomorrow's temperature) that the check catches. `scripts/check_feature_leakage.py` runs it on real data.
- Results (`reports/tables/feature_summary.csv`):
  - Feature table: 70,816 turbine-days, 366 model columns.
  - Every labelled row has features, except 4 target-A test rows.
  - Real-data look-ahead check: 48 random turbine-days on 6 turbines, no difference found (4 turbines and 20 days in an earlier run).
- Consequences and caveats:
  - Missing values are common: on average 13% to 22% of cells per split. 12 of 110 base sensor columns are over 50% missing because the sensor only exists in some years (for example `IO1IntTm`, `NMMBrdTm`, `TrafoTMa`). About 6% to 8% of labelled rows have no producing interval at all.
  - The modelling step must drop or impute columns using train years only (for example, keep columns observed in at least a set share of train rows). That share is not chosen yet.
  - The 2022 drop in alarm codes 20 and 25 (open decisions) affects the alarm window counts across years.
  - Only trailing mean and spread are used; trends and cross-sensor ratios are not built. Add them only if baselines justify it.
  - Normal-behaviour residual features are not here yet; they need the fitted models.

### D-022: Scores are drawn for every day with features; blanks are explained
- Date: 2026-10-22
- Owner: Kanav
- Status: decided
- Context: plots of filtered data had many gaps (about 30% of the calendar has no label: 18% zone, 5% purged, 4% in episode, 1% censored for target A). Users read them as missing data.
- Decision: a risk score or feature is computed and plotted for every turbine-day that has features, including days dropped from labelling. Dropped days are left out only when metrics are computed (D-019, D-020). `evaluation/plotting.py` and `scripts/plot_risk_timeline.py` draw the line on a gap-free calendar with three bands: day status (the 7 label statuses plus `no_data`), split, and where the value itself is NaN.
- Consequences: a gap in the line now means the value is missing (for example 2019 has no temperature or pressure data; days with under 12 producing intervals), not that the day was dropped from labelling. Dropped days show as coloured bands under a continuous line.

### D-023: Baseline interface and score contract
- Date: 2026-10-22
- Owner: Kanav
- Status: decided
- Context: `docs/BASELINE_SPLIT.md` splits the baselines across four people; they need one shared shape so a runner can evaluate any of them the same way.
- Decision: implemented in `src/wt_predmaint/models/base.py` (interface and helpers) and `models/dummy.py` (the two floors as examples).
  - `Baseline` protocol: `name`, `target` ("a" or "b"), `fit(train, cfg)` and `score(rows)` (higher is riskier).
  - Working table: `attach_labels` joins `status`, `label` and `split` onto the feature table. Every feature day is kept; days with no label row get status `no_data`, an empty label and a split from the year (D-006), so they can be scored (D-022). `labelled(table, split)` gives the rows used for fitting and metrics; `scoring_rows(table, split)` gives every day.
  - `fit` receives labelled train rows only and calls `require_train_only` first.
  - Score contract, checked by `check_score_contract` and to be called in every baseline's tests: one finite number per row indexed like the rows (a documented fallback instead of NaN); no use of the `label` or `status` columns; a row's score does not depend on the other rows in the batch.
  - The random floor hashes (StationId, Day, seed) so that its score keeps the row-independence rule. This changes the exact random numbers of `baselines_dummy.csv`; that table was made with a shared random stream. It will be regenerated by the runner and its numbers are floors, not findings.
- Results: on real data the table matches the label counts (target A: train 22,309, validation 9,690, test 17,947 labelled rows; 4 test rows lack features), and both floors pass the contract check on 2,000 sampled validation days for each target.
- Consequences: the runner (`evaluation/runner.py`) is the next step; rule baselines must handle tied scores (see BASELINE_SPLIT 2.3).

### D-024: Runner for baselines
- Date: 2026-10-22
- Owner: Kanav
- Status: decided
- Context: the baseline split (D-023, `docs/BASELINE_SPLIT.md`) needs one way to evaluate any baseline, so results are comparable and every row sits next to the random floor (D-020).
- Decision: implemented in `src/wt_predmaint/evaluation/runner.py`, `models/registry.py` and `scripts/run_baselines.py`.
  - Per baseline: check the score contract on a seeded sample of 2,000 validation days (a violation stops the run with `ContractError`); fit on labelled train rows; score every day of the split that has features (D-022); choose the threshold on validation to meet the budget (D-009); evaluate with `evaluation_row`.
  - Validation is always run. Test is scored only with `frozen=True` (`--frozen` in the script, which prints a warning): test years are scored once per frozen configuration.
  - The constant and random floors are always added. Every row carries `random_floor_pr_auc`, `random_floor_recall` and `recall_minus_random_floor` for the same variant, target and split.
  - `variant` is `all`, or `excl_<ids>` when turbines are excluded from fitting and evaluation (for example turbine 2304515, D-019); the sensitivity table gets its own file.
  - Registering a baseline is one line in `REGISTRY`; nothing else changes.
  - Outputs: `reports/tables/baselines_<target>.csv` (with `_<variant>` for a sensitivity run) and per-day scores in `data/interim/scores/` for `scripts/plot_risk_timeline.py --scores`.
  - Replaces `scripts/run_dummy_baselines.py` and `baselines_dummy.csv` (D-020 results line); the floors now come from `models/dummy.py` (D-023).
- Results (validation only, floors only; no real baseline is registered yet): constant score PR-AUC 0.470 (A) and 0.492 (B) with recall 0; random score recall 0.78 (A) and 0.76 (B) at 0.999 and 1.0 false alarms per turbine-month, median lead 20 and 21 days. These replace the D-020 random numbers, which used a shared random stream (D-023); the headline finding is unchanged.
- Consequences: Farhan's T0 table (BASELINE_SPLIT 6.4) reads these files rather than recomputing. The test split has not been scored.

### D-030: H0 healthy training days and ten-minute intervals
- Date: 2026-10-22
- Owner: Parvez (implemented with Codex, requested by the user)
- Status: decided; operational defaults, not tuned using validation/test
- Context: BASELINE_SPLIT 4.1 requires a shared healthy-row selector for B3, R1,
  and R2, using the 2016–2025 data scope and the existing model-interface helpers.
- Decision: Add `models/healthy.py` with `healthy_days(labels, features, cfg,
  target)` and `healthy_intervals(labels, features, scada, cfg, target)`. These
  return boolean masks aligned with their input rows; H0 is a selector, not a
  Baseline risk scorer. Reuse `attach_labels`, `require_train_only`, interval-start
  day assignment, and shared stop-run detection. No hidden file reads or fitting.
- Day rules: Require both configured training-year membership and declared train
  split, `negative` status with label 0, and explicitly false outage/suspect-wind
  flags. Add the separate `healthy` protocol block with scope 2016–2025 and a
  seven-day recovery buffer. Seven days matches the existing weekly context and
  provides a conservative post-episode settling period; it is not empirically
  optimised or a claim about mechanical recovery time.
- Recovery: Infer recovery from the supplied complete target-specific train-label
  history, including in-episode/unlabelled rows. Exclude the seven days following
  an episode's end; require complete preceding calendar history, so missing days
  and training-block starts also get a buffer. Never use validation/test history
  or target A's fault-history input to establish target B recovery.
- Intervals: Require a healthy interval-start day, clear quality flags, finite
  rpm/wind, and wind >= configurable 4.0 m/s. This is an operating cut-in proxy
  aligned with the existing stop wind threshold; a provider/manufacturer cut-in
  specification has not been verified locally. Exclude whole qualifying stop
  runs using existing configured rpm/wind/duration rules. Find runs across complete
  clean training history, including midnight/year boundaries, before day selection.
- Report: Add `scripts/report_healthy_rows.py` writing
  `reports/tables/healthy_rows.csv`, by target/turbine/year over the scoped decade.
  Missing clean-year files produce blank interval counts and an availability flag.
  Local shared inputs computed A=2,291 healthy days/247,738 intervals and
  B=1,735 days/169,163 intervals. These are partial-source counts: only 2018/2019
  shared clean files are present, and 2019 is not a configured train year.
- Validation: Full pytest suite passed (102 tests, including 11 H0 cases).
  Changed Python files pass Ruff. `ruff check .` reports 38 existing findings
  in unrelated files; those were not expanded into this small H0 implementation.
  Tests cover exclusion of held-out/spoofed splits and out-of-scope years,
  held-out perturbation invariance, exact recovery bounds, missing history and
  flags, finite inputs, cut-in equality, station isolation, index/input preservation,
  midnight/year-crossing stop runs, and reproducible missing-year reporting.
- Consequences: B3/R1/R2 may fit on these selected training rows without changing
  their score contract. No B3 curve, R1/R2 estimator, fault classifier or validation/
  test model evaluation is implemented by H0. Source data and global split years
  are unchanged. Usage is documented in `docs/HEALTHY_ROWS.md`.
- Review note (Kanav, 2026-10-23): the healthy-day and interval counts quoted above came from a
  partial source (only the 2018 and 2019 clean files) and are superseded by the table regenerated
  from the full clean data (target A: 11,255 healthy days and 1,186,895 healthy 10-minute intervals; target B: 9,259 days and
  925,340 intervals, out of 68,299 feature days; `reports/tables/healthy_rows.csv`). The remark that `ruff check .` had 38 findings came from a
  branch that added `sys.path` shims to scripts; they are removed here and `ruff check .` passes.

### D-040: Train-only preprocessing for P0
- Date: 2026-10-22
- Owner: Revael
- Status: decided
- Context: sensor coverage changes between years, about 12 base sensors are over 50% missing, and the linear baseline needs finite, comparable inputs without learning from validation or test.
- Options considered: retain every column / keep columns observed in at least 50% or 80% of labelled train rows; mean or median imputation; missing indicators on or off; no clipping or train-quantile clipping.
- Decision: P0 treats NaN and infinity as missing; keeps a column when at least 50% of labelled train values are finite (inclusive); then drops train-constant columns. It learns medians on train, does not add missing indicators in the primary run, does not clip, and standardises linear-model inputs with the train mean and population standard deviation. Column order and every learned statistic are frozen after `fit`; tree models may use the same object with scaling disabled.
- Consequences: P0 is fitted separately for targets A and B because their labelled train rows differ. Missing sensor availability is not itself a primary B4 signal. `scripts/report_p0_preprocessing.py` reports retained/dropped columns and before/after non-finite shares for train and validation only.

### D-041: B4 logistic settings and validation tuning
- Date: 2026-10-22
- Owner: Revael
- Status: decided (selected C frozen 2026-10-23 after validation tuning)
- Context: L2 regularisation strength is not fixed by the brief and may differ between targets. It must be selected on validation outside `fit` (C11), without touching test years.
- Options considered: C in {0.01, 0.1, 1.0}; validation PR-AUC or recall-at-budget as the selection metric; smaller or larger C for an exact tie; L2 alone or an extra L1/elastic-net variant.
- Decision: `scripts/tune_b4_logistic.py` compares C = 0.01, 0.1 and 1.0 separately for A and B through the shared runner, selects highest validation PR-AUC, and breaks an exact tie toward the smaller C. The primary model is balanced L2 logistic regression with `liblinear`, `max_iter=2000`, `tol=1e-4`, and the protocol seed. An unexpected non-finite model probability falls back to the neutral score 0.5. L1/elastic-net is out of scope for the primary baseline.
- Results (re-run by Kanav 2026-10-23 on the full prepared data, validation only, 1,000 bootstrap resamples): C = 0.01 had the highest
  validation PR-AUC for both targets (A: 0.4474, 0.4382 and 0.4336 for C = 0.01, 0.1 and 1.0; B: 0.4511, 0.4473 and 0.4451). The values are frozen as
  `baseline_b4.selected_c.a = 0.01` and `.b = 0.01`. Tables: `reports/tables/b4_logistic_tuning.csv`, `b4_logistic_validation.csv` and
  `b4_logistic_coefficients.csv`. B4 stays below the random score's PR-AUC on validation (A 0.4474 against 0.4649; B 0.4511 against 0.4830), so it
  does not demonstrate useful early warning. Test years were not scored.
- Consequences: a B4 run fails clearly if either requested target's `selected_c` is `TODO`; both are now set, so `scripts/run_baselines.py` works with the default registry.

### D-042: B4 coefficient reporting
- Date: 2026-10-22
- Owner: Revael
- Status: decided
- Context: correlated, standardised window features make a full coefficient dump hard to read, while the report needs a compact first view of what the linear baseline associates with risk.
- Decision: report the 20 largest positive and 20 largest negative standardised coefficients per target, with direction, rank and absolute magnitude. Describe them as model attribution/association, never as causal effects.
- Consequences: `scripts/tune_b4_logistic.py` writes `reports/tables/b4_logistic_coefficients.csv` for the validation-selected candidate. Correlated features can split or reverse weights, so coefficients must be interpreted together with their preprocessing and limitations.

### D-043: B4 primary population and sensitivity
- Date: 2026-10-22
- Owner: Revael
- Status: decided
- Context: four turbine-years have suspect wind signals, and turbine 2304515 is positive on almost every labelled validation day (D-019). Excluding either group only after seeing model results would make the primary population validation-selected.
- Decision: the primary B4 fit/evaluation keeps all labelled train/validation rows and never uses the `suspect_wind` flag as a predictor. Report the runner's pre-specified sensitivity excluding turbine 2304515 separately; do not replace the primary result with it. Suspect-wind turbine-years remain a stated limitation rather than a post-hoc exclusion.
- Consequences: B4 stays comparable with the other primary baselines and does not learn a quality flag, but wind-derived features in flagged years may be less reliable. Any later suspect-wind exclusion is a new, explicitly named sensitivity decision.

### D-050: B1 recent-fault rules
- Date: 2026-10-22
- Owner: Farhan
- Status: decided
- Decision: `models/rules.py` implements three fixed scores, all using existing past-only target-A history for both targets (BASELINE_SPLIT 6.1).
  - Primary `B1_recent_fault`: `fault_days_30 + tie_weight / (1 + days_since_fault_day)`, with `tie_weight = 0.5` from the new `b1_recent_fault` config block. Count is primary: the recency contribution is bounded below one count step. Recency breaks ties without randomness or using outcomes.
  - Single-column comparisons: `B1_fault_count = fault_days_30` and `B1_fault_recency = 1 / (1 + days_since_fault_day)`.
  - Missing recency (no known previous fault, or missing history) contributes zero: the limit of the formula for a fault infinitely long ago. Missing count contributes the configured zero fallback. This means unknown history is scored as no known faults, not proven health; report missing-history counts alongside results.
  - No fitting or variant selection from outcomes. `fit` checks labelled train rows and reads config only. Each score is finite, indexed like the input, independent of labels/status and of other rows in the scoring batch (D-023).
  - Ties can remain when both features match. Keep the strict threshold from D-020; report realised false alarms and any zero-alert result rather than adding arbitrary noise.
- Evaluation: register all three with the shared runner (D-024). Validation only, both targets, constant and random floors included. Test is reserved for frozen configurations. Config choices are fixed before observing validation results.
- Reproduction: `python scripts/run_b1_recent_fault.py` writes dedicated `b1_a.csv`, `b1_b.csv`, per-day validation scores and missing-history counts; `--exclude 2304515` writes separate sensitivity files. The general runner also accepts the three registered B1 names.
- Data preparation: the supplied sibling raw folder contains 2019 sensor files, unlike the earlier copy described in DATA_TRAPS. SCADA conversion now defaults to the years in the protocol splits (with an explicit `--years` override), preserving D-006 and avoiding silently adding 2019 stop episodes. The source daily-summary history is still loaded by the existing daily loader, unchanged.
- Results (validation only): primary B1 PR-AUC 0.559 / 0.550 and episode recall 0.225 / 0.219 for A / B, versus random recall 0.780 / 0.760. Excluding 2304515 gives primary recall 0.205 / 0.169, still below random 0.774 / 0.746. All three rules, floors and intervals are in `reports/tables/b1_*.csv`; interpretation and coverage are in `reports/B1_README.md`. No test scores were produced. Do not promote recency-only to primary after seeing validation outcomes.

### D-031: B3 pooled healthy wind/RPM proxy
- Date: 2026-10-22
- Owner: Parvez
- Status: decided; provisional operating choices
- Implemented `models/power_curve.py` under the shared Baseline interface,
  registered as `B3_power_curve`, with `scripts/run_b3_power_curve.py` and
  `tests/test_power_curve.py`. See `docs/POWER_CURVE.md` for usage and outputs.
- Pooled fleet medians in 0.5 m/s bins; working range 4–25 m/s inclusive;
  minimum 30 intervals/bin; linear interpolation between supported centres,
  no outer extrapolation. These are provisional operating choices, not verified
  manufacturer limits or choices tuned on held-out data.
- Target-specific H0 training intervals only; exclude stop runs, outages and
  suspect-wind intervals from curve fitting. Exclude outage/suspect/nonfinite
  intervals from scoring, but retain stop intervals as possible underperformance.
- Tolerance is twice the population standard deviation of training residuals.
  Daily exceedance share and signed mean shortfall use interval-start days.
  Score is trailing 7-calendar-day mean share with configured minimum four
  observed days. Unsupported windows fall back to zero, with evidence availability
  exported separately; zero is not proof of healthy behaviour.
- Full labels are used only for train-only H0 selection. Curve/tolerance remain
  fixed when constructing validation features; test data is loaded only upon
  explicit scoring (runner requires frozen mode). Score is a key lookup with no
  label or batch dependence. CLI `--interim` is forwarded to B3 source paths.
- Initial A validation: AP .454594, episode recall .215311, median lead 21 days,
  false-alert days/turbine-month .998878; random recall .779904. B: AP .499860,
  recall .302083, lead 21 days, false-alert rate .996538; random recall .760417.
  These results fail to demonstrate useful early warning. No test evaluation.
- Saved curves/plots, availability, fit diagnostics, scores and validation tables
  for A/B. Fit uses 1,186,879 healthy A intervals and 938,075 B intervals.
- Checks: 107 tests pass; changed Python files pass Ruff. Extended
  `check_feature_leakage.py --b3` to recompute after future truncation at fixed fit:
  two real sampled turbine-days passed, plus synthetic causal-window tests.
- Review note (Kanav, 2026-10-23): B3 deviates from BASELINE_SPLIT rules C4, C9 and C14. Its constructor takes optional
  sources, its `fit` reads the clean 10-minute tables (train years, and validation years to build features with the train-fitted
  curve) and its `score` loads further years on demand, instead of reading new feature columns from `daily_features.parquet`.
  The design is leak-safe (the curve and tolerance are fitted on train only, the score is a key lookup that does not use labels or
  other rows, and it passed the contract check and the truncation check), so the deviation is accepted for now. Moving the
  B3 daily columns into `scripts/build_features.py` is a follow-up, to be done before the frozen test run (Gate G3).
  Re-run by Kanav 2026-10-23 on the full prepared data through `scripts/run_baselines.py`: validation PR-AUC 0.455 (A) and 0.499 (B)
  against 0.465 and 0.483 for the random score, recall 0.215 and 0.307 against 0.780 and 0.760.

### D-025: Report figures are generated by registered functions
- Date: 2026-10-23
- Owner: Kanav
- Status: decided
- Context: the final report needs a chart for every important step, and figures drawn by hand or in notebooks cannot be reproduced or checked against the data.
- Decision: each figure is a function registered with `reporting.style.figure` (id, folder, takeaway, source, caveats) in `src/wt_predmaint/reporting/`; `scripts/make_report_figures.py` draws all of them into `reports/figures/<section folder>/` and writes `reports/figures/README.md` (index with takeaway, report section, sources and caveats). Folders: `01_data`, `02_labels`, `03_features`, `04_evaluation`, `05_baselines`. 30 figures so far (F01 to F25, F26a to F26e).
  - Style: the project palette (target A blue, target B orange everywhere), one blue ramp for magnitude, neutral ink for text, direct labels, no dual axes.
  - Model results use validation years only; the test years are never scored for a figure. Label and feature figures describe all years because they involve no model.
  - Every takeaway sentence was checked against the rendered chart; about a dozen were rewritten because the first draft did not match the chart (for example B3 was missing from the comparison charts until T0 was rerun with it).
- Results worth knowing for the report:
  - Episode recall at the project budget is below the random score for every baseline, including R1; R1 beats the random line only at the tightest budgets (F15). Real scores alert in streaks on neighbouring days, so the same false-alarm count covers fewer distinct 30-day windows than scattered random alerts.
  - Alarm codes 3130, 20 and 25 are positive in 99% of 30-day windows and 10105 in 83%, against 42% to 49% for the fault targets (F03).
  - A and B episodes match within 7 days for 75% (A) and 72% (B) of episodes (F07).
  - Lead time of the baselines is about the same as the random score's (F19).
- Consequences: a new important step gets a registered figure in the same PR. `reports/tables/t0_*.csv` were regenerated with B3 included (values unchanged for the other baselines). Older loose figures in `reports/figures/` (`B2/`, `R1/`, `b3_curve_*.png`, one timeline) are superseded by F21, F23 and F26a to F26e.

## Open decisions
- Dataset version: confirm v2.1.0 with the teammate who downloaded the data (D-004).
- Licence: none chosen; README carries a TODO.
- `TimeErr` meaning: the data dictionary does not describe `tblDailySummary`. It is inferred from its name and from its agreement with SCADA stop time (correlation 0.86, D-008); `tblSCTurFlag` would settle it (D-018).
- Handling of the four flagged `suspect_wind` turbine-years (flag only for now, D-015): decide per model whether to exclude them.
- Whether the 2022 drop of alarms 20 and 25 (about 60%) reflects a controller change; relevant to alarm-count features across years.
- Upgrade analysis needs `Hill_of_Towie_AeroUp_install_dates.csv` from Zenodo.


### D-051: B2 alarm comparison
- Date: 2026-10-22
- Owner: Farhan
Compare total, tier2, tier1+tier2 and each of the twelve documented alarm codes.
Use existing causal seven-day count features. Missing/nonfinite counts contribute zero;
no random jitter breaks ties. Select primary separately per target by validation PR-AUC,
with configured candidate order breaking exact ties. Save every candidate's metrics.
Validation-only monthly mean counts document seasonality and codes 20/25 coverage changes.

### D-052: R1 healthy reference
- Date: 2026-10-22
- Owner: Farhan
Reuse Parvez H0 verbatim from commit 889a90c (healthy.py, report_healthy_rows.py, tests, HEALTHY_ROWS.md).
The runner builds the mask from complete training label history before filtering labelled
fitting rows. healthy_fit is training-selection metadata, never a scoring feature.
Use seven configured operating temperature sensors: per-turbine healthy daily mean and
population standard deviation, minimum 30 observations and std > 1e-6. Otherwise use pooled
healthy-training moments; disable sensors with no usable pooled reference. Unseen turbines
use pooled moments. Score maximum absolute z of the causal seven-day operating mean.
Missing sensors are skipped; all missing scores are zero. No validation fitting or tuning.
Fail explicitly if no sensor reference is usable. Export fitted counts/moments/fallback sources.

### D-053: T0 integration
- Date: 2026-10-22
- Owner: Farhan
Preserve Revael P0/B4 commit e6c2428 and use its documented validation C grid/tuning script.
Run every available registered baseline for A and B, full fleet and excluding 2304515,
with shared floors, thresholds, episode metrics and configured bootstrap intervals.
No test scoring or frozen flag. B3 and R2 remain pending until their implementations arrive;
T0 is a reproducible available-baseline comparison, not final team sign-off.

Validation selection: B2 A=tier1_2 (PR-AUC 0.517018), B=code_10105 (0.510763).
B4 C=0.01 for both targets (A 0.4474, B 0.4511), chosen by the predeclared PR-AUC rule.
These are development selections, not authorization to score held-out test years.
