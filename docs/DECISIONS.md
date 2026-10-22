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

## Open decisions
- Dataset version: confirm v2.1.0 with the teammate who downloaded the data (D-004).
- Licence: none chosen; README carries a TODO.
- `TimeErr` meaning: the data dictionary does not describe `tblDailySummary`. It is inferred from its name and from its agreement with SCADA stop time (correlation 0.86, D-008); `tblSCTurFlag` would settle it (D-018).
- Handling of the four flagged `suspect_wind` turbine-years (flag only for now, D-015): decide per model whether to exclude them.
- Whether the 2022 drop of alarms 20 and 25 (about 60%) reflects a controller change; relevant to alarm-count features across years.
- Upgrade analysis needs `Hill_of_Towie_AeroUp_install_dates.csv` from Zenodo.
