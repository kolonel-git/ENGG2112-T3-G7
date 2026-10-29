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

### D-057: Ridge normal-behaviour models and the O1 indicator
- Date: 2026-10-26
- Owner: Kanav
- Status: provisional; values below are proposals for team confirmation, not tuned on validation or test
- Context: O1 needs a degradation indicator from the gap between predicted and observed temperatures on healthy rows (`modelling_plan.md` 6). The plan left inputs, targets, smoothing and the healthy-row source as TODO.
- Decision: `models/nbm.py` fits one ridge per target on daily producing-interval means (`op_` columns of the D-021 feature table), with turbine offsets removed by a within-turbine fit so slopes are shared and each turbine keeps its own offset. Inputs: primary wind speed, generator rpm, main-shaft rpm, ambient and nacelle temperature. Targets: gearbox oil, generator bearings (G and R), main bearing, the three stator windings, transformer oil, hydraulic oil, IMS generator bearing. Ridge alpha 1.0 and a 7-day trailing smoothing window are fixed, not tuned. All values are in the `nbm` block of `configs/protocol.yaml`.
- Fit rows: healthy train days that are healthy for both target A and target B (D-030), so the model does not depend on the target choice (7,635 days; 5,718 usable after input and target availability). Test years are never read.
- Residual: `predicted - observed`, divided by the turbine's healthy train residual spread (fleet spread when a turbine has fewer than 100 rows). Positive means cooler than expected. A turbine unseen in training uses the fleet offset and spread.
- Indicator: largest absolute 7-day trailing smoothed residual over the sensors (needs 4 observed days, `features.window_min_days`). Looks backwards only; a test perturbs future rows and checks the smoothed value does not move.
- Results on validation years (`reports/tables/nbm_*.csv`, F32, F33): train R-squared from 0.11 (hydraulic oil) to 0.95 (windings). Standardised residual means on healthy-looking validation days range from -0.92 (main bearing) to +0.61 (gearbox oil), so there is drift between train and validation years. The indicator's mean is higher in each of the four windows before an episode than far from one (target A: 1.9 to 2.5 against 1.8; B: 1.8 to 2.7 against 1.6), but it does not rise as the episode nears and the medians are 1.4 to 1.6 everywhere. Treat it as a weak signal until the residual features are tested inside the O2 models.
- Not done: lagged inputs (thermal delay), per-sensor target pruning, LightGBM NBM, `PowerRef`, autoencoder. Residual columns are in `data/interim/features/nbm_residuals.parquet` and are not yet joined into the O2 feature table (that must pass `scripts/check_feature_leakage.py` first).
- Consequences: new figures F32 and F33 in `06_nbm`. Whether alpha, smoothing window and the sensor list are accepted is an open item for the team.
- Update 2026-10-27 (D-055): the validation numbers above are for the own-turbine residual. The saved columns are now fleet-relative, which removes most of the drift; F32 and F33 show both. The NBM is frozen.
- Renumbered 2026-10-27: this entry was first logged as D-054, an ID Farhan's alert-burden analysis (PR #15) already holds on `main`; all references were changed to D-057.

### D-055: Final-sprint evaluation protocol and the fleet-relative NBM residual
- Date: 2026-10-27
- Owner: Kanav (proposed; for the team to confirm)
- Status: decided for the sprint; written before any LightGBM or test-year result existed
- Context: five days remain to the submission (assumed Sunday 2026-11-01, not yet confirmed with the coordinator). Positive windows cover 42 to 49% of days, so a random score reaches recall 0.79 to 0.84 at the budget of one false alarm per turbine-month and no baseline beats it (D-020, F15). The O1 indicator drifted between train and validation years (D-057).
- Decision (O2 evaluation): the labels and windows of D-005 and D-019 stay as they are. Changing them would invalidate T0, the baseline tables and most of F01 to F33, and choosing a new window after seeing validation results would be selection on validation. Every model is reported with PR-AUC against the random-score floor, recall at the budget next to the random floor, event-level (episode) recall and median lead time, each with turbine-bootstrap intervals and positive counts (D-020). The headline claim is the comparison with the floor, not recall alone. A 7-day positive window is allowed only as a clearly labelled sensitivity run, and only if time remains; it is not a replacement and not tuned.
- Decision (test years): scored once, in a single `--frozen` run for configurations frozen the day before. No second test run and no tuning after seeing test numbers. Robustness runs use validation years only.
- Decision (NBM): compare own-turbine and fleet-relative residuals on validation years only, with the rule fixed in advance: adopt fleet-relative if the mean absolute validation mean of the standardised daily residual falls. Fleet-relative subtracts the same-day median residual of the turbines observed that day (at least `nbm.fleet_min_turbines`), so it looks at no later day. The rule is met: mean absolute validation mean z falls from 0.444 to 0.061 and the mean validation sd of z from 1.057 to 0.745 (`reports/tables/nbm_fleet_compare.csv`). `nbm.fleet_relative` is set to true and the NBM is frozen: no further changes to its inputs, targets, alpha or smoothing.
- Cost of fleet-relative: it also removes any fleet-wide change, and the pre-episode contrast shrinks in ratio terms. The indicator at 15 to 30 days before an episode against more than 60 days or none is 1.35 times (A) and 1.69 times (B) for own-turbine residuals, and 1.14 times (A) and 1.12 times (B) fleet-relative (`reports/tables/nbm_lead_variants.csv`, validation). Part of the larger own-turbine contrast is the drift. Both versions are weak and neither rises steadily as an episode nears.
- Leakage: the residual, smoothed residual and indicator columns, including the fleet median, were checked on real data with `scripts/check_feature_leakage.py --nbm` (9 random turbine-days, rows unchanged when later days are removed) and by a unit test that also catches a planted leak. The fitted NBM is held fixed in this check, as a scaler would be; it is fitted on train years only.
- Consequences: `models/nbm.py` gains `fleet_relative`, `nbm_features`, `fit_nbm` and `check_nbm_no_lookahead`; `scripts/build_nbm.py` writes both variants' validation tables and saves the chosen one; `scripts/run_baselines.py` joins `nbm_residuals.parquet` onto the feature table when it exists (other baselines ignore the extra columns). The NBM columns are not part of the stored `daily_features.parquet`.

### D-056: Class-weighted LightGBM (L1), first validation result
- Date: 2026-10-27
- Owner: Kanav
- Status: model built; hyper-parameters provisional until the validation grid is run and frozen (Day 2 of the sprint)
- Decision: `models/lgbm.py` implements `Baseline` (D-023). Inputs are the columns B4 uses (daily state, 7-day window means and standard deviations, alarm counts, fault history), plus the NBM residual columns for the variant `L1_lgbm_nbm`. Column selection (at least `preprocessing_p0.min_observed_share` observed, not constant), the positive-class weight (negatives over positives in train) and the trees all come from labelled train rows only; missing values go to LightGBM unchanged; seeded from `seed`; single thread and deterministic. Provisional settings in the `lgbm` block: 15 leaves, 200 trees, learning rate 0.05, at least 200 rows per leaf, row subsample 0.8, column subsample 0.5. The whole search is the grid in the same block (3 x 2 = 6 settings), chosen by validation PR-AUC with ties to the simplest setting. Registered as `L1_lgbm` and `L1_lgbm_nbm`.
- Result with the provisional settings, validation years, one run, nothing tuned (`reports/tables/baselines_a.csv`, `baselines_b.csv`): PR-AUC A 0.445 and 0.443 with NBM, B 0.447 and 0.440 with NBM, against the random floor 0.465 (A) and 0.483 (B) and R1 at 0.577 and 0.588. Recall at the budget is 0.13 and 0.11 (A), 0.20 and 0.22 (B), against a random floor of 0.78 and 0.76. B4 logistic is at the same level (0.447, 0.451).
- Diagnosis (train years only, no validation data used): PR-AUC on the training rows is 0.94 for target A but 0.475 when the model is fitted on 2016 to 2018 and scored on 2020, so the fit does not carry across years. The largest gain features are window means of pressure, humidity, tower-frequency and an internal temperature (`A21IntTm`); `DATA_TRAPS.md` records cross-year drift for some pressure and temperature sensors (`HydPress`, `MainBTmp`), but drift in these particular features was not tested, so the cause is not established.
- Consequences: the supervised models do not beat the random floor on validation with these features. This is reported as it is. The Day 2 grid also varies regularisation only; any change of feature set (for example drift-robust columns only) would be a separate, logged decision made on validation, not on test. If nothing beats R1 on validation, R1 stays the best model and L1 is reported as a model that does not generalise across years.


### D-054 — Validation alert-burden analysis after baseline integration
- **Date:** 2026-10-25
- **Owner:** Farhan
- **Decision:** Extend the D-025/F15 budget grid into reproducible tables for all available
  baselines, both targets and both T0 fleet variants. Move the existing grid into protocol.yaml.
  Reuse fixed validation scores and the shared strict threshold/episode evaluator. Rechoose
  thresholds separately at each budget; do not refit or retune models for this analysis.
- **Outputs:** Allowed and realised false-alarm rates, false-alert days, total labelled alert
  days, detected/missed evaluable episodes, lead time, prevalence and the random floor at the
  same target/fleet/budget. Counts are operational burden proxies, not monetary costs or work
  orders; no cooldown or new event-matching rule is introduced.
- **Limits:** Exploratory validation point estimates, no additional significance claim or
  budget selection. Keep the official D-009 budget at 1. Test years remain untouched. B3 is
  merged; R2 and main LightGBM are still absent, so extend these tables when they arrive.

### D-058: R2, the NBM indicator as an O2 baseline
- Date: 2026-10-27
- Owner: Kanav
- Status: decided; fixed before looking at the result
- Context: `docs/BASELINE_SPLIT.md` assigns R2 (ridge normal-behaviour model) to the baseline set, and the alert-burden notes (D-054) listed R2 as pending. The NBM (D-057) existed but its indicator was not a registered baseline, so it was missing from T0.
- Decision: `models/nbm_indicator.py` registers `R2_nbm_indicator`. The score of a turbine-day is its `nbm_indicator` (largest absolute 7-day smoothed standardised residual over the 10 NBM sensors, fleet-relative as frozen in D-055). Nothing is fitted in the baseline; the NBM was fitted on healthy train days in `scripts/build_nbm.py`. A day without an indicator scores `nbm.indicator_missing_score` (0.0), so every day with features is scored (D-022). `scripts/make_baseline_table.py` and `scripts/run_baselines.py` join the NBM columns through `models.nbm.attach_nbm`.
- Result (validation years; `reports/tables/t0_validation.csv`, full fleet): PR-AUC A 0.496 and B 0.533 against the random floor 0.465 and 0.483 and R1 at 0.577 and 0.588. Recall at the budget is 0.383 (A) and 0.458 (B) against a random floor of 0.78 and 0.76. So the indicator is above the random floor on PR-AUC but below R1 and below the floor on recall at the budget. Existing baselines are unchanged.
- Consequences: T0 and the alert-burden tables (D-054) were regenerated with R2 and the LightGBM models (D-056) included; the other models' numbers did not change, except B4 logistic in the fifth or sixth decimal place (solver noise between reruns, no effect on any reading). Figures with a fixed model list (F15 to F26) do not show R2 or L1 yet.

### D-060: LightGBM settings frozen after the validation grid (G3)
- Date: 2026-10-29
- Owner: Kanav
- Status: decided; the whole search and the selection rule were fixed in D-056 before the grid ran
- Decision: `scripts/tune_lgbm.py` fitted all 6 settings in the `lgbm.grid` block (3 leaf counts x 2 tree counts, learning rate 0.05, at least 200 rows per leaf) for each target on train years and scored validation. Selection used `L1_lgbm` (without NBM columns) and the highest validation PR-AUC, ties to the simpler setting. Frozen in `lgbm.selected`: target A 7 leaves and 300 trees, target B 31 leaves and 300 trees. `L1_lgbm_nbm` uses the same settings. The grid is not widened and nothing else is tuned. The grid used 100 bootstrap resamples, which do not affect the PR-AUC point estimates it selects on; reported intervals use the protocol's 1,000.
- Result (validation years, `reports/tables/lgbm_tuning.csv`): the best PR-AUC is 0.460 for A and 0.456 for B, against the random floor 0.465 and 0.483 and R1 at 0.577 and 0.588. The 6 settings span 0.441 to 0.460 (A) and 0.440 to 0.456 (B) for `L1_lgbm`; none reaches the floor. Adding the NBM columns does not help (best 0.449 for A and 0.454 for B). Recall at the budget is 0.11 to 0.24, against a random floor of 0.78 and 0.76.
- Consequences: gate G3 is passed for the LightGBM models: the configurations are frozen and no further tuning is allowed. L1 is not better than random on PR-AUC on validation, so the report states that supervised models on these features do not generalise from the train years to the validation years (D-056 diagnosis: in-sample PR-AUC 0.94 against about 0.45 held out). R1 remains the best model on validation. The single test-year run scores the frozen L1 models as they are, alongside the other baselines, and is not repeated.

### D-061: Single frozen test-year run
- Date: 2026-10-29
- Owner: Kanav
- Status: done once; not to be repeated or followed by any change to a model, setting or threshold
- Decision: `scripts/run_baselines.py --target both --frozen` was run once from commit `78755ba` (clean tree) for every registered model and both targets, full fleet only. Models were fitted on 2016 to 2020 with the settings frozen in D-060 and the earlier decisions; thresholds were chosen on validation and carried to the test years (2023 to 2026-04). The sensitivity variant without turbine 2304515 was not scored on test, to avoid a second look at the test years with a different fit. `scripts/write_test_results.py` copies the test rows into `reports/tables/test_results.csv` and `reports/TEST_RESULTS.md`; the full runner table, with validation and test rows, is `reports/tables/baselines_a.csv` and `baselines_b.csv`, and per-day test scores are in `data/interim/scores/` (git-ignored).
- Result: test prevalence is 0.451 (A, 17,947 labelled days, 336 evaluable episodes) and 0.505 (B, 17,281 days, 373 episodes). No model's PR-AUC interval excludes the random-score floor (0.450 for A, 0.506 for B). The highest point estimates are LightGBM, A 0.492 [0.441, 0.540] and B 0.556 [0.488, 0.630], and B4 logistic, A 0.462 and B 0.563 [0.503, 0.627]. R1, the best model on validation (0.577 and 0.588), falls to the floor on test (0.448 and 0.499), so its validation lead did not hold. The supervised models were below the floor on validation (D-056, D-060) and are above it in point estimate on test, within the intervals. No model reaches the random-score episode recall (0.795 for A, 0.839 for B) at a comparable false-alarm rate: R1 and R2 reach recall 0.62 and 0.55 (A) and 0.68 and 0.64 (B) but at 2.1 to 2.5 false alarms per turbine-month, about twice the budget, because the validation threshold does not carry to the test years. Median lead times (17 to 30 days) are close to the random score's 21.
- Consequences: the claim for the report is that, on these features and labels, none of the models is distinguishable from the random-score floor on the test years with 95% turbine-bootstrap intervals; the ranking seen on validation does not hold on test; and the validation threshold does not transfer to a fixed false-alarm rate. Intervals do not include the uncertainty of model or threshold selection. These are inferred operational episodes, not confirmed failures. SHAP and the robustness runs use validation years only and do not touch the test years again.

### D-062: SHAP and robustness of the frozen LightGBM (validation years)
- Date: 2026-10-29
- Owner: Kanav
- Status: done; validation years only, after the single test run (D-061), which was not repeated
- Decision: all runs use the frozen `L1_lgbm` settings (D-060) and the validation labelled rows; no test data is read. Sensor groups are defined in `robustness.sensor_groups` of `protocol.yaml`: pressure (13 sensors), drivetrain temperature (15), power electronics (10) and wind and rpm (5 sensors plus the wind, rpm and operating-state summaries); a column belongs to a group if its sensor is listed, and a column in two groups is an error. Dropout sets a group's feature columns to missing at scoring time (the model is not refitted) and evaluates at the threshold chosen on the intact validation scores. Leave-one-turbine-out refits without one turbine's train rows and scores that turbine's validation rows; all 21 turbines are used. SHAP is `TreeExplainer` on 4,000 seeded validation rows; its stability is the rank correlation of mean absolute SHAP, and the overlap of the top 20, between the full model and the 21 leave-one-turbine-out refits on a shared 1,000-row sample (`explain` block).
- Result (`reports/tables/robustness_*.csv`, `shap_*.csv`, `nbm_attribution.csv`; F34, F35): dropping any group changes validation PR-AUC by at most 0.023 (A: 0.460 intact; B: 0.456). Recall at the fixed threshold changes more, but with the false-alarm rate: for target B, dropping the pressure group raises recall from 0.219 to 0.458 while false alarms rise from 1.0 to 2.9 per turbine-month, so it is a score shift, not skill. Leave-one-turbine-out pooled PR-AUC is 0.467 (A) and 0.447 (B) against 0.460 and 0.456 for the full fit and pooled prevalences of 0.470 and 0.492. Per-turbine PR-AUC ranges from 0.24 to 0.99 and follows each turbine's own prevalence (0.26 to 0.97); every turbine has at least 113 positive validation rows. The largest SHAP features are ambient and weather-linked inputs (DeltaTmp, ambient temperature, tower humidity and frequency), pressure sensors (inverter, offline and hub pressure) and days since the last fault day; component temperatures such as MainBTmp and HydOilTm rank lower. The SHAP ranking is stable across folds: median rank correlation 0.86 (A) and 0.93 (B), median top-20 overlap 18 of 20. On the NBM, the sensor with the largest smoothed residual on a validation day is most often GenBeRTm (20%), GenBeGTm (14%) and TraOilTF (14%), and the shares on label-1 and label-0 days differ by at most 3 percentage points, so the indicator does not point to a component before episodes.
- Limits: SHAP describes how the model uses its inputs, not what causes faults, and this model does not beat the random-score floor (D-060, D-061), so a stable attribution is not evidence of a useful signal. Several top features track season and weather. Dropout intervals were not bootstrapped; changes below about 0.03 PR-AUC should not be read as effects. Dropout recall must be read with the false-alarm rate. Turbine 2304515 is included in leave-one-turbine-out. The explanations use validation rows, not the test years.
- Consequences: gate G4 is met for robustness and SHAP; the upgrade analysis stays cut (AeroUp dates absent). New figures F34 and F35 in `07_explain_robustness`.

### D-063: Final report rewrite, README and stakeholder summary
- Date: 2026-10-29
- Owner: Kanav
- Status: drafted; items not yet confirmed are marked in red (`	bc`) in the report
- Decision: the report in `reports/report/main.tex` is rewritten to match what the project did and measured: the target is fault and downtime episodes (targets A and B, 30-day horizon), not generator alarms; objectives O1 to O4 follow the project (indicator, prediction, robustness, explanation); the horizon sweep, random forest and precision are removed because they were not done; results come from the test run (D-061), the freeze (D-060) and the robustness and SHAP runs (D-062). The title is changed accordingly. The LaTeX tables are generated by `scripts/write_report_tables.py` from `reports/tables/*.csv`, so the numbers trace to scripts. The citations and the Background and Related Work claims are carried over from the earlier draft and are marked in red as unchecked, following the rule to cite only verified sources. `README.md` is refreshed (status, findings, pipeline commands, repository map) and `reports/STAKEHOLDER_SUMMARY.md` gives the findings in plain language.
- Limits: the report could not be compiled here (no LaTeX installation), so it has been checked only for balanced environments, references and included files; `refs.bib` and the logo image are not in the repository.
- Open items (red in the report): author lines, dataset version (D-004), the number of temperature fields, the publishers' stated intended use and identifier removal, the upgrade names and dates, hardware and run times, cost figures for the threshold, whether a train-years year-shift run is added, the ethics review, and the citations.
- Update 2026-10-29 (report rewrite after D-064 to D-068): the report is restructured so the method comes before the results. A study-design section (pipeline and split diagram, safeguards), a methods subsection for the pre-registered exploratory protocol with a table of the four rounds, a results section ordered main run, O1, O3, O4, exploratory rounds, and a single limitations section grouped by data, labels, evaluation, modelling and process replace the earlier scattered notes. New figures F36 to F39 (validation against test, pruning, label sensitivity, turbine ranking) are registered in `reporting/figures_explore.py`; existing figures F01, F03, F07, F09, F10, F12 and F21 are now used. Methods not tried (autoencoder or PCA reconstruction error, isolation forest, survival models) are listed as future work with their citations still to be verified (red). `scripts/export_report.py` writes a standalone copy with tables inlined. No result changed.

### D-064: Exploratory validation-only runs: drift-robust features and low-capacity models
- Date: 2026-10-29
- Owner: Kanav
- Status: pre-registered before any of these models was run; exploratory; not part of the frozen results
- Context: the diagnosis in D-056, D-060 and D-062 is year-to-year drift (in-sample PR-AUC 0.94 against about 0.45 held out), a broad label, and models that lean on weather-linked inputs. D-061 used the test years once, so they cannot be used again to compare new models.
- Decision: run, on validation years only, (1) drift-robust inputs and (2) low-capacity models, and report them as future work, exploratory, not tested on the test years. Nothing here changes a frozen result, a protocol setting for the main models or the test table.
  - Feature sets: `raw` (as now); `fleet`: each sensor-derived window feature minus the median of the same feature over the turbines observed on the same day (at least `exploratory.min_turbines`; same-day only, so no look-ahead); `fleet_season`: the same, then minus the month-of-year mean of that feature learned on the train years only. Alarm counts, fault history and operating-state fractions are left as they are.
  - Models: L1 LightGBM with the frozen D-060 settings; LightGBM with very small trees (3 leaves, depth 2, 500 rows per leaf, 50 and 150 trees); L1-penalised logistic regression (C 0.003, 0.01, 0.03; train-only scaling and imputation as in B4); B4 logistic (frozen C). Every combination in the grid is run and reported; none is selected.
  - Evaluation: (a) the standard validation run through the shared runner (PR-AUC with turbine-bootstrap intervals, recall at the budget, thresholds from validation); (b) a year-shift check on train years only: fit on labelled days up to 2018, score labelled days of 2020 (PR-AUC against that year's prevalence). Part (b) is also the roadmap's year-shift stress test.
- Limits: 42 configurations are compared on two validation years, so the best of them is optimistically biased; validation also failed to predict test performance for R1 (D-061). A configuration that looks better here is a lead for a new pre-registered study with fresh data, not a result. No model is chosen, tuned further or scored on the test years.
- Result (`reports/tables/exploratory_validation.csv`, 42 configurations, validation years; the random-score PR-AUC is 0.465 for A and 0.483 for B): making the sensor features fleet-relative raised validation PR-AUC for the logistic models and, for target B, for the tree models: for A, B4 logistic 0.447 to 0.483 and sparse logistic (C=0.01) 0.455 to 0.478, while L1 LightGBM did not improve (0.460 to 0.452); for B, L1 LightGBM 0.456 to 0.516, B4 logistic 0.451 to 0.516 and the very small trees 0.433 to 0.529. Also removing the month-of-year mean (`fleet_season`) lowered PR-AUC again in almost every case. The highest value is 0.529 [0.440, 0.618] (B, very small trees, 50 trees) and the highest for A is 0.483 [0.416, 0.569] (B4 logistic); every interval contains the random-score value. Recall at the budget stays between 0.10 and 0.32 against a random-score recall of 0.78 (A) and 0.76 (B). Year-shift check on train years (fit up to 2018, score 2020; prevalence 0.412 for A and 0.399 for B): with raw features every model scores above the prevalence (A 0.416 to 0.466, B 0.498 to 0.529), and the fleet-relative sets give no consistent gain (for example A L1 LightGBM 0.466 to 0.432, B B4 logistic 0.529 to 0.543).
- Reading: the validation gain from fleet-relative features is a lead, not a result. 42 configurations were compared on two validation years, no interval excludes the random-score value, and the year-shift check on the train years does not confirm a gain. Nothing here changes the frozen results (D-060, D-061) and no model was scored on the test years.

### D-065: Second exploratory round: rolling self-baseline features and a turbine-ranking metric
- Date: 2026-10-29
- Owner: Kanav
- Status: pre-registered before these runs; exploratory; validation years only; not part of the frozen results
- Context: D-064 found that fleet-relative features lift some validation scores without any interval excluding the random-score value. Two further ideas remain that need no new data: a baseline that adapts to each turbine's own recent history, and a metric that matches how a planner would use a score.
- Decision: (1) `self`: each sensor-derived window feature minus the median of the same turbine's own values over the previous `window_days` days (the window ends the day before, so a day never contributes to its own baseline; at least `min_days` observed days, else missing); `self_fleet`: the same applied to the fleet-relative values of D-064. Both use the past only. The same seven models as D-064 are run on both feature sets (28 configurations), none is selected and nothing is tuned. (2) Turbine ranking: each day, rank the turbines that have a label that day (at least `min_turbines`) by score, and measure precision among the top 3 and top 5 against the chance level, the share of positive labels among those turbines that day. Differences are averaged over days, with a 95% interval from resampling calendar months. Ties are broken at random with the protocol seed. The ranking metric is computed for the main models (R1, R2, B1 to B4, LightGBM, random) from the saved validation scores and for the 28 new configurations.
- Limits: a descriptive validation-only analysis on two years. The ranking metric uses overlapping daily windows, so its intervals are optimistic about independence even with month resampling. Nothing here changes a frozen result or scores the test years. The test years are not ranked either, because that would be a new look at them.
- Result, self-baseline features (`reports/tables/exploratory_self_baseline.csv`, 28 configurations, validation years; random-score PR-AUC 0.465 for A and 0.483 for B): validation PR-AUC ranges from 0.445 to 0.516. The `self_fleet` set gives the highest values for the logistic models (A: B4 logistic 0.495, sparse logistic 0.494 to 0.501; B: 0.501 to 0.516), `self` alone is mixed (A: 0.454 to 0.508; B: 0.445 to 0.484). No PR-AUC interval of the 28 excludes the random-score value (the highest, B sparse logistic C=0.01 on `self_fleet`, is 0.516 [0.441, 0.594]). The train-year year-shift check does not favour the new sets: for target B it is lower than with raw features for almost every model (for example B4 logistic 0.529 to 0.478 and 0.467), for target A it is mixed (0.416 to 0.485).
- Result, turbine ranking (`reports/tables/ranking_validation.csv`, validation years, daily top 3 and top 5 against chance): for the main models, the B3 power-curve baseline puts positive turbines in the top three more often than chance for both targets (A +0.082 [0.040, 0.120], B +0.071 [0.023, 0.123]; top five +0.060 and +0.052, all intervals above zero), the B2 alarm-count rule does so for target B (+0.032 [0.008, 0.057]), and the R1 z-score does worse than chance (A -0.063 [-0.105, -0.022]); all other main-model intervals include zero. For the 28 new configurations, 8 of 56 intervals exclude zero (4 of 28 at the top three), all above zero and all for target B: very small trees on `self` (+0.076 and +0.059) and sparse logistic on `self_fleet` (+0.064 and +0.074).
- Reading: 92 ranking intervals were examined (36 for the main models, 56 for the new configurations), so some exclude zero by chance; one random score did (B, top three: -0.018 [-0.034, -0.002]). The consistent positive result for B3 on both targets and both k is a lead worth confirming on new data, not a finding. No PR-AUC interval excludes the random-score value, so the PR-AUC conclusion of D-061 is unchanged. Nothing here changes a frozen result, the test years were not scored or ranked, and no configuration was selected.

### D-066: Third exploratory round: rank the features and prune the least useful
- Date: 2026-10-29
- Owner: Kanav
- Status: pre-registered before these runs; exploratory; validation years only; not part of the frozen results
- Context: the models carry about 300 candidate columns (daily state, fault history, alarm counts, 7-day means and standard deviations) and overfit year-specific levels (D-056, D-062). Fewer, more stable columns may generalise better. Feature importance was only looked at after fitting (SHAP, D-062); this round uses it to remove columns and measures what happens.
- Decision: three train-only rankings, then keep the top k columns (k = 10, 25, 50, 100, all run). `gain_cv`: LightGBM gain share averaged over leave-one-train-year-out fits (frozen D-060 settings), so a column must matter in every year to rank high. `year_stable_auc`: per train year, the AUC of each column on its own against the label; a column qualifies only if its AUC sits on the same side of 0.5 in every train year, and qualifying columns are ranked by the weakest year's distance from 0.5. `gain_cv_decorr`: `gain_cv` order, skipping a column whose absolute Spearman correlation with an already kept column is above `decorrelate_above` (computed on a train sample). Models: L1 LightGBM (frozen settings) and B4 logistic, on the `raw` and `self_fleet` feature sets, with selection refitted inside every `fit` on its labelled train rows (so the year-shift check and any fit are honest). Also, per target, refit with one whole group removed (the four sensor groups of `robustness.sensor_groups`, alarm counts, fault history, day state) to see which families carry signal. 96 pruning configurations (both targets) and 28 group-removal configurations on the raw set; none is selected, nothing is tuned.
- Limits: 124 configurations on two validation years, so the best is optimistically biased and no one of them should be frozen. Gain importance is attribution, not causation, and correlated columns share credit. A pruned model that matches the full model is a simplicity result, not an accuracy result. Nothing here changes a frozen result, and the test years are not scored.
- Result, pruning (`reports/tables/exploratory_selection.csv`, 132 configurations including the all-columns references, validation years; random-score PR-AUC 0.465 for A and 0.483 for B): validation PR-AUC ranges from 0.404 to 0.530, and no interval excludes the random-score value (the highest, B4 logistic on `self_fleet` with `gain_cv_decorr` top 50 for target B, is 0.530 [0.448, 0.615]). Against the same model with all columns, the 96 pruned configurations changed validation PR-AUC by -0.051 to +0.027 (mean -0.002; 44 better, 52 worse), and lowered the train-year year-shift score in 79 of 96 (mean -0.021). By method the mean change was +0.002 (`gain_cv`), +0.003 (`gain_cv_decorr`) and -0.009 (`year_stable_auc`); by size it was -0.002 (10 columns), -0.006 (25), +0.001 (50) and +0.001 (100).
- Result, group removal (28 refits on `raw`): removing one family changed validation PR-AUC by -0.030 to +0.021 against the all-columns model (10 of 28 higher). Mean change by family: drivetrain temperature +0.008, wind and rpm +0.003, pressure +0.002, alarm counts -0.004, day state -0.004, fault history -0.005, power electronics -0.007. All within the noise of two validation years.
- Result, what the rankings show (`reports/tables/feature_ranking_a.csv`, `feature_ranking_b.csv`; `raw`, train years only): the top columns by cross-year gain are days since the last fault day, the inverter and hub pressures (11 of the top 25 for both targets), and tower frequency, humidity and internal temperatures (A). The 25 highest columns carry 59% (A) and 52% (B) of the gain, and the columns outside the top 50 carry 24% and 31%. Only 31 columns for A and 85 for B have a single-column AUC on the same side of 0.5 in every train year. For B the most year-stable columns are ambient, nacelle and hub temperatures (weakest-year distance from 0.5 of 0.13 to 0.15), which move with the season, so they may track weather rather than turbine wear. Only 5 of the top 25 by gain are also in the top 25 by year-stability, for both targets.
- Reading: pruning to 50 or 100 columns costs nothing measurable and pruning to 10 or 25 does not help, so the model can be made simpler without a clear loss, but no feature subset produces a result beyond the random-score margin. The two ranking methods disagree about which columns matter, and the year-stability filter lowered validation and year-shift scores, so year-stable single columns are not the useful ones. The best of 132 is optimistically biased and is not a selection; nothing frozen changed and the test years were not scored.

### D-068: Label sensitivity: other horizons, severities and a fault-only target B
- Date: 2026-10-29
- Owner: Kanav
- Status: pre-registered before these runs; exploratory; validation years only; the frozen labels and results are unchanged
- Context: about half of all labelled rows are positive (prevalence 0.45 to 0.50) because any episode in the next 30 days counts, target A counts any day with 4 hours of time-error, and target B counts every stop, including weather stops. Three rounds of model changes (D-064 to D-066) did not move any interval beyond the random-score floor, which points at the labels as the bottleneck. This round asks whether more specific labels make the signal easier to learn, without changing a frozen setting.
- Decision: rebuild the labels in memory with the existing builders (`fault_episodes_a`, `stop_runs`, `merge_into_episodes`, `build_horizon_labels`) under a changed copy of the protocol, never overwriting the frozen label files. Variants: horizon (positive window 7, 14 or 30 days, exclusion zone ending at twice the window, as 30 and 60 now); target A fault-day threshold 4, 8 or 12 hours of time-error; target B stop length 18, 36 or 72 consecutive intervals (3, 6 or 12 hours); and for target B, whether stop runs that start with a weather alarm active (tier 2: low wind, high wind, icing) are kept or dropped. That is 9 variants for A and 18 for B. The reference variant (30 days, 4 hours, 18 intervals, keep) must reproduce the frozen label files exactly or the run stops. The features (daily table) are unchanged, including the fault-history features, which keep their 4-hour definition. Models: B1 recent fault, B2 alarm count, R1 z-score, B3 power curve, B4 logistic, R2 NBM indicator and L1 LightGBM, with every setting as frozen (nothing is retuned for the new labels), scored on the validation years with the usual runner. Each variant is judged against its own random-score floor and its own prevalence; I report PR-AUC with the turbine-bootstrap interval, the excess over that floor, recall at the false-alarm budget, positives and episodes, and the daily top-3 ranking difference (D-065). No variant is chosen as the new target by this run.
- Limits: a validation-only analysis on two years with 27 variants and seven models, so some intervals will exclude the floor by chance. Rarer labels have fewer positives and wider intervals. A variant that looks better is not a better model of the original question, it is a different question (an easier or narrower one), and a lower prevalence changes the floor, which is why every variant is compared with its own floor. The models are frozen on the 30-day labels, so a shorter horizon is partly a mismatch for them. Adopting any variant would need a new decision, a frozen protocol and a fresh look at the test years, logged as post-hoc, or new data.
- Result (`reports/tables/label_sensitivity.csv`, 27 variants, validation years, seven models and two floors each; the reference variant reproduced the frozen label files): the 189 model rows gave 37 PR-AUC intervals above their variant's random-score floor and none below; the random score itself was above its floor in 0 of 27. The 37 are concentrated in simple rules: B3 power curve 11 of 18 for target B and 0 of 9 for A; B1 recent fault 3 of 9 (A) and 9 of 18 (B); R1 z-score 3 of 9 and 6 of 18; B2 alarm count 2 of 9 (A); R2 NBM indicator 2 of 18 (B). The trained models gained almost nothing: L1 LightGBM 1 of 27, B4 logistic 0 of 27, with median lifts over the floor of 1.05 and 1.15 (LightGBM) and 0.99 and 1.07 (B4) for A and B.
- Result, does a more specific label help? The median lift (PR-AUC over its floor, all models) rises with severity, from 1.05 (4 hours) to 1.06 (8) to 1.18 (12) for target A and from 1.12 (18 intervals) to 1.18 (36) to 1.25 (72) for B; it is higher for B when weather stops are dropped (1.26 against 1.11). The horizon has no clear effect for A (1.10, 1.12, 1.09 for 7, 14 and 30 days) and a small one for B (1.20, 1.17, 1.11). The highest lifts sit in the rarest variants, where the intervals are widest (B3 on 72-interval stops with a 7-day window: 0.137 against 0.029, interval [0.033, 0.300]).
- Result, ranking (daily top 3 against chance): 44 of 189 intervals are above chance and 9 below; the random score was above chance in 0 of 27 and below in 5. B3 power curve is above chance in 4 of 9 variants for A and 16 of 18 for B (median differences +0.033 and +0.053), B1 recent fault in 11 of 18 for B, L1 LightGBM in 2 and 6, and the R1 z-score is below chance in 1 variant for A and 8 for B (median -0.011 and -0.014).
- Reading: the labels are not the main bottleneck for the learned models. Narrower labels leave LightGBM and the logistic model at the floor, and the only things that stand out are two simple rules: B3 (on the ranking view, above chance in 16 of 18 target B definitions and 4 of 9 for A) and B1, which says that a turbine that recently faulted tends to fault again, so it measures persistence of the same fault process more than early warning. R1 has the highest validation PR-AUC at the reference labels but ranks turbines worse than chance, and it did not hold on the test years (D-061), so validation PR-AUC above the floor is not enough. The 27 variants overlap heavily (same days, nested labels), so the 37 and 44 counts overstate independent evidence; with 189 intervals some would clear the floor by chance, and nothing was selected. Nothing frozen changed and the test years were not scored. If the team wants to act on B3, the next step is a fresh, pre-registered check on data not used here, not a new look at the test years.
