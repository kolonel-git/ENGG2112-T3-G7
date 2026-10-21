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
- Date: 2026-10-21
- Owner: Kanav
- Status: open
- Decision pending: minimum `TimeErr` hours for a fault day (4 h used in exploration), fault-free gap that separates episodes (`targets.daily_fault_time`). To be fixed after a sensitivity check on train and validation years only and logged here.

### D-008: Target B parameters
- Date: 2026-10-21
- Owner: Kanav
- Status: open
- Decision pending: SCADA stop definition (maximum generator rpm, minimum wind speed, minimum consecutive intervals, episode gap; `targets.scada_stops`). To be fixed by checking agreement with `TimeErr` on train years and logged here.

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
- Status: decided
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
- Decision: alarm events join to SCADA with `ceil` to 10 min. `parse_times` accepts ISO (with or without time) or `M/D/YYYY H:MM` and raises on anything else. SCADA keys are de-duplicated on (StationId, TimeStamp).

### D-014: Sentinel handling
- Date: 2026-10-21
- Owner: Kanav
- Status: decided (implementation pending)
- Decision: values `200`, `-50`, `-100` in sensor columns are masked to NaN per column, with a boolean flag column per affected sensor; LightGBM uses NaN natively, linear models use train-fit imputation. Outage months (2022-08, 2022-09, 2025-09) are flagged and excluded from healthy-row fits.

## Open decisions
- Dataset version: confirm v2.1.0 with the teammate who downloaded the data (D-004).
- Licence: none chosen; README carries a TODO.
- `TimeErr` meaning: fetch `Hill_of_Towie_data_dictionary.md` from Zenodo (small metadata file; needs the user's go-ahead to download) and confirm.
- Target A and B parameters (D-007, D-008).
- Handling of the possible anemometer faults (turbine-years 2304511 in 2022 and 2025, 2304520 in 2023, 2304514 in 2022): include with a flag or exclude.
- Whether the 2022 drop of alarms 20 and 25 (about 60%) reflects a controller change; relevant to alarm-count features across years.
- Upgrade analysis needs `Hill_of_Towie_AeroUp_install_dates.csv` from Zenodo.
