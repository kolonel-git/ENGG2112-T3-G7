# Modelling plan

**Status 2026-10-21: planning closed.** The current plan is this summary, `docs/roadmap.md` (schedule, owners, gates) and `docs/DECISIONS.md` (D-001 to D-014).
Sections below are kept as the working history. Where they conflict with this summary, this summary and the decision log win; superseded sections are marked.

## Current plan (v2)

- **Data:** raw tables 2016 to 2026-04 (D-004). Alarms from raw `tblAlarmLog` only. `TimeStamp` is the interval end, daily rows the day start (D-013). Cleaning rules and flags in D-015, columns in D-016, wind speed from the primary channel (D-017). Measured active power and 10-min error time exist in `tblSCTurGrid` and `tblSCTurFlag` but are not in `data/raw` yet (D-018).
- **O2 targets (D-003):** A = fault-time episodes from `tblDailySummary.TimeErr`; B = unplanned stops derived from 10-min SCADA. Both trained and reported. Thresholds are open (D-007, D-008).
- **Label geometry (D-005):** daily prediction time; positive if an episode starts in the next 30 days; 30 to 60-day zone and rows already in an episode dropped; 7-day feature window.
- **Alarm codes (D-001 to D-003):** the 12 documented codes are features (counts, time since last event, deviation from the 5-day pitch-lubrication schedule), not targets.
- **Split (D-006):** train 2016, 2017, 2018, 2020; validation 2021, 2022; test 2023 to 2026-04. Leave-one-turbine-out only in the robustness stage.
- **Models:** prevalence dummy, recent-fault rule, rpm-vs-wind curve proxy for the power curve (D-011), class-weighted logistic regression, class-weighted LightGBM, ridge and LightGBM normal-behaviour models for O1, optional autoencoder, SHAP.
- **Metrics (D-009):** PR-AUC, recall at 1 false alarm per turbine-month, event-level median lead time, bootstrap over turbines.
- **Seed (D-010):** 42, read from `configs/protocol.yaml`.


Based on `data/processed/00-Processed-Data/turbines_2018.xlsx` (+ `.cleaning.json`): a 2018 snippet, not the full dataset.
All numbers below were computed from that file. Anything marked TODO is a team decision, not a finding.
No data has been edited; every data step below is a planned step to be implemented in `src/`.

## 0a. Raw data exploration (2026-10-21), supersedes sections 0 to 2 where they conflict

Scope explored: `data/raw` 2016 to 2026-04 (590 CSVs, 34 GB): `tblAlarmLog` (6,394,219 events), `tblSCTurbine`, `tblSCTurTemp`, `tblSCTurPress` (10.1M turbine-intervals after joining), `tblDailySummary`. Rebuilt alarm counts match `Alarm_Codes.csv` exactly for all 8 occurring in-scope codes, 2016 to 2019, so the loader is validated.

**Tier 1 alarm codes behave like routine operations, not faults (inference from the data, to confirm with the dataset authors):**
- 3130 (pitch lubrication): median gap between events 5.0 days (p10 4.8, p90 5.9; CV 0.29); 672 to 721 events per turbine over 10 years; 8.1 to 9.0% of events in every calendar month; duration 3.3 min median, 11 min p99. This is a near-fixed schedule, so presence in a 30-day window is 99.8% positive and the only unpredictable part is delays (gaps over 7 days: 2.3 per turbine-year).
- 10105 (untwisting cables): duration 9.2 min median (p99 10.1); 2.5% of events in January vs 16.5% in July; 55.8% of events occur in hours where at least 5 turbines log it (fleet-wide, wind-direction driven). Driven by yaw cable twist, not component wear.
- Consequence: Tier 1 as the O2 degradation target is not supported. Label rates under a 30-day window: 3130 99.8%; 10105 84%; 1005 54%; 8000 17%; 8230 14%. With a 30-day quiet gap (episode start) 3130 drops to 0.4%, i.e. almost no signal.
- Tier 2: 1005 (low wind), 8000 (high wind, 82.7% of events in hours with 5+ turbines, 55% in Dec to Feb) and 8230 (icing, Nov to Mar) are weather-driven. 8210, 8234, 8235, 8236 never occur in any year.
- Tier 3: 20 and 25 (about 2.5M each) are pure cut-in/cut-out counts; 102 (ice detection) has 265 events, 93% in January.

**Alternative O2 target found: `tblDailySummary` fault time.** `TimeErr` (hours per day, name suggests time in an error state; meaning to verify, the data dictionary is not in the repo) is above 4 h on 2% to 7% of turbine-days (peak 7.4% in 2022). Runs of such days give 5.8 episode starts per turbine-year, median 32 days apart (p10 4, p90 167), and a 30-day window is positive 33% of the time: a balanced, non-trivial target. Other candidates: stops derived from SCADA (rpm near zero while wind is above cut-in and `PowerRef` is not curtailed), and the 578 undocumented alarm codes (1.34M events, 21% of the log; largest 55, 50346, 115, 127, 111), checked for co-occurrence with real stops.

**Raw data is better than the processed workbooks:** raw SCADA covers 2016 to 2026-04 with ~99% grid coverage (except 2019: turbine table only, Jan to May and Jul), while the workbooks omit 2018-01/02, 2022-01 to 2022-04-04, and all of 2016, 2017, 2019 and 2026. Raw has yaw/nacelle position (`NacelPos`, `YawPos`, full 0 to 360 range, almost no NaN), pitch positions (`PitcPosA/B/C`), a secondary anemometer, and from 2022-04 `WTOperSt` (operating state) and wind direction. It does **not** have measured active power at 10-min resolution: only `PowerRef_endvalue`, plus daily energy and mean power in `tblDailySummary`.

**Other facts that change the build:** `TimeStamp` marks the END of its interval (alarm join uses ceil); fleet is very homogeneous (median producing-state temperatures differ by 0.2 to 1.1 C across turbines), so pooled models and leave-one-turbine-out are reasonable; cross-year drift exists (`HydPress` median 234.9 bar in 2016 to 2017, 219.4 in 2018, ~225 later; `MainBTmp` +5 C from 2016 to 2022); turbine-months with a flatlined sensor are rare (1 of 2,333).

## 0. Update 2026-10-21: alarm reference files received

Sources: `Alarm_Codes.csv`, `Alarm_Codes.txt`, `Processed_Data_Description.txt` (repo root, from the teammate who built the processed data). These change the plan:

1. **The processed xlsx lost almost all alarms.** Cleaning removed alarm-only rows (no sensor values), so only alarms landing exactly on a 10-min boundary survive: 1,331 of 706,572 raw events for Mar to Dec 2018 (~0.2%). Every section below that uses snippet alarm counts (section 1 alarms bullet, section 2 counts, 2.1) is superseded by the raw counts here. **Alarm labels must come from the raw `tblAlarmLog`**, joined to the SCADA grid by flooring `TimeOn` to 10 min.
2. **Tier 1 is not rare in the raw logs** (earlier concern withdrawn). Raw counts 2016 to 2019 (`Alarm_Codes.csv`):

| Code | Tier | 2016 | 2017 | 2018 | 2019 | Total | Median duration |
|------|------|-----:|-----:|-----:|-----:|------:|-----------------|
| 3130 | 1 | 1,494 | 1,457 | 1,485 | 1,495 | 5,931 | 3.5 min (max 20 min) |
| 10105 | 1 | 645 | 403 | 749 | 703 | 2,500 | 9.2 min (max 19 min) |
| 1005 | 2 | 1,495 | 1,131 | 1,401 | 1,160 | 5,187 | 2.0 min (max 2.0 d) |
| 8000 | 2 | 646 | 224 | 430 | 174 | 1,474 | 14 min (max 5.3 d) |
| 8230 | 2 | 51 | 120 | 39 | 144 | 354 | 2.1 h (max 2.9 d) |
| 102 | 3 | 0 | 0 | 8 | 0 | 8 | instant |
| 20 | 3 | 427,896 | 199,096 | 314,821 | 297,890 | 1,239,703 | instant |
| 25 | 3 | 427,895 | 199,096 | 314,821 | 297,893 | 1,239,705 | instant |
| 8210, 8234, 8235, 8236 | 2 | 0 | 0 | 0 | 0 | 0 | never seen 2016 to 2019 |

3. **New problem: Tier 1 events are frequent and short.** 3130 is ~71 per turbine-year (2018: 1,485 / 21), about one every 5 days; 10105 is ~36 per turbine-year, about one every 10 days; 1005 about one every 5.5 days. If events were spread evenly, a 30-day window would hold ~5.8 (3130), ~2.9 (10105) and ~5.5 (1005) events, i.e. ~99.7%, ~94.5% and ~99.6% of windows positive. Real events are likely clustered, so the true rate is lower, but it must be measured on the raw log before choosing the label. The plain "any event in the next 30 days" label is probably saturated. Options (TODO decide, log in `DECISIONS.md`):
   - **Episode label (recommended):** merge events into episodes with a refractory gap (`labels.episode_gap_days`, TODO) and predict only episode starts, i.e. the first event after a quiet period. This is also a natural reading of the 30 to 60 day exclusion window. `group_episodes` in `data/alarms.py` implements the merge.
   - **Count/rate label:** predict an unusually high event count in the next 7 or 30 days (threshold from train).
   - **Duration filter:** only count stops longer than a threshold (3130 and 10105 are all under ~20 min, so this suits 1005, 8000 and 8230 better).
4. **Raw logs in these files are 2016 to 2019 only; processed SCADA covers 2018 and 2020 to 2025 (no 2019).** Raw `tblAlarmLog` for 2020 onward is needed from Zenodo; SCADA for 2016, 2017 and 2019 is not in the processed set. The teammate cites Zenodo record 14870023 (v1.0.0, which ends 2024) while the repo pins v2.1.0 (to 2026); the version decision is still open.
5. Other facts from `Processed_Data_Description.txt`: `-50` and `-100` also occur as sensor fault values (not just `200`); `Station_91` is empty in every year; `TimeStamp` is stored as text; each year file ends with the next 1 Jan 00:00 row (watch for duplicates when concatenating years); the official docs do not say whether a timestamp marks the start or the end of its 10-min window (check against `wtc_CurTime_endvalue` in raw `tblSCTurbine`). Several column meanings in the description file are the author's interpretation, not official (hydraulic and hub pressure, A-series cabinet temperatures, converter and transformer temperatures, `PowerRef_endvalue`).

## 1. What the snippet contains

- 21 turbines (Station_2304510 to 2304530), 10-min rows, 2018-03-01 to 2019-01-01, about 919k rows, ~43.6k to 44.0k per turbine.
- No duplicate (turbine, timestamp); 10-min grid nearly complete (gaps >1 day: 6 across the fleet; gaps of 20 to 250 min: ~150).
- Empty sheet `Station_91` (1 row, removed). Drop it.
- 41 sensor columns, plus `Alarmcode` and `AlarmEvents`. All 41 are in the team's feature list (section 5); the two columns the cleaning step removed (`wtc_InvUpTmp_mean`, `wtc_TrafoTMa_mean`) are on that list but absent from the snippet.
- Not present: **active power output, pitch, yaw, wind direction**. `PowerRef_endvalue` is a setpoint (2300 = full rating, <2300 in 6.7% of rows), not measured power.
- No missing values in the snippet. Sentinel value `200.0` appears in several temperature columns (0.8% of rows; `A9IntTmp` 5,647 rows, `HSGenTmp`/`HSRotTmp`/`IMSGenTm`/`IMSRotTm` ~1,120 each, hub pressures 65 to 155). Likely a sensor-fault placeholder, TODO confirm and log in `DATA_TRAPS.md`.
- Alarms: 0.14% of rows carry one; 1,331 events in 42 distinct codes, stored on the onset row only (`TimeOn` equals the row timestamp). 91% have `TimeOff = null`, so durations are mostly unknown here.

## 2. Alarm codes in scope (decided by the team; role as targets superseded by D-003)

Only these 12 codes are targets. All other codes are dropped from labelling (see 2.2 for the one exception).
Source: the team's alarm table (`tblAlarmLog` code descriptions). Counts are unique events in the 2018 snippet.

| Code | Description | Stopping | Events | Turbines |
|------|-------------|----------|-------:|---------:|
| 20 | Large generator cut-in | 0 | 507 | 21 |
| 25 | Fast cut-out of generator | 0 | 505 | 21 |
| 102 | Ice detection | 0 | 0 | 0 |
| 1005 | Availability, low wind | 1 | 2 | 2 |
| 3130 | Pitch lubrication | 1 | 0 | 0 |
| 8000 | Windspeed too high to operate | 1 | 0 | 0 |
| 8210 | Stopped, due to icing | 1 | 0 | 0 |
| 8230 | Ice detection: low torque | 1 | 0 | 0 |
| 8234 | Ice detection: no cut-in | 1 | 0 | 0 |
| 8235 | Ice detect: stopped | 1 | 0 | 0 |
| 8236 | Ice detect: stopped, de-ice | 1 | 0 | 0 |
| 10105 | Stopped, untwisting cables | 1 | 1 | 1 |

In the snippet: 1,015 of 1,331 events are in scope, 316 dropped (top dropped codes: 127 x123, 50346 x62, 55 x39, 52020 x24).
Stopping codes total **3 events** (2 turbines); codes 20 and 25 are 1,012 of the 1,015. Codes 102, 3130, 8000 and all 82xx ice codes do not occur in Mar to Dec 2018 (the snippet misses Jan and Feb, and ice codes may be seasonal; unverified).

### 2.1 What this means for O2 (superseded by D-003 and section 0a)
- **Codes 20 and 25 are not faults**, they are generator cut-in and cut-out events (non-stopping). They happen about every 12 days per turbine (24 each over ~10 months). A "30-day window before an event" is positive on ~93% of daily samples for these codes (computed on the snippet), so presence-in-30-days cannot be the label for them.
- **Snippet counts understate stopping events** (3 events) because the processed file lost the alarms (section 0). Raw counts show Tier 1 is plentiful but frequent and short; see section 0 point 3 for the label consequence.
- **Weather-driven codes**: 1005 (low wind), 8000 (high wind) and the 82xx ice codes are mostly caused by weather, not equipment degradation. Predicting them from SCADA mostly means predicting weather from ambient temperature and wind, which does not match the "degradation" goal and can be near-trivial. TODO team to confirm they stay in scope.
- Proposed priority (my proposal, TODO team confirm):
  - **Tier 1: 3130 (pitch lubrication) and 10105 (untwisting cables).** Equipment and control related stops, closest to maintenance. Primary O2 target if the full data has enough events.
  - **Tier 2: 1005, 8000, 8210, 8230, 8234, 8235, 8236 (stopping, weather-driven).** Grouped as "stopping events (weather)". Reported separately; icing codes can be merged into one "icing" group (8210/8230/8234/8235/8236, plus 102 ice detection as an early indicator).
  - **Tier 3: 20, 25, 102 (non-stopping).** Used as operating-state features (counts and rates per window) and as an exploratory event-rate target, not a presence target.

### 2.2 Planned data steps (not yet implemented, no data edited)
1. Read the raw `tblAlarmLog` CSVs (columns `TimeOn`, `TimeOff`, `StationNr`, `Alarmcode`) from `data/raw` (loose or inside the year zips); drop station 91 and exact duplicates. Implemented in `src/wt_predmaint/data/alarms.py` and `scripts/build_alarm_table.py`. Do not use the `AlarmEvents` column of the processed workbooks.
2. Filter to the 12 codes above (list, tiers and stopping flags in `configs/protocol.yaml`, `labels.alarm_codes`). Drop the rest from target labels. Floor `TimeOn` to 10 min (`TimeStamp`) for joining to SCADA; keep exact `TimeOn`/`TimeOff` for lead time and duration.
3. Keep the dropped events in a separate "other alarm" table used **only to mask rows** when fitting normal-behaviour models (healthy-data filter, 4.1), so faults in dropped codes do not contaminate the "healthy" training set. TODO team confirm this exception to "drop the rest".
4. Tag each in-scope event with `stopping` (0/1) and tier; compute per-turbine, per-month counts and report them before any model work.
5. Because `TimeOff` is mostly null, derive stopping duration from `Hill_of_Towie_ShutdownDuration.zip` or from the sensor state (rpm near zero) rather than from `TimeOff`. TODO.

## 3. Other issues that change the plan

1. **No power column.** The power-curve baseline cannot use measured power from this export. Options: (a) pull `wtc_ActPower_mean` from `tblSCTurbine` in the raw data if present (preferred); (b) fall back to a rpm vs wind speed curve. TODO Farhan/Parvez: check the raw `tblSCTurbine` fields.
2. **One year only.** Year-based train/val/test (protocol) cannot be done on the snippet. Within 2018 use a chronological split for code testing only.
3. **Cleaning side effects.** `.cleaning.json` shows input rows well above kept rows (e.g. 97,299 input vs 43,606 kept), with most removed rows carrying alarm events. Kept counts match a full 10-min grid (306 days x 144 = 44,064), so the extra input rows are probably duplicate timestamps from an alarm join, but this is not verified. TODO Parvez: confirm no genuine data (e.g. downtime) was dropped.
4. **xlsx is slow** (several minutes to open). A parquet copy is at `data/interim/turbines_2018.parquet` (git-ignored). `openpyxl` is not in `pyproject.toml`.
5. **Fleet heterogeneity:** sensor offsets differ by turbine; normal-behaviour models need per-turbine handling and a leave-one-turbine-out check.

## 4. Pipeline (all code in `src/wt_predmaint/`, settings from `configs/protocol.yaml`)

| Stage | Module | Content |
|-------|--------|---------|
| Load | `data/` | read parquet, drop empty station, set sentinels (200.0) to NaN, build event table (2.2) |
| Clean | `data/` | run-state filter, gap flags, no imputation beyond train-fit imputers |
| Labels | `labels/` | in-scope event filter, tiers, O2 windows from config |
| Features | `features/` | operating-state features, rolling 7-day stats, residuals, alarm-rate features (codes 20/25/102) |
| Models | `models/` | baselines, NBM, LightGBM, optional autoencoder |
| Evaluation | `evaluation/` | PR-AUC, recall at budget, median lead time, LOTO splitter |
| Explain | `explain/` | SHAP, residual attribution |

## 5. Feature groups (by source table)

All 41 snippet columns are covered. Owners are for feature curation (who understands the table); coding ownership stays per `roadmap.md`.

| Table | Owner | Features |
|-------|-------|----------|
| `tblSCTurPress` | Revael | `GearPres`, `HydPress`, `HubPresA/B/C` (gear, hydraulic, hub-hydraulic pressure). Sentinel 200 seen in hub pressures. |
| `tblSCTurTemp` | Parvez | Drivetrain: `HSGenTmp`, `HSRotTmp`, `IMSGenTm`, `IMSRotTm`, `GeOilTmp`, `GenBeGTm`, `GenBeRTm`, `MainBTmp`; generator windings `Gen1U1/V1/W1`; environment `AmbieTmp`, `NacelTmp`; power electronics `ConvWTmp`, `TraOilTF`, `TraRooTF`, `HydOilTm`; cabinets and others `A1ExtTmp`, `A3LefTmp`, `A3RigTmp`, `A9IntTmp`, `A21IntTm`, `DeltaTmp`, `BrkTmpGn`, `GFilB1/2/3Tm`, `ReacU/V/WTmp`. `InvUpTmp` and `TrafoTMa` are on the team list but were removed in cleaning; TODO decide to re-add from raw. |
| `tblSCTurbine` | Farhan | `PrWindSp_mean/stddev` (wind speed), `GenRpm_mean/stddev` (generator speed), `MainSRpm_mean` (rotor speed), `PowerRef_endvalue` (active power setpoint). |

Feature ideas tied to the in-scope codes:
- **Pitch lubrication (3130):** hub-hydraulic pressures (`HubPresA/B/C`, spread between the three), `HydPress`, `HydOilTm`, rotor speed vs wind speed.
- **Untwisting cables (10105):** needs yaw/nacelle position, which is **not in the data**; rely on proxies (long periods of directional wind are not observable). TODO flag as a limitation.
- **Weather stops (1005, 8000, 82xx, 102):** wind speed mean and std, ambient and nacelle temperature, `DeltaTmp`, rpm vs wind speed. High risk of near-trivial prediction; treat as a separate, clearly labelled benchmark.
- **Cut-in / cut-out (20, 25):** counts and rates per 7-day window, cut-outs per cut-in, `GenRpm_stddev`, `PrWindSp_stddev`.

## 6. Objective 1: degradation indicator (normal-behaviour model)

Predict health-bearing temperatures from operating conditions on healthy data; the smoothed gap (predicted minus observed) is the indicator.

### 6.1 Preparation
- **Healthy-data filter** (TODO thresholds, in config): exclude rows near any alarm (including dropped codes, 2.2 step 3), low wind or non-producing states, sentinel rows. Fit only on train-period healthy rows.
- **Inputs** (operating condition): `PrWindSp_mean/stddev`, `GenRpm_mean/stddev`, `MainSRpm_mean`, `PowerRef_endvalue`, `AmbieTmp_mean`, `NacelTmp_mean`, with lags (thermal response is delayed).
- **Targets** (one NBM each): `GeOilTmp`, `GenBeGTm`/`GenBeRTm`, `MainBTmp`, `Gen1U1/V1/W1`, `HSGenTmp`/`IMSGenTm`/`HSRotTmp`/`IMSRotTm` (watch sentinel), `TraOilTF`, `HydOilTm`. For the pressure side: `HubPresA/B/C` and `HydPress` as NBM targets for the Tier 1 pitch-lubrication hypothesis. TODO mechanical and electrical members to prune.

### 6.2 Models, in order
1. **Ridge NBM** (baseline), per target, per turbine and pooled with turbine offset.
2. **LightGBM regression NBM** (main).
3. **Optional autoencoder** on healthy rows.
4. **Reference:** per-turbine z-score of the raw sensor against its own healthy history (no model).

### 6.3 Indicator and checks
- Residual smoothed over 1 to 7 days (TODO), standardised on train healthy residual spread.
- Check residual vs time-to-next in-scope event; start with Tier 1 codes. Report only computed numbers.

## 7. Objective 2: alarm / downtime prediction (labels superseded by D-003, D-005; models and metrics still apply)

### 7.1 Labels (windows are TODO in config)
- Prediction time t, per turbine. Feature window: 7 days before t. Positive: an in-scope **target-tier** event within 30 days after t. Exclusion window: 30 to 60 days; rule must be written in `DECISIONS.md` before coding (TODO: after a prior event, or after the positive window).
- Label sets (separate models or separate heads, reported separately): Tier 1; Tier 1 + Tier 2 stopping; weather-only benchmark. Tier 3 only as features and an exploratory rate target (e.g. unusually high cut-out count in the next 7 days, threshold TODO).
- Sample daily per turbine to limit overlap. Count positives per fold before modelling; if folds have too few positives, merge groups or use grouped CV, and say so in the report.

### 7.2 Features (7-day window, using data up to t only)
- Mean, std, max, min, slope of the section 5 sensors and NBM residuals; cross-sensor deltas (generator bearing minus ambient, U/V/W winding imbalance, hub pressure A/B/C spread); operating-state fractions; past counts of in-scope alarms (past only).

### 7.3 Models, in order
1. **Prevalence dummy** and **recent-alarm-count rule** (floors).
2. **Power-curve threshold baseline** (needs active power, issue 3.1).
3. **Class-weighted logistic regression** on window features.
4. **Class-weighted LightGBM** (main), `scale_pos_weight` from train only.

### 7.4 Thresholding and metrics
- **PR-AUC**, **recall at 1 false alarm per turbine-month**, **median lead time**. Threshold from validation, frozen for test. Alert cooldown so false alarms count per episode (TODO).
- Report per label set and per tier; with Tier 1 events this rare, report counts next to every metric and use turbine bootstrap intervals.

## 8. Objective 3: robustness

- **Leave-one-turbine-out:** 21 folds for NBM, logistic and LightGBM; report per-turbine spread. Residual standardisation uses training turbines only. Turbines with no Tier 1 events contribute only false alarms; report that.
- **Missing sensors:** drop each group (pressure, drivetrain temperature, power electronics, wind/rpm) and random dropout at test time; report PR-AUC and recall-at-budget change.
- **Upgrades and drift:** `Hill_of_Towie_AeroUp_install_dates.csv` (Zenodo, not downloaded) for before/after splits; needs multi-year data.
- **Temporal:** train early years, test later years.

## 9. Explainability

- SHAP TreeExplainer on LightGBM (global and local for true alerts); NBM per-sensor residual attribution; rank stability across LOTO folds. SHAP is model attribution, not causation.

## 10. Evaluation harness (Farhan)

- One function: (y_true, score, turbine, time) to PR-AUC, recall at budget, median lead time, turbine bootstrap intervals. Splitters: chronological and LOTO.
- Tests on synthetic labels with known answers, plus a leakage test (future rows must not change features at t).

## 11. Suggested order of work (superseded by `docs/roadmap.md`)

1. Parvez: put the raw `tblAlarmLog` files (all years, including 2020 onward) in `data/raw`; run `python scripts/build_alarm_table.py` (builds `data/interim/alarm_events.parquet` and prints counts per tier, code and year); confirm sentinel (`200`, `-50`, `-100`) handling; locate active power.
2. Team: tiering is accepted (D-002). Next decisions: label design (section 0 point 3), `episode_gap_days`, masking, and the other TODOs in `protocol.yaml`.
3. Kanav and Revael: parquet loader, event filter, label builder, tests on the snippet.
4. Farhan: metrics and splitters on synthetic data.
5. Baselines, then LightGBM, then robustness and SHAP.

## 12. Open decisions (history; current list is in `docs/DECISIONS.md`)

- Tiering and whether weather-driven codes stay in scope; whether Tier 3 is only features.
- Keep dropped-code events as a masking table (2.2 step 3).
- Exact meaning of the 30 to 60 day exclusion window.
- Healthy-row filter thresholds, NBM targets, smoothing window.
- Sentinel (200.0) handling; `PowerRef_endvalue < 2300` as curtailment.
- Re-adding `InvUpTmp` and `TrafoTMa` from raw.
- Alert cooldown, sampling cadence, source of active power.
