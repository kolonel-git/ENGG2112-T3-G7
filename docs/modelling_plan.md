# Modelling plan (draft for team review)

Based on `data/processed/00-Processed-Data/turbines_2018.xlsx` (+ `.cleaning.json`): a 2018 snippet, not the full dataset.
All numbers below were computed from that file. Anything marked TODO is a team decision, not a finding.

## 1. What the snippet contains

- 21 turbines (Station_2304510 to 2304530), 10-min rows, 2018-03-01 to 2019-01-01, about 919k rows total, ~43.6k to 44.0k per turbine.
- No duplicate (turbine, timestamp); 10-min grid nearly complete (gaps >1 day: 6 across the fleet; gaps of 20 to 250 min: ~150).
- Empty sheet `Station_91` (1 row, removed). Drop it.
- 41 sensor columns (`wtc_*_mean`, plus `_stddev` for wind speed and generator rpm, `PowerRef_endvalue`), plus `Alarmcode` and `AlarmEvents`.
  - Temperatures: gearbox oil, generator bearings and windings (U/V/W), main bearing, nacelle, ambient, converter, transformer, hydraulic oil, filter banks, reactors, various `A*` cabinet sensors.
  - Pressures: gearbox, hydraulic, hub A/B/C.
  - Operating: wind speed (mean/std), generator rpm (mean/std), main shaft rpm, power reference.
- Not present: **active power output, pitch, yaw, wind direction**. `PowerRef_endvalue` is a setpoint (2300 = full rating, <2300 in 6.7% of rows), not measured power.
- No missing values in the snippet (cleaning already removed or dropped them). `wtc_InvUpTmp_mean` and `wtc_TrafoTMa_mean` were removed as columns.
- Sentinel value `200.0` appears in several temperature columns (0.8% of rows; `A9IntTmp` 5,647 rows, `HSGenTmp`/`HSRotTmp`/`IMSGenTm`/`IMSRotTm` ~1,120 each, hub pressures 65 to 155). Likely sensor-fault placeholder, TODO confirm and log in `DATA_TRAPS.md`.
- Alarms: only 0.14% of rows carry an alarm. 1,331 events in total, all stored on the onset row only (`TimeOn` equals the row timestamp). 1,217 (91%) have `TimeOff = null`, so alarm duration is mostly unknown here.
  - Counts by code: 20 (507), 25 (505), 127 (123), 50346 (62, one turbine only), 55 (39), 52020 (24, ~5 h median), then a long tail of ~20 rare codes.
  - Events per turbine: median 57, range 45 to 131 (Station_2304527 is the outlier with 131).
  - What the codes mean is unknown to me. `Hill_of_Towie_alarms_description.csv` is on Zenodo (metadata file) and has not been downloaded.

## 2. Issues that change the plan

1. **Alarm base rate makes a naive O2 label useless.** ~63 events per turbine over ~10 months is roughly one every 5 days. A "30-day window before any alarm" would label almost every row positive. O2 needs a **restricted target**: specific alarm codes and/or alarms that cause downtime (use `Hill_of_Towie_ShutdownDuration.zip` or fault-class codes from the alarm description). TODO decide target definition; this is the most important open decision.
2. **Alarm codes 20 and 25 (~76% of events) dominate** and are probably routine (this is a guess until the description file is read). Do not predict them as the main target without checking.
3. **No power column.** The power-curve baseline cannot use measured power from this export. Options: (a) pull `wtc_ActPower_mean` from `tblSCTurbine` in the raw data and add it to the export (preferred); (b) fall back to a rpm vs wind speed curve. TODO Parvez: check the raw `tblSCTurbine` fields.
4. **One year only.** Year-based train/val/test (protocol) cannot be done on this snippet; it is only for developing code. Real splits need the full 2016 to 2026 data. Within 2018, use a chronological split (e.g. first 6 months fit, next 2 validate, last 2 test) for code testing only.
5. **Long gaps and cleaning side effects.** `.cleaning.json` shows input rows far above kept rows per turbine (e.g. 97,299 input vs 43,606 kept), with most "removed rows" carrying alarm events. Kept row counts match a full 10-min grid (306 days x 144 = 44,064), so the extra input rows are probably duplicate timestamps from an alarm join, but this is not verified. TODO Parvez: confirm no genuine data (e.g. downtime periods) was dropped.
6. **Data stored as xlsx is slow** (opening takes several minutes). I converted it to `data/interim/turbines_2018.parquet` (git-ignored). Use parquet from here on. Note `openpyxl` is not in `pyproject.toml`; add it only if xlsx reading is needed.
7. **Fleet heterogeneity:** turbine sensors differ in offset (e.g. `Gen1*` temps). Normal-behaviour models should include turbine-level handling (per-turbine models or station-id features) and be checked under leave-one-turbine-out.

## 3. Pipeline (all code in `src/wt_predmaint/`, settings from `configs/protocol.yaml`)

| Stage | Module | Content |
|-------|--------|---------|
| Load | `data/` | read parquet, drop empty station, set sentinels (200.0) to NaN, parse `AlarmEvents` JSON to an event table (turbine, code, on, off) |
| Clean | `data/` | run-state filter (see 4.1), flag gaps, no imputation beyond train-fit imputers |
| Labels | `labels/` | O2 positive window / exclusion window / feature window from config; per-turbine event table |
| Features | `features/` | operating-state features, rolling 7-day stats of residuals and raw sensors, deltas between related sensors |
| Models | `models/` | baselines, NBM, LightGBM, optional autoencoder |
| Evaluation | `evaluation/` | PR-AUC, recall at 1 false alarm per turbine-month, median lead time, LOTO splitter |
| Explain | `explain/` | SHAP for LightGBM, residual attribution for the NBM |

## 4. Objective 1: degradation indicator (normal-behaviour model)

Idea: predict a health-bearing target temperature from operating conditions on healthy data; the gap (predicted minus observed, smoothed) is the indicator.

### 4.1 Preparation
- **Healthy-data filter** (TODO thresholds, in config): exclude rows near alarms (e.g. N days after an alarm onset), low wind or non-producing states (`GenRpm_mean` low, `PowerRef_endvalue` below rating), and sentinel rows. Fit only on train-period healthy rows.
- **Inputs** (operating condition, not health): `PrWindSp_mean/stddev`, `GenRpm_mean/stddev`, `MainSRpm_mean`, `PowerRef_endvalue`, `AmbieTmp_mean`, `NacelTmp_mean`.
- **Targets** (one NBM per target): gearbox oil `GeOilTmp`, generator bearings `GenBeGTm`/`GenBeRTm`, main bearing `MainBTmp`, generator windings `Gen1U1/V1/W1`, `HSGenTmp`/`IMSGenTm` (watch the 200 sentinel), transformer oil `TraOilTF`. TODO: mechanical and electrical members to pick the targets that matter physically.
- Use lagged inputs (e.g. previous 1 to 6 steps of wind and rpm) since temperatures respond with thermal lag.

### 4.2 Models, in order
1. **Ridge NBM** (baseline): linear in operating inputs, per target. Fit per turbine and also pooled with turbine offset.
2. **LightGBM regression NBM** (main): same inputs and lags, one model per target.
3. **Optional autoencoder** (MLP, reconstruction error on healthy rows) as a multivariate alternative; only if time allows.
4. **Reference baseline for O1:** raw sensor level thresholds / z-score against turbine's own healthy history (no model), to show the NBM adds value.

### 4.3 Indicator and checks
- Residual = predicted minus observed, smoothed (rolling median over 1 to 7 days; TODO choice), standardised on train healthy residual spread.
- Indicator should rise before alarms: plot residual vs time to next event per turbine; check against the restricted target set (section 2.1).
- Report only what is computed; no claims of detection without lead-time numbers from the evaluation harness.

## 5. Objective 2: alarm / downtime prediction

### 5.1 Labels (windows are TODO in config)
- Prediction time t, per turbine. Feature window: 7 days before t. Positive: restricted-target event within 30 days after t. Exclusion window: 30 to 60 days (rows in the exclusion window are dropped, so the interpretation must be agreed: after a previous event, or between positive window and a later period; TODO write the exact rule in `DECISIONS.md` before coding).
- Sample at a coarser cadence (e.g. daily) than 10 min, to limit overlap between samples.
- With the snippet: count positives per code set before choosing windows; if positives per turbine are only a handful, report that honestly.

### 5.2 Features (7-day window aggregates, computed with data up to t only)
- Mean, std, max, min, slope of key sensors and of NBM residuals; fraction of time in operating states; count of alarms by code in the window (check this is allowed: past alarms are inputs, future ones are not).
- Cross-sensor deltas (e.g. generator bearing minus ambient, winding phase imbalance U/V/W, hub pressure A/B/C spread).

### 5.3 Models, in order
1. **Power-curve threshold baseline:** per turbine, flag when measured power falls below a power-curve fit by a margin for X% of the window. Needs active power (issue 3).
2. **Logistic regression** (class-weighted, L2) on the window features.
3. **Class-weighted LightGBM** (main), with `scale_pos_weight` from train only.
4. **Naive reference:** previous-alarm-count rule (alarms in last 7 days) and a prevalence-only dummy, so every metric has a floor.

### 5.4 Thresholding and metrics
- Metrics: **PR-AUC**, **recall at a budget of 1 false alarm per turbine-month**, **median lead time** (time from first alert to event).
- Choose the threshold on validation to meet the false-alarm budget, then freeze it for test.
- Alert de-duplication: once an alert fires, suppress repeats for a cooldown so false alarms are counted per episode (TODO cooldown).

## 6. Objective 3: robustness

- **Leave-one-turbine-out (LOTO):** 21 folds; pool results, report per-turbine spread. Run for NBM, logistic, LightGBM. Note NBM residual standardisation must use only the training turbines.
- **Missing sensors:** retrain/evaluate with each sensor group removed (gearbox, generator, hydraulic, transformer, ambient), plus random sensor dropout at test time. Report the drop in PR-AUC and recall at budget.
- **Upgrades and drift:** use `Hill_of_Towie_AeroUp_install_dates.csv` (on Zenodo, not yet downloaded) to split before/after the AeroUp upgrade on affected turbines; check NBM residual shift and metric change. Needs the full multi-year data.
- **Temporal robustness:** train on early years, test on later years.

## 7. Explainability

- SHAP TreeExplainer on LightGBM (global bar and beeswarm; local plots for the top true alerts).
- For the NBM: per-sensor residual contribution to indicate which component drifts.
- Stability check: SHAP rank agreement across LOTO folds.
- Explain only what the model uses; label SHAP as model attribution, not causal.

## 8. Evaluation harness (Farhan)

- One function takes (y_true, score, turbine, time) and returns PR-AUC, recall at budget, median lead time, with a bootstrap over turbines for intervals.
- Splitter functions: chronological split from config, LOTO.
- Tests: synthetic labels where the right answer is known (not results), plus a leakage test (shuffling future rows must not change features at t).

## 9. Suggested order of work

1. Parvez: download `Hill_of_Towie_alarms_description.csv`, `..._tables_description.csv`, `..._ShutdownDuration.zip`; confirm sentinel and cleaning questions; locate active power.
2. Team: decide the O2 target definition (section 2.1) and fill the TODOs in `protocol.yaml`.
3. Kanav and Revael: parquet loader, event table, label builder, tests on the snippet.
4. Farhan: metrics and splitters, tested on synthetic data.
5. Baselines (ridge NBM, logistic, power-curve) then LightGBM, then robustness and SHAP.

## 10. Open decisions to log in `docs/DECISIONS.md`

- O2 target alarm set / downtime definition.
- Exact meaning of the 30 to 60 day exclusion window.
- Healthy-row filter thresholds; NBM target sensors; residual smoothing window.
- Sentinel handling (200.0) and whether `PowerRef_endvalue < 2300` rows count as curtailed.
- Alert cooldown; sampling cadence for O2 samples.
- Source of active power for the power-curve baseline.
