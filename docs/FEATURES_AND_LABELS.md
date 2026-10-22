# Labels and features reference

H0 normal-behaviour fitting now uses [healthy-day and interval masks](HEALTHY_ROWS.md)
(D-030), scoped to 2016–2025 and train years only. It selects target-specific
negative labels with clear quality flags and a seven-day recovery buffer.
It is a training-selection helper, not a new risk feature or a revised label definition.

B3 now uses those masks to fit its [wind/generator-RPM proxy](POWER_CURVE.md)
(D-031). Its target-specific daily diagnostics are written separately to
`data/interim/features/b3_daily_{a,b}.parquet`; they are not automatically
appended to the shared `daily_features.parquet` or used by other classifiers.
The seven-day exceedance score is not a calibrated fault probability. Initial
A/B validation has lower episode recall than the random reference at comparable
false-alert budgets; see the linked results and project logs (2026-10-22).

Reference for the O2 labels (D-005, D-019) and the daily feature table (D-021). Counts come from the
files built on 2026-10-22; if the code changes, rebuild and update this page. Sensor groupings are
read from the column names (the data dictionary in `data/raw/metadata/` is the authority), so treat
the plain-language meanings as inferred.

One example is one turbine on one day (21 turbines, 2016 to 2026-04, 2019 unused). The row for
(`StationId`, `Day`) holds what is known at the end of `Day`; the label asks about the following days.

## 1. Labels

Built by `scripts/build_labels.py` ([horizon.py](../src/wt_predmaint/labels/horizon.py),
[episodes.py](../src/wt_predmaint/labels/episodes.py)). Files in `data/interim/labels/`:

| File | Columns |
|---|---|
| `a_labels.parquet`, `b_labels.parquet` | `StationId`, `Day`, `status`, `label`, `split` |
| `a_episodes.parquet`, `b_episodes.parquet` | `StationId`, `Start`, `End` |

### Targets (what counts as a fault)

| Target | Source | Rule | Episode merging |
|---|---|---|---|
| **A** | `tblDailySummary.TimeErr` | a usable day with at least 4 hours of fault time | fault days within 7 days form one episode |
| **B** | 10-min SCADA | at least 18 consecutive intervals (3 hours) with generator rpm below 200 and wind at least 4.0 m/s, outside outage months | stop runs within 7 days form one episode |

Alarm codes are features, not targets (D-003). A and B agree on about 80% of days, so they are not
independent evidence.

### Label columns

| Column | Values | Meaning |
|---|---|---|
| `label` | 1, 0, empty | 1 if an episode starts in days D+1 to D+30, 0 for a clean negative, empty if the row is dropped |
| `status` | see below | why the row has that label |
| `split` | `train`, `val`, `test`, `none` | by year: train 2016 to 2018 and 2020, val 2021 to 2022, test 2023 to 2026-04 (D-006) |

### Status values

Precedence from highest: `in_episode`, `purged`, `positive`, `zone`, `censored`, `low_coverage`,
`negative`.

| Status | Label | Meaning | Used for training or scoring |
|---|---|---|---|
| `positive` | 1 | an episode starts in the next 30 days | yes |
| `negative` | 0 | no episode in the next 60 days, horizon observed | yes |
| `in_episode` | empty | the day lies inside an episode span | no |
| `zone` | empty | no start within 30 days, but one in days 31 to 60 | no |
| `purged` | empty | the 60-day horizon reaches a different split | no |
| `censored` | empty | the horizon runs past the last observed day | no |
| `low_coverage` | empty | fewer than 80% of the next 60 days were observed | no |

### Label counts (`reports/tables/label_summary.csv`)

| Target | Split | Labelled rows | Positive | Episodes |
|---|---|---|---|---|
| A | train / val / test | 22,309 / 9,690 / 17,951 | 37.3% / 47.0% / 45.1% | 355 / 217 / 336 |
| B | train / val / test | 20,911 / 8,827 / 17,281 | 47.8% / 49.2% / 50.5% | 434 / 209 / 382 |

A random score already reaches episode recall of about 0.8 at the false alarm budget (D-020), so
always read results next to `reports/tables/baselines_<target>.csv`.

## 2. Features

Built by `scripts/build_features.py` ([daily.py](../src/wt_predmaint/features/daily.py)) into
`data/interim/features/daily_features.parquet`: 70,816 turbine-days and **366 model columns**, plus
the keys `StationId` and `Day`. Every labelled row has features except 4 target-A test rows.

A 10-min interval belongs to the day of its start (`TimeStamp` minus 10 minutes). Windows are
trailing (day D and the 6 days before it). `scripts/check_feature_leakage.py` deletes the future and
checks each row is unchanged (D-021).

| Group | Columns | What it is |
|---|---|---|
| `op_<sensor>_mean` | 73 | daily mean of each 10-min mean sensor over producing intervals (rpm at least 1000); missing if fewer than 12 such intervals |
| `max_<sensor>_max` | 37 | daily maximum of each 10-min maximum sensor |
| Day state | 6 | how the day went (below) |
| `w7mean_<column>` | 116 | trailing 7-day mean of the `op_`, `max_` and day-state columns; needs at least 4 observed days |
| `w7std_<column>` | 116 | trailing 7-day standard deviation of the same 116 columns |
| `alarm_*_w7` | 16 | alarm counts over the trailing 7 days |
| Fault history | 2 | past fault time from `tblDailySummary` |
| **Total** | **366** | |

### Day-state columns (6)

| Column | Meaning |
|---|---|
| `frac_producing` | share of the day's intervals with generator rpm at least 1000 |
| `frac_idle` | share with rpm below 200 |
| `frac_stopped` | share with rpm below 200 while wind is at least 4.0 m/s |
| `wind_mean` | mean primary wind speed |
| `wind_std` | standard deviation of the 10-min wind speeds in the day |
| `rpm_mean` | mean generator rpm |

### Fault-history columns (2)

| Column | Meaning |
|---|---|
| `days_since_fault_day` | days from the latest fault day (target A rule) up to and including D; 0 if D is one; empty if none yet |
| `fault_days_30` | fault days among the 30 days ending at D |

### Alarm columns (16)

Counts over the trailing 7 days of the 12 documented codes (`configs/protocol.yaml`), by tier and in
total. Days with no alarm count as zero, including days without SCADA coverage.

| Tier | Codes | Columns |
|---|---|---|
| 1 | 3130, 10105 | `alarm_3130_w7`, `alarm_10105_w7`, `alarm_tier1_w7` |
| 2 | 1005, 8000, 8210, 8230, 8234, 8235, 8236 | `alarm_<code>_w7` for each, `alarm_tier2_w7` |
| 3 | 20, 25, 102 | `alarm_20_w7`, `alarm_25_w7`, `alarm_102_w7`, `alarm_tier3_w7` |
| all | | `alarm_total_w7` |

### Sensors behind `op_` (73, daily mean while producing)

The column is `op_wtc_<name>_mean`; the groups are inferred from the names.

| Group | Names |
|---|---|
| Wind and rotor | `SecAnemo`, `PriAnemo`, `PrWindSp`, `SeWindSp`, `AcWindSp`, `ActualWindDirection`, `GenRpm`, `MainSRpm`, `YawPos`, `ScYawPos`, `NacelPos` |
| Blade pitch | `PitchRef_BladeA`, `PitchRef_BladeB`, `PitchRef_BladeC`, `PitcRefA`, `PitcPosA`, `PitcPosB`, `PitcPosC` |
| Vibration and frequency | `TetAnFrq`, `TowerFrq_Frequenc` |
| Generator temperatures | `Gen1U1Tm`, `Gen1V1Tm`, `Gen1W1Tm`, `GenBeGTm`, `GenBeRTm`, `GenAvgTm`, `BrkTmpGn` |
| Gearbox and drivetrain temperatures | `GeOilTmp`, `HSGenTmp`, `HSRotTmp`, `IMSGenTm`, `IMSRotTm`, `TraOilTF`, `TraRooTF`, `MainBTmp` |
| Converter and electrical temperatures | `ConvWTmp`, `ReacUTmp`, `ReacVTmp`, `ReacWTmp`, `InvUpTmp`, `TrafoTMa`, `GFilB1Tm`, `GFilB2Tm`, `GFilB3Tm`, `DeltaTmp` |
| Nacelle, hub, ambient and board temperatures | `A1ExtTmp`, `A3LefTmp`, `A3RigTmp`, `A9IntTmp`, `A21IntTm`, `AmbieTmp`, `NacelTmp`, `HubTemp`, `HydOilTm`, `TwrHumid`, `IO1IntTm`, `IO2IntTm`, `IO3IntTm`, `NMMBrdTm` |
| Pressures and related | `BrakPres`, `GearPres`, `HubPresA`, `HubPresB`, `HubPresC`, `HydPress`, `InlPrAft`, `InlPrBef`, `OfflPres`, `InvPress`, `BrAccPrs`, `BoostAva`, `BoostAct`, `PwrRedLowSpeed` |

### Sensors behind `max_` (37, daily maximum)

The column is `max_wtc_<name>_max`.

- Temperatures (32): `A1ExtTmp`, `A3LefTmp`, `A3RigTmp`, `A9IntTmp`, `AmbieTmp`, `BrkTmpGn`, `Gen1U1Tm`,
  `Gen1V1Tm`, `Gen1W1Tm`, `GenBeGTm`, `GenBeRTm`, `GeOilTmp`, `HSGenTmp`, `HSRotTmp`, `IMSGenTm`,
  `IMSRotTm`, `NacelTmp`, `TraOilTF`, `TraRooTF`, `ConvWTmp`, `HydOilTm`, `GFilB1Tm`, `GFilB2Tm`,
  `GFilB3Tm`, `ReacUTmp`, `ReacVTmp`, `ReacWTmp`, `MainBTmp`, `DeltaTmp`, `A21IntTm`, `TrafoTMa`,
  `InvUpTmp`.
- Pressures (5): `GearPres`, `HubPresA`, `HubPresB`, `HubPresC`, `HydPress`.

## 3. Not model inputs

Kept in the table for filtering and checking, excluded by `feature_columns`:

| Column | Why |
|---|---|
| `StationId`, `Day` | keys |
| `outage_month`, `suspect_wind` | data-quality flags (D-015, D-017); filter or stratify on them, do not feed the model |
| `n_intervals`, `n_producing`, `masked_frac` | data-quality counts |

## 4. Known limits

- About 12 of the 110 base sensor columns are over 50% missing, because the sensor only exists in some
  years (for example `IO1IntTm`, `NMMBrdTm`, `TrafoTMa`). On average 13% to 22% of feature cells are
  missing per split. Choose columns and imputation on train years only.
- 366 columns against about 20,000 training rows: simple models need a selected subset.
- Not built yet: normal-behaviour residual features (need the fitted models), trends and cross-sensor
  ratios.
- Features are not model-ready: no scaling or imputation has been applied.

See D-019 and D-021 in [DECISIONS.md](project/DECISIONS.md) for the reasoning and
[DATA_TRAPS.md](project/DATA_TRAPS.md) for the data problems behind these choices.
