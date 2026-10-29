# Revael: targets, labels and features

This section describes the two predictive-maintenance targets and the daily feature table used by
the models. It is based on the fixed rules in D-005, D-019 and D-021, the generated tables in
`reports/tables/`, and figures F05 to F14. Revael reviewed the implementation and signed it off on
2026-10-27. The review did not score or tune a model on the test years.

## Targets and labels

One example is one turbine at the end of one UTC day, D. Target A represents recorded fault time:
a usable `tblDailySummary` day (neither `partial_day` nor `bad_hours`) is a fault day when `TimeErr`
is at least 4 hours. Fault days are merged into an episode until seven fault-free days separate
them. Target B is a second
operational proxy built from the 10-minute SCADA: a stop requires at least 18 consecutive intervals
(3 hours) with generator speed below 200 rpm while primary wind speed is at least 4 m/s, outside
flagged outage months. A target-B candidate day must contain at least 115 of the expected 144
non-outage intervals. Stop runs are merged with the same seven-day episode rule. These thresholds
come from `configs/protocol.yaml` and D-007/D-008; `scripts/build_labels.py` applies them.

![Daily fault-time distribution and target-A threshold](figures/02_labels/F05_timeerr_distribution.png)

Figure F05 shows why the 4-hour threshold selects only the long tail of daily `TimeErr`: 3.3% of
usable turbine-days cross it. `TimeErr` is not defined by the provider's available dictionary, so
its interpretation remains an inference supported by its agreement with the SCADA stop proxy.

![Episodes per turbine and year](figures/02_labels/F06_episodes_per_turbine_year.png)

Across the available period, the episode builders produce 972 target-A episodes and 1,025 target-B
episodes (F06). Target B has no 2019 episodes because the full cleaned 10-minute table is
unavailable when the temperature and pressure files are absent. Turbine 2304515 has unusually high
episode and positive-label prevalence, which is why later evaluations include a pre-specified
sensitivity analysis without it.

![Agreement between targets A and B](figures/02_labels/F07_target_agreement.png)

The targets are related rather than independent: 75% of target-A episodes have a target-B episode
starting within seven days, while 72% of target-B episodes have a corresponding target-A episode
(F07). Reporting both therefore checks robustness to the fault definition; it does not provide two
independent confirmations of performance.

### Window and split rules

![Label and feature window geometry](figures/02_labels/F08_label_geometry.png)

At prediction time, the row for D uses information available by the end of D. An episode starting
in D+1 through D+30 makes the row positive. If no episode starts in that interval but one starts in
D+31 through D+60, the row is placed in the exclusion zone and receives no label. Days already
inside an episode are also dropped. A nominal negative is retained only when at least 80% of the
next 60 days are observed. The 60-day horizon may not cross from train to validation or from
validation to test; such rows are purged before labels are assigned. `horizon.py` applies the status
precedence `in_episode`, `purged`, `positive`, `zone`, `censored`, `low_coverage`, then `negative`.

The time split is fixed in `configs/protocol.yaml`: train years are 2016, 2017, 2018 and 2020;
validation years are 2021 and 2022; test years are 2023 to April 2026. The 2019 rows have split
`none` and are not used for modelling. Label construction may use future days within the same split
because it defines the outcome; model features and fitted preprocessing may not.

Rows with `in_episode`, `zone`, `purged`, `censored` or `low_coverage` status carry no label and are
excluded from fitting and metrics. Under D-022, they still retain a feature row and may receive a
score for a continuous risk timeline; their status band explains why they are absent from metrics.

![Label status by target and split](figures/02_labels/F09_label_status_by_split.png)

The generated label counts are:

| Target | Split | Labelled rows | Positive rows (rate) | Episodes |
|---|---|---:|---:|---:|
| A | train | 22,309 | 8,315 (37.3%) | 355 |
| A | validation | 9,690 | 4,558 (47.0%) | 217 |
| A | test | 17,951 | 8,099 (45.1%) | 336 |
| B | train | 20,911 | 10,001 (47.8%) | 434 |
| B | validation | 8,827 | 4,346 (49.2%) | 209 |
| B | test | 17,281 | 8,725 (50.5%) | 382 |

These values come directly from `reports/tables/label_summary.csv`, written by
`scripts/build_labels.py`. The high prevalence means that PR-AUC must always be compared with the
prevalence floor. Overlapping 30-day horizons also make neighbouring rows strongly autocorrelated;
the evaluation therefore bootstraps turbines rather than individual days.

![Positive-label rates by year and turbine](figures/02_labels/F10_positive_rate_year_turbine.png)

Figure F10 shows substantial year and turbine variation. In particular, 97% of target-A validation
rows for turbine 2304515 are positive. This row imbalance is reported as a sensitivity issue rather
than removed after seeing model results.

## Features

`scripts/build_features.py` creates `data/interim/features/daily_features.parquet`: 70,816
turbine-days with 366 model-input columns, in addition to the `StationId` and `Day` keys. The feature
families are:

| Feature family | Columns | Construction |
|---|---:|---|
| Daily producing means (`op_`) | 73 | Mean 10-minute sensor value while rpm is at least 1,000; missing with fewer than 12 producing intervals |
| Daily maxima (`max_`) | 37 | Maximum of each available 10-minute maximum sensor |
| Day state | 6 | Producing, idle and stopped fractions; daily wind mean/spread and rpm mean |
| Trailing means (`w7mean_`) | 116 | Mean over D and the previous six calendar days; at least 4 observed days |
| Trailing spreads (`w7std_`) | 116 | Standard deviation over the same causal seven-day window |
| Alarm counts | 16 | Counts by 12 documented codes, three tiers and total over the same seven days |
| Fault history | 2 | Days since the latest target-A fault day and fault-day count over the 30 days ending at D |
| **Total** | **366** | |

The counts above are computed by `feature_columns()` and plotted by F11 in
`src/wt_predmaint/reporting/figures_data.py`.

![Feature families](figures/03_features/F11_feature_groups.png)

A SCADA row timestamp marks the end of its 10-minute interval. The feature builder subtracts ten
minutes before assigning the interval to a day, so the row stamped at midnight belongs to the day
that just ended. Window features use a gap-free calendar and never compress missing dates into
adjacent observations. Alarm and fault-history features end on D as well. The keys, interval and
producing counts, masked-cell share, outage flag and suspect-wind flag remain available for quality
checks but are excluded by `feature_columns()` from model inputs.

![Sensor availability by year](figures/03_features/F12_sensor_availability_by_year.png)

Sensor availability changes sharply across years (F12). Some sensors exist only in 2016 and others
appear from 2022, while 2019 lacks the full SCADA inputs. About 12 of the 110 base sensor columns are
more than 50% missing. Consequently, feature selection and imputation must be learned from training
rows only; validation or test availability must not decide which columns survive.

![Operating-state features](figures/03_features/F13_operating_state.png)

Turbines are producing for roughly 40% to 50% of intervals in most complete years, rising to about
60% in the partial 2026 period (F13). Daily sensor means therefore use producing intervals only,
which reduces contamination by idle operation but does not adjust for load or wind.

Feature coverage from `reports/tables/feature_summary.csv` is:

| Target | Split | Labelled rows | Rows with features | Mean missing-cell share | Rows with every `op_` feature missing |
|---|---|---:|---:|---:|---:|
| A | train | 22,309 | 22,309 | 18.26% | 1,788 |
| A | validation | 9,690 | 9,690 | 21.25% | 1,475 |
| A | test | 17,951 | 17,947 | 13.40% | 1,418 |
| B | train | 20,911 | 20,911 | 18.67% | 1,790 |
| B | validation | 8,827 | 8,827 | 21.70% | 1,403 |
| B | test | 17,281 | 17,281 | 13.26% | 1,353 |

Only four labelled target-A test rows lack a feature row. Missing feature values are expected and
are handled later by train-only preprocessing; they are not silently converted into evidence of
healthy operation.

Alarm windows count the 12 documented codes by code, tier and total. A day without a recorded alarm
contributes zero, including a day without SCADA coverage, so the separate coverage flags must be
consulted when interpreting zero. There is also a recording shift: codes 20 and 25 fall sharply from
2022 onward, and code 1005 shows substantial chatter in 2022 (`docs/project/DATA_TRAPS.md`). Alarm-count
changes across that boundary cannot be interpreted as changes in failure risk alone. Trends,
cross-sensor ratios and normal-behaviour residuals are not part of this 366-column table.

![Healthy training rows by year](figures/03_features/F14_healthy_rows_by_year.png)

For the normal-behaviour references, H0 further restricts fitting to negative training days with
acceptable data-quality flags and a seven-day post-episode recovery period. About 20% to 45% of
training feature days remain, depending on year and target (F14). These target-specific healthy
masks select fitting examples only and are never model inputs.

## Look-ahead review and sign-off

The review checked `labels/episodes.py`, `labels/horizon.py`, `features/daily.py`,
`features/leakage.py`, the protocol values, and the boundary tests. The feature leakage check
rebuilds a sampled row after deleting SCADA and alarm records after D+1 00:00 and daily-summary rows
after D. A valid row must be unchanged. Its synthetic control deliberately adds tomorrow's
temperature and confirms that the check fails on a leaky feature. Boundary tests also confirm that
midnight interval ends belong to the preceding day, seven-day windows are trailing, missing
calendar days are not bridged, alarm windows expire on the correct day, and fault-history features
do not see future faults.

On 2026-10-27, the 27 targeted label, feature, leakage and report-figure tests passed, and Ruff
passed on the reviewed pipeline. The real-data truncation result recorded in D-021 checked 48
turbine-days on six turbines without finding a changed feature. On this evidence, Revael signs off
D-019 and D-021: the documented window rules match the implementation, and no feature look-ahead
was found.

## Number and figure traceability

| Claim or figure | Reproducible source |
|---|---|
| Label statuses, counts, positive rates and split episode counts | `scripts/build_labels.py` → `reports/tables/label_summary.csv` |
| F05 | `timeerr_distribution()` in `reporting/figures_data.py`, from `daily_summary.parquet` |
| F06–F07 and F10 | Registered functions in `reporting/figures_data.py`, from A/B episode or label Parquets |
| F08 | `configs/protocol.yaml`, D-005 and D-019 |
| F09 | `reports/tables/label_summary.csv` |
| Feature coverage and missingness | `scripts/build_features.py` → `reports/tables/feature_summary.csv` |
| F11–F13 | Registered functions in `reporting/figures_data.py`, from `daily_features.parquet` |
| F14 | `scripts/report_healthy_rows.py` → `reports/tables/healthy_rows.csv` |
| F05–F14 image files and index | `scripts/make_report_figures.py` → `reports/figures/` |

Rebuild the evidence, after preparing `data/interim/`, with:

```powershell
python scripts/build_labels.py
python scripts/build_features.py
python scripts/check_feature_leakage.py --stations 6 --days 8
python scripts/report_healthy_rows.py
python scripts/make_report_figures.py --only F05 F06 F07 F08 F09 F10 F11 F12 F13 F14
python -m pytest -q tests/test_labels.py tests/test_features.py tests/test_report_figures.py
python -m ruff check src/wt_predmaint/labels src/wt_predmaint/features
```

The detailed column reference remains in `docs/FEATURES_AND_LABELS.md`; decisions and caveats remain
authoritative in `docs/project/DECISIONS.md` and `docs/project/DATA_TRAPS.md`.
