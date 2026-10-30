# Target C: alarm within the next six hours

This is an experimental binary classification task, separate from the frozen A/B
fault-episode experiments. It predicts **any new turbine alarm**, not necessarily a
mechanical failure. Weather, operational and repeated alarms count too. Site-level
station 91 is excluded, following the raw alarm loader. All turbine codes are used,
including those outside the A/B feature-code shortlist.

## Exact target

Make a prediction for each available 10-minute SCADA interval end `t` and turbine:

- **yes (1):** at least one raw `TimeOn` is in `(t, t + 6 hours]`.
- **no (0):** no new alarm in that window, with confirmed full alarm-log coverage.
- **unknown:** coverage is insufficient and no future alarm is witnessed.

At 13:00, an alarm at 13:03 or exactly 19:00 makes the label yes. An alarm at
13:00 has already happened and does not count as a future alarm. A continuing
alarm does not create another positive unless a new alarm record starts.
Use exact alarm start times; do not round 13:03 to 13:00. SCADA timestamps mark
interval **ends**, so sensor data through 13:00 is available at 13:00; the
13:10 sensor row is forbidden. This assumes ingestion is available without delay;
an operational deployment must account for actual ingestion latency.

## Inputs

Use current cleaned mean/max sensor readings and trailing means/standard deviations
over 1 hour, 24 hours and 7 days, separately per turbine. These capture both recent
changes and the longer operating baseline. Windows include only `(t-window, t]`;
they never include the rest of the current day. Missing history stays missing,
with observed SCADA row counts provided for each window. Past alarm counts and
time since the most recent alarm are also allowed at `t`. The builder marks past
alarm counts missing unless the corresponding historical window has verified
coverage; recency requires verified seven-day history.
Full-day fault summaries and retrospective month/year quality flags are excluded.
Seven days is a starting choice, not a demonstrated optimum. Early rows use partial
history; the model receives coverage counts and imputed sensor values.

## Build and run

Use the project environment (`pip install -e ".[dev]"`). First prepare raw alarm
logs and cleaned SCADA as described in `docs/general/USAGE.md`. This clone does not
include the raw data or generated cleaned SCADA.

Prepare a CSV of **verified complete observation intervals** with columns:

```csv
StationId,Start,End
```

Add one row per turbine and uninterrupted log-coverage interval, using ISO timestamps
in the same time basis as SCADA and raw alarms. `(Start, End]` certifies that **all**
alarm starts in that interval were recorded, including quiet periods. Obtain this
from the source export inventory; the first/last observed alarm cannot establish
coverage. Split intervals around known missing periods; touching intervals merge.
Do not fabricate full-year coverage simply because a SCADA file exists.

```powershell
python scripts/build_target_c.py --coverage path/to/verified_alarm_coverage.csv
python scripts/run_target_c.py
```

The builder writes partitioned feature/label tables in `data/interim/target_c/`.
Each partition contains one turbine and SCADA source year. It carries seven days
of preceding SCADA context across file boundaries. All years must use identical
sensor schemas and consistent timestamps. Rebuild into an empty output directory
when changing the set of input files, to avoid stale partitions.

The runner uses the existing configured train/validation years, requires fully
observed horizons for **both** classes, and purges labels whose six-hour outcome
window crosses into another split or an unused year. Historical inputs from before
a split boundary remain available, as they would in a real forecast. No random
row split is used. Official test years are not scored and A/B results are unchanged.

The first model is class-weighted logistic regression with train-only median
imputation and scaling. A fixed score threshold of 0.5 produces English `yes`/`no`
predictions. Class-weighted scores are **not calibrated probabilities**. Files in
`reports/target_c/` include the fitted model, validation predictions and metrics:
PR-AUC (average precision), constant-score reference, precision, recall, balanced
accuracy and confusion matrix. Accuracy alone can hide an always-no model.

These are prediction-point metrics: neighbouring six-hour windows overlap heavily.
They are not event-level recall or independent statistical samples. Before choosing
a deployment threshold, inspect false alarms, compare a recent-alarm baseline and
use time-block evaluation. Evaluate a model without alarm-history inputs if the
goal is specifically to anticipate alarms from SCADA alone.

The full dataset can be large: construction processes one year at a time and one
turbine's features at a time, while fitting currently loads train/validation tables
into memory. No performance claims are made until it runs on the actual data.

## Memory-bounded local performance run

`scripts/audit_target_c_sources.py` audits the supplied monthly exports. It writes
a calendar-based **provisional** coverage file, explicitly assuming complete logging
for represented turbines in supplied months. Entire days with no fleet alarm records
are conservatively excluded. This is not independently certified coverage.

To use cleaned SCADA from an existing local checkout, write new outputs in this clone:

```powershell
python scripts/audit_target_c_sources.py --raw ../ENGG2112-T3-G7/data/raw --out data/interim/target_c_audit
python scripts/build_target_c.py --clean ../ENGG2112-T3-G7/data/interim/clean --events data/interim/target_c_audit/raw_alarm_events.parquet --coverage data/interim/target_c_audit/provisional_alarm_coverage.csv --out data/interim/target_c_hourly --years 2016 2017 2018 2020 2021 2022 --prediction-minutes 60
python scripts/evaluate_target_c_local.py --input data/interim/target_c_hourly --max-train-rows 100000
```

The hourly option computes rolling statistics from **all available 10-minute source
rows**, then retains hourly prediction anchors. It changes evaluation cadence, not
the six-hour outcome window or the underlying history. The default builder remains
10-minute cadence. The local runner takes an exactly sized, reproducible uniform
sample across all eligible training points and scores every eligible validation
anchor in batches. Reports explicitly distinguish this sample from a full-data fit.
It also reports always-yes, always-no and a recent-one-hour-alarm rule as baselines.
The complete prediction export is Parquet; the CSV is a 1,000-row preview only.
