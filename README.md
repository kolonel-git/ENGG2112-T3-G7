# Predicting Wind Turbine Degradation from SCADA

**Explainable predictive maintenance on open wind-farm data.**
ENGG2112 (University of Sydney), Group 7, "AI for Good".

Wind turbines are expensive to repair and expensive to leave idle. An unplanned breakdown means a
stopped turbine, a crew sent out at short notice and electricity that is never sold. This project
asks whether the sensor data a turbine already records can warn us **before** it fails, and whether
we can explain the warning well enough for a maintenance engineer to trust it.

## The problem

Each turbine logs hundreds of readings every 10 minutes (temperatures, pressures, rotor speed, wind
speed) plus a log of alarms. Many alarms are routine events that happen to every turbine all the
time, so "an alarm fired" is not the same as "something is wrong". We need a better definition of
failure, and a way to see trouble building up.

We work with real data: the Hill of Towie wind farm in Scotland, 21 Siemens SWT-2.3-VS-82 turbines,
about ten years (2016 to April 2026).

## What we are building

Three objectives, from the project brief:

| | Objective | Plain-language question |
|---|---|---|
| **O1** | Degradation indicator | Learn what "normal" looks like for a turbine (for example, how warm the gearbox should be at a given wind speed). How far today's readings drift from normal is a health score. |
| **O2** | Fault and downtime prediction | Will this turbine have a fault or a long stoppage in the next 30 days? |
| **O3** | Robustness | Do the predictions still hold for a turbine the model never saw, with a sensor missing, or after an upgrade? |

On top of that, every alert comes with an explanation (SHAP and residual attribution), written up as
"which signals pushed this up", not as proof of cause.

## How it works

1. **Clean the raw data.** Join the SCADA tables, mask placeholder sensor values (such as 200 or
   -50 for "no reading"), remove duplicates and flag bad periods. Nothing is silently dropped.
2. **Define a fault.** Two independent definitions that agree closely at day level:
   - **Target A:** a day with at least 4 hours of recorded fault time in `tblDailySummary`.
   - **Target B:** at least 3 hours in a row with the rotor stopped despite enough wind.

   Nearby fault days are merged into one episode (7-day gap). Alarm codes are used as input
   features, not as the target.
3. **Label every turbine-day.** Positive if a new episode starts in the next 30 days. Days we cannot
   label honestly are dropped: already inside an episode, the 30 to 60 day buffer, days whose
   horizon reaches another data split, and poorly observed days.
4. **Split by time.** Train on 2016 to 2018 and 2020, tune on 2021 to 2022, test on 2023 to 2026.
   There is no 2019 temperature or pressure data. Scalers, thresholds and models are fitted on the
   training years only.
5. **Build features.** Daily summaries of the 10-minute readings, a 7-day window, alarm counts, and
   the gap between observed and expected values from the normal-behaviour model.
6. **Train models.** Baselines first (prevalence, recent-fault rule, a power-curve style threshold,
   logistic regression), then class-weighted LightGBM, with an optional autoencoder.
7. **Evaluate fairly.** PR-AUC, recall at a budget of 1 false alarm per turbine-month, and median
   lead time (days of warning), each with confidence intervals that resample turbines, not rows.
8. **Stress test and explain.** Leave-one-turbine-out, missing-sensor, year-shift and upgrade
   analyses, then SHAP.

## Where we are

| Stage | Status |
|---|---|
| Repo, tooling, CI, decision log | Done |
| Raw data exploration and trap list | Done |
| Cleaning layer (flags for outage months and suspect wind sensors) | Done |
| Labels for targets A and B | Done and verified (see below) |
| Evaluation harness and random-score floor | Done |
| Daily features and leakage test | Next |
| Baselines, normal-behaviour models, LightGBM | Planned |
| Robustness analyses and SHAP | Planned |
| Report and ethics review | Planned |

Dates, owners and review gates are in [`docs/roadmap.md`](docs/roadmap.md). There are no model
results yet; this README will link to them when they exist.

### What the labels look like (`reports/tables/label_summary.csv`)

| Target | Split | Labelled rows | Positive | Episodes |
|---|---|---|---|---|
| A | train / val / test | 22,309 / 9,690 / 17,951 | 37.3% / 47.0% / 45.1% | 355 / 217 / 336 |
| B | train / val / test | 20,911 / 8,827 / 17,281 | 47.8% / 49.2% / 50.5% | 434 / 209 / 382 |

### Reading the metrics honestly

Positives are common after the exclusions (37 to 50% of rows), and the 30-day window is wide. A
**random score** already reaches episode recall of 0.79 (target A) and 0.84 (target B) on the test
years at 1 false alarm per turbine-month, and a constant score gets a PR-AUC equal to the positive
rate. Any model must be compared with these floors (`reports/tables/baselines_dummy.csv`,
decision D-020), and every results table shows them.

## Known limits

- One wind farm only; results may not transfer to other sites or turbine types.
- No measured active power (`tblSCTurGrid` is not in our files), so the power-curve baseline uses
  generator speed against wind speed instead.
- `TimeErr` (fault time) is not documented by the data provider; its meaning is inferred from its
  name and its agreement with SCADA stop time (correlation 0.86).
- Turbine 2304515 is positive on 97% of its validation days; we report results with and without it.
- It is a decision aid for planning inspections, not an automatic shutdown trigger.

The full list, with the evidence, is in [`docs/DATA_TRAPS.md`](docs/DATA_TRAPS.md).

## Dataset

- Hill of Towie wind farm open dataset (RES/TRIG), CC-BY-4.0, 21 Siemens SWT-2.3-VS-82 turbines,
  10-minute SCADA.
- Provisionally pinned to **v2.1.0**: DOI [10.5281/zenodo.22662930](https://doi.org/10.5281/zenodo.22662930)
  (concept DOI for all versions: 10.5281/zenodo.14870021). The brief cited 10.5281/zenodo.14870023,
  which is **v1.0.0** and ends before our 2025 to 2026 data; version to be confirmed
  (`docs/DECISIONS.md` D-004).
- Tables used: `tblSCTurTemp`, `tblSCTurbine`, `tblSCTurPress`, `tblAlarmLog`, `tblDailySummary`.
- TODO: add the exact citation text from the Zenodo record page.

## Quickstart

```bash
git clone <repo-url>
cd <repo>
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pre-commit install
nbstripout --install                 # strips notebook outputs
```

Put the raw monthly CSVs in `data/raw/<year>/`, or download them
(`python scripts/download_data.py --list`, then `python scripts/download_data.py`; all years are
about 15 GB of yearly zips). Then run, in order:

| Command | Produces |
|---|---|
| `python scripts/build_alarm_table.py` | `data/interim/alarm_events.parquet` from the raw alarm logs |
| `python scripts/build_scada_parquet.py --workers 4` | one parquet per SCADA table and year in `data/interim/scada/` |
| `python scripts/build_clean_data.py` | cleaned `scada_<year>.parquet` and `daily_summary.parquet` in `data/interim/clean/`, plus `reports/tables/cleaning_report.csv` |
| `python scripts/build_labels.py` | episodes and 30-day labels for targets A and B in `data/interim/labels/`, plus `reports/tables/label_summary.csv` |
| `python scripts/run_dummy_baselines.py` | constant and random-score floors in `reports/tables/baselines_dummy.csv` |

Metadata only: `python scripts/download_data.py --only Hill_of_Towie_data_dictionary.md --out data/raw/metadata`

Check the code with `pytest` and `ruff check .`.

## Repository map

- `src/wt_predmaint/`: all reusable code (`data`, `features`, `labels`, `models`, `evaluation`, `explain`)
- `configs/protocol.yaml`: the single source of experiment settings (windows, splits, thresholds, seed)
- `scripts/`: command-line entry points for the steps above
- `docs/`: [`roadmap.md`](docs/roadmap.md) (schedule, owners, gates), [`DECISIONS.md`](docs/DECISIONS.md)
  (every decision, with an ID), [`DATA_TRAPS.md`](docs/DATA_TRAPS.md),
  [`modelling_plan.md`](docs/modelling_plan.md), `reference/` (alarm code and data descriptions)
- `notebooks/`: exploration only, named `NN_owner_topic.ipynb`; outputs are never committed
- `reports/{figures,tables}`: generated outputs
- `data/{raw,interim,processed}`: git-ignored, never committed
- `tests/`: unit tests on synthetic data

## Team

| Person | Area |
|---|---|
| Parvez | Data acquisition and cleaning |
| Revael | Features and labelling |
| Kanav | Modelling and tuning |
| Farhan | Evaluation harness and cost analysis |
| Cyrus | Ethics review and report |

Working rules: every decision is logged in `docs/DECISIONS.md` the same day; `configs/protocol.yaml`
changes only together with its decision entry; no results or references are invented; all randomness
is seeded from the config.

## Licence

TODO: no licence chosen yet. The team must decide before any public release.
