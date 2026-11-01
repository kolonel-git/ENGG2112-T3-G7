# ENGG2112 Group 7 — repository guide

This repository contains our main wind-turbine downtime study and a separate
**Iteration 2 exploratory alarm-forecasting workflow**. This page explains how to
review the submitted work, where its evidence lives, and how to run either workflow.
The two workflows ask different questions and use different labels, splits and
metrics; their scores should not be combined into one results table.

## Suggested reading order for the marker

1. Read the [main findings below](#findings) and the
   [stakeholder summary](reports/STAKEHOLDER_SUMMARY.md) for the project outcome.
2. Inspect [saved test results](reports/TEST_RESULTS.md), the generated
   [figures index](reports/figures/README.md), and the
   [report source](reports/report/main.tex). Reviewing saved evidence does not
   require retraining models or downloading all raw data.
3. Use the [labels/features reference](docs/FEATURES_AND_LABELS.md),
   [decision log](docs/project/DECISIONS.md), and
   [data traps](docs/project/DATA_TRAPS.md) to trace methodological choices.
4. Explore Iteration 2 through [Start Here](exploratory_study/Iteration2/00_START_HERE.ipynb)
   and its [run guide](exploratory_study/Iteration2/README.md). This is additional
   investigation of alarm starts, not a replacement for the main study's frozen results.
5. For implementation review, see the reusable code in `src/wt_predmaint/`,
   command-line entry points in `scripts/`, and synthetic checks in `tests/`.

## Choose a workflow

| Workflow | Question and scope | Start here |
|---|---|---|
| **Main study / original protocol** | Daily 30-day downtime warning, targets A (fault time) and B (SCADA stops), fleet training and fixed chronological evaluation | [Main-study quickstart](#quickstart), [protocol](configs/protocol.yaml), [saved test results](reports/TEST_RESULTS.md) |
| **Iteration 2 / exploratory notebooks** | Forecast new alarm starts at configurable horizons; inspect sensor patterns and compare classifiers | [Run guide](exploratory_study/Iteration2/README.md), [notebook directory](exploratory_study/Iteration2/notebooks/README.md) |
| **Iteration 2 / all-turbine temporal CV** | Separate six-hour icing-alarm experiment with development folds and a disabled reserved-test step | [CV guide](exploratory_study/Iteration2/notebooks/06_all_turbine_cross_validation/README.md) |
| **Additional main-protocol comparisons** | Post-hoc validation-only classifier comparisons using the original daily A/B task | [Additional-model notebook](additional_models_main_protocol.ipynb), [scope below](#additional-validation-only-model-comparison-d-071) |

An alarm start is not necessarily a component fault. Operational codes can be
predictable without providing useful degradation warnings. Iteration 2 evaluation
years have already been inspected in exploratory work; they are not new independent
confirmation of the main report's findings.

## Using Iteration 2 as an exploratory workspace

The actual project directory is **`exploratory_study/Iteration2/`**. The extra directory
level is intentional; keep it so existing notebook paths continue to resolve.
The original root-level preparation scripts are not prerequisites for this workflow.

### Environment and data

Install Python 3.10 or newer and the VS Code Python/Jupyter extensions. Open
[Iteration2.code-workspace](Iteration2.code-workspace), then run from the repository root:

```powershell
cd exploratory_study/Iteration2
.\setup_environment.ps1
```

Select `exploratory_study/Iteration2/.venv/Scripts/python.exe` as the notebook kernel.
The setup script creates the environment and installs
[Iteration 2 dependencies](exploratory_study/Iteration2/requirements.txt); recreate copied
virtual environments on a new computer. Keep this environment distinct from the
root environment used for the original protocol.

Open [00_START_HERE.ipynb](exploratory_study/Iteration2/00_START_HERE.ipynb) and run its
cells first. It audits available files and recommends a starting point; it does
not convert data or train models. Raw data are large and are not bundled as
ordinary Git files. Processed Iteration 2 Parquets use Git LFS: with Git LFS
installed, run `git lfs install` and `git lfs pull` from the repository root, then
check availability with Start Here. Raw CSVs and Excel inputs must be supplied
separately if rebuilding from those stages.

### Notebook sections and run order

All folders below are relative to `exploratory_study/Iteration2/`.

| Folder / section | Purpose | Run when |
|---|---|---|
| `notebooks/01_data_export/` | Monthly raw SCADA/alarm CSVs to yearly turbine Excel exports | Starting with `Data/Raw/<year>/` |
| `notebooks/02_parquet_setup/` | Excel exports to per-year/per-turbine source sensor/alarm Parquets; includes a preview notebook | Starting with `Data/Interim/turbine_exports/` |
| `notebooks/03_preprocessing/` | Source Parquets to regular ten-minute modelling data with metadata | Source Parquets exist under `Data_Parquets/` |
| `notebooks/04_data_visualisation/` | Inspect sensor/alarm behaviour over a day, week, month or year | Optional after preprocessing |
| `notebooks/05_model_training/` | Independent alarm-group, horizon and feature-mode classifier comparisons | Processed Parquets exist under `Data_Parquets_10min/` |
| `notebooks/06_all_turbine_cross_validation/` | Separate temporal-CV experiment and `cv_pipeline.py` implementation | Advanced alternative after preprocessing, not a continuation of Stage 5 |
| `notebooks/shared/` | Reusable helpers imported by notebooks | Do not run these modules as sequential notebooks |
| `evaluate_cross_validation_results.ipynb` | View saved development CV metrics and predictions | CV outputs already exist; no retraining required |
| `Data_Parquets_CV/` | Derived feature caches for the CV workflow | Generated by CV |
| `reports/` | Iteration 2 models, configurations, predictions, metric tables and plots | Generated by the selected experiment |
| `DevNotes/`, `CONCLUSIONS.md` | Development notes and historical experiment interpretation | Context; historical prose does not automatically update after reruns |

If `Data_Parquets_10min/` is populated, **skip Stages 1–3**. You do not need to
run every visualization, alarm group, horizon or feature variant.

### Recommended first experiment

Open [all alarms / ten minutes / all features](exploratory_study/Iteration2/notebooks/05_model_training/all_alarms/01_horizon_10min/01_train_all_features.ipynb).
Review the configuration cells, select the correct kernel, then choose
**Restart Kernel → Run All**. Current Stage 5 defaults pool five turbines
(2304510–2304514), train on 2016–2018, select thresholds on 2020, and evaluate
2021. They use the preceding two hours of sensor readings. A plotted turbine is
only the displayed example; read the configured turbine list to identify training scope.

Choose a different target or horizon through the
[training directory](exploratory_study/Iteration2/notebooks/05_model_training/README.md).
All-alarm notebooks include operational codes, including 20 and 25. Specific-group
notebooks cover pitch lubrication, cable untwisting and icing. All-features,
selected-features and variant notebooks are alternative experiments, not mandatory
successive steps. Restart and rerun after changing settings.

Each run prints its directory under `exploratory_study/Iteration2/reports/`. Inspect:

- `configuration.json` for the exact target, years, turbines and feature settings.
- `metrics.csv` for validation/test metrics; compare average precision with prevalence.
- `precision_recall.png`, `confusion_matrices.png`, and `roc_curves.png` where available.
- `trained_models.joblib` for fitted estimators and preprocessing, and
  `test_predictions.parquet` for scores. Some notebooks also export detailed binary metrics.

Identical configurations can reuse the same output directory and replace earlier
outputs; the configuration hash is not an archive of every rerun.

The separate [all-turbine CV workflow](exploratory_study/Iteration2/notebooks/06_all_turbine_cross_validation/README.md)
uses development data from available 2016–2023, with outer evaluations in
2021/2022/2023 and earlier calibration years. Its default icing target is
8230/8234/8235/8236; the historical filename's “8200 family” is broader than the
current configured code list. `RUN_RESERVED_TEST = False` leaves 2024–2025
unopened by that workflow. Those years have been used elsewhere in this project,
so they are not globally unseen data. Consult the CV guide before enabling that step.

## Overall repository structure

| Location | What it contains |
|---|---|
| `src/wt_predmaint/` | Main-study reusable loading/cleaning, labels, features, models, evaluation, explanations and reporting |
| `configs/` | Original protocol: years, windows, quality rules, model settings and seeds |
| `scripts/` | Main-study command-line pipeline; see the command table below |
| `notebooks/` | Original-protocol exploration and diagnostic notebooks; separate from Iteration 2 |
| `data/` | Main-study raw/interim/processed data, generally excluded from Git |
| `reports/` | Main-study saved evidence, figures, tables, stakeholder summary and LaTeX report |
| `docs/` | Usage guides, label definitions, decisions, limitations and project history |
| `tests/` | Software correctness/leakage tests; these are not model performance results |
| `exploratory_study/Iteration2/` | Separate exploratory notebook workflow, environment, data layout and outputs |
| `additional_models_main_protocol.ipynb` | Additional validation-only experiments on original A/B labels |
| `.github/`, `pyproject.toml`, `CONTRIBUTING.md` | CI, package/dependency configuration and contribution rules |

The sections below document the **main study**: its problem, methods, findings,
limitations, dataset, setup, ordered script commands, team roles and licence status.
Use the saved report evidence for marking; reproducing the full large-data pipeline
is optional and requires the corresponding source files. Test results in the main
study are already frozen; do not rerun its test step to select or tune models.

---

# Predicting Wind Turbine Degradation from SCADA

**Explainable predictive maintenance on open wind-farm data.**
ENGG2112 (University of Sydney), Group 7, "AI for Good".

Wind turbines are expensive to repair and expensive to leave idle. An unplanned breakdown means a
stopped turbine, a crew sent out at short notice and electricity that is never sold. This project
asks whether the sensor data a turbine already records can warn us **before** it fails, and whether
we can explain the warning well enough for a maintenance engineer to trust it.

## Headline result

**On the test years (2023 to April 2026) no model could be told apart from a random score.** Every
95% interval for PR-AUC contains the random-score value. The model that looked best on the tuning
years fell back to the random level on the test years, and the alert threshold chosen on the tuning
years did not hold a fixed false-alarm rate. We therefore do **not** recommend using any of these
models for operations. Details, numbers and caveats are in
[Findings](#findings), [`reports/TEST_RESULTS.md`](reports/TEST_RESULTS.md) and
[`reports/STAKEHOLDER_SUMMARY.md`](reports/STAKEHOLDER_SUMMARY.md) (a plain-language version).

## The problem

Each turbine logs hundreds of readings every 10 minutes (temperatures, pressures, rotor speed, wind
speed) plus a log of alarms. Many alarms are routine events that happen to every turbine all the
time (three alarm codes are present in more than 99% of 30-day windows), so "an alarm fired" is not
the same as "something is wrong". We need a better definition of failure, and a way to see trouble
building up.

We work with real data: the Hill of Towie wind farm in Scotland, 21 Siemens SWT-2.3-VS-82 turbines,
about ten years (2016 to April 2026).

## What we built

Three objectives from the project brief, plus explanation:

| | Objective | Plain-language question | Status |
|---|---|---|---|
| **O1** | Degradation indicator | Learn what "normal" looks like for a turbine (for example, how warm the gearbox should be at a given wind speed). How far today's readings drift from normal is a health score. | Built; weak, and drifts between years unless made relative to the fleet |
| **O2** | Fault and downtime prediction | Will this turbine have a fault or a long stoppage in the next 30 days? | Built and tested once on the test years; not better than random |
| **O3** | Robustness | Do the predictions still hold for a turbine the model never saw, or with a sensor group missing? | Leave-one-turbine-out and sensor-group dropout done (validation years); upgrade analysis not done; a train-years year-shift check is in the exploratory runs |
| | Explanation | Which signals does the model lean on? | SHAP done, stable across turbines; model attribution, not proof of cause |

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
   There is no 2019 temperature or pressure data. Scalers and models are fitted on training years; alert thresholds are selected on
   validation years. The test years are scored once, for frozen settings.
5. **Build features.** Daily summaries of the 10-minute readings, a 7-day window, alarm counts, and
   the gap between observed and expected values from the normal-behaviour model.
6. **Train models.** A constant and a random score as floors, simple rules (recent faults, alarm
   counts), a healthy-curve baseline (rotor speed against wind), a per-turbine z-score (R1), logistic
   regression (B4), the normal-behaviour indicator (R2) and class-weighted LightGBM (L1). The
   optional autoencoder was not built.
7. **Evaluate fairly.** PR-AUC against the random-score value, recall at a budget of 1 false alarm
   per turbine-month next to the random-score recall, and median lead time (days of warning), each
   with confidence intervals that resample turbines, not rows.
8. **Stress test and explain.** Leave-one-turbine-out, sensor-group dropout and SHAP, on the
   validation years only.

## Where we are

| Stage | Status |
|---|---|
| Repo, tooling, CI, decision log | Done |
| Raw data exploration and trap list | Done |
| Cleaning layer (flags for outage months and suspect wind sensors) | Done |
| Labels for targets A and B | Done and verified |
| Evaluation harness and random-score floor | Done |
| Daily features and look-ahead test | Done |
| Baselines B1 to B4, R1 and the two floors | Done (validation and test) |
| Normal-behaviour model and indicator (O1, R2) | Done; frozen (D-055, D-057, D-058) |
| LightGBM (L1) | Done; settings frozen on validation (D-060) |
| Single test-year run | Done once (D-061); not repeated |
| Exploratory runs: fleet-relative features, low-capacity models | Done on validation years only (D-064, D-065, D-066, D-068); leads for future work, not tested on the test years |
| SHAP and robustness | Done on validation years (D-062) |
| Final report | In progress: `reports/report/main.tex` (items still to confirm are marked in red) |
| Ethics review | In progress (Cyrus) |

Dates, owners and review gates are in [`docs/project/roadmap.md`](docs/project/roadmap.md); the open items per
person are in [`docs/TEAM_CHECKLIST.md`](docs/TEAM_CHECKLIST.md).

### What the labels look like (`reports/tables/label_summary.csv`)

| Target | Split | Labelled rows | Positive | Episodes |
|---|---|---|---|---|
| A | train / val / test | 22,309 / 9,690 / 17,951 | 37.3% / 47.0% / 45.1% | 355 / 217 / 336 |
| B | train / val / test | 20,911 / 8,827 / 17,281 | 47.8% / 49.2% / 50.5% | 434 / 209 / 382 |

## Findings

PR-AUC (area under the precision-recall curve, full fleet). A model with no skill scores about the
fraction of positive days, which is what the random score gets:

| Model | A validation | A test | B validation | B test |
|---|---:|---:|---:|---:|
| Random score | 0.465 | 0.450 | 0.483 | 0.506 |
| R1 z-score | 0.577 | 0.448 | 0.588 | 0.499 |
| B4 logistic regression | 0.447 | 0.462 | 0.451 | 0.563 |
| R2 normal-behaviour indicator | 0.496 | 0.440 | 0.533 | 0.503 |
| L1 LightGBM | 0.460 | 0.492 | 0.456 | 0.556 |

Full tables with all models, 95% intervals, episode recall and false alarms are in
`reports/TEST_RESULTS.md`, `reports/tables/t0_validation.csv` and `reports/tables/baselines_*.csv`.

- **No model's test PR-AUC interval excludes the random-score value** (0.450 for A, 0.506 for B).
- **R1, best on validation, falls to the random level on test.** The supervised models were below
  the random value on validation and are above it in point estimate on test, within their intervals.
- **The validation threshold does not carry to a fixed false-alarm rate.** R1 and R2 reach recall of
  0.55 to 0.68 on test, but at 2.1 to 2.5 false alarms per turbine-month, about twice the budget.
  No model reaches the random-score recall (0.795 and 0.839) at a comparable rate.
- **O1 indicator:** high fit for the stator windings (R-squared 0.95), low for the hydraulic oil
  (0.11); residual drift between the training and validation years (up to 0.9 standard deviations)
  falls to about 0.2 when residuals are made relative to the fleet; only slightly higher before
  episodes than far from them.
- **Exploratory follow-up (validation years only, D-064):** fleet-relative features raised some validation scores (for example logistic regression for target A from 0.447 to 0.483, LightGBM for target B from 0.456 to 0.516), but every interval still contains the random-score value and the train-year year-shift check shows no consistent gain. A lead for future work, not a result; not scored on the test years.
- **Second exploratory round (validation years only, D-065):** each turbine's own trailing 90-day median as the baseline gave no interval beyond the random-score margin either. A turbine-ranking view (precision among the top 3 turbines each day against chance) was above chance for the power-curve baseline B3 on both targets (about +0.07 to +0.08, intervals above zero); with 92 intervals examined, this is a lead to confirm on new data, not a result.
- **Third exploratory round (validation years only, D-066):** features ranked on training years only and pruned to 10, 25, 50 or 100 columns (three ranking methods), plus refits with one feature group removed. Pruned models changed validation PR-AUC by about -0.05 to +0.03 against all columns and none beat the random-score margin; 50 to 100 columns are enough, so the model can be simplified without a clear loss. Pressure and hub sensors and days since the last fault carry the most gain; the columns that are stable across years for target B are seasonal temperatures.
- **Label sensitivity (validation years only, D-068):** 27 label variants (positive window 7, 14 or 30 days; stricter fault and stop definitions; target B without weather stops), each judged against its own random-score floor. Narrower labels did not help the learned models (LightGBM above the floor in 1 of 27 variants, logistic in 0). The few variants above the floor come from simple rules, mainly the power-curve baseline B3, which ranks turbines above chance in 16 of 18 target B variants; with 189 intervals examined and heavy overlap between variants this is a lead to check on new data.
- **Robustness (validation):** removing any sensor group changes PR-AUC by at most 0.023; pooled
  PR-AUC with a turbine left out of training is at or below the pooled prevalence.
- **Explanation:** SHAP ranks ambient and weather-linked inputs, pressure sensors and days since the
  last fault above component temperatures; the ranking is stable across turbines (rank correlation
  0.86 for A, 0.93 for B). This describes the model, not causes, and the model scores at the random
  level.

### Reading the metrics honestly

Positives are common after the exclusions (37 to 50% of rows), and the 30-day window is wide. A
**random score** already reaches episode recall of 0.76 to 0.84 at 1 false alarm per turbine-month,
and a constant score gets a PR-AUC equal to the positive rate. Any model must be compared with these
floors, and every results table shows them (decision D-020). The 95% intervals resample turbines
and do not include the uncertainty of choosing the model or the threshold. The episodes are inferred
operational episodes, not confirmed component failures.

## Known limits

- One wind farm only; results may not transfer to other sites or turbine types.
- No measured active power (`tblSCTurGrid` is not in our files), so the power-curve baseline uses
  generator speed against wind speed instead.
- `TimeErr` (fault time) is not documented by the data provider; its meaning is inferred from its
  name and its agreement with SCADA stop time (correlation 0.86).
- Turbine 2304515 is positive on 97% of its validation days; we report validation results with and
  without it, and did not score the variant without it on the test years.
- No confirmed failure or maintenance records; "fault" and "degradation" are inferred.
- The upgrade analysis needs the AeroUp installation dates, which are not in our files; a year-shift check
  was run only on training years, as part of the exploratory runs.
- It is a decision aid for planning inspections at most, not an automatic shutdown trigger, and on
  our evidence not yet a useful one.

The full list, with the evidence, is in [`docs/project/DATA_TRAPS.md`](docs/project/DATA_TRAPS.md).

## Dataset

- Hill of Towie wind farm open dataset (RES/TRIG), CC-BY-4.0, 21 Siemens SWT-2.3-VS-82 turbines,
  10-minute SCADA.
- Provisionally pinned to **v2.1.0**: DOI [10.5281/zenodo.22662930](https://doi.org/10.5281/zenodo.22662930)
  (concept DOI for all versions: 10.5281/zenodo.14870021). The brief cited 10.5281/zenodo.14870023,
  which is **v1.0.0** and ends before our 2025 to 2026 data; version to be confirmed
  (`docs/project/DECISIONS.md` D-004).
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
| `python scripts/build_features.py` | daily feature table `data/interim/features/daily_features.parquet` and `reports/tables/feature_summary.csv` |
| `python scripts/check_feature_leakage.py` and `--nbm` | re-computes random feature rows (and the normal-behaviour columns) with the future deleted and fails if any value changes |
| `python scripts/build_nbm.py` | normal-behaviour residuals `data/interim/features/nbm_residuals.parquet` and the `nbm_*` tables |
| `python scripts/make_baseline_table.py` | all registered models on validation, both fleet variants: `reports/tables/t0_*.csv` |
| `python scripts/tune_lgbm.py` | the six-setting LightGBM grid on validation: `reports/tables/lgbm_tuning.csv` (the winners are frozen in `configs/protocol.yaml`, D-060) |
| `python scripts/run_baselines.py` | floors and models in `reports/tables/baselines_<target>.csv`; **`--frozen` also scores the test years: it was run once (D-061) and must not be repeated** |
| `python scripts/write_test_results.py` | `reports/TEST_RESULTS.md` and `reports/tables/test_results.csv` from the saved test run |
| `python scripts/analyse_alert_budgets.py` | validation alert-burden tables over a budget grid (run `make_baseline_table.py` first) |
| `python scripts/run_shap.py` | SHAP importance and its stability across turbines (validation years only) |
| `python scripts/run_robustness.py` | sensor-group dropout and leave-one-turbine-out (validation years only) |
| `python scripts/run_exploratory.py` | the 42 exploratory configurations on validation years and the train-year year-shift check: `reports/tables/exploratory_validation.csv` (D-064) |
| `python scripts/run_exploratory.py --round 2` and `python scripts/run_ranking.py` | the self-baseline configurations and the turbine-ranking metric on validation years: `reports/tables/exploratory_self_baseline.csv`, `ranking_validation.csv` (D-065) |
| `python scripts/run_exploratory.py --round 3` | feature ranking, pruning and group removal on validation years: `reports/tables/exploratory_selection.csv`, `feature_ranking_a.csv`, `feature_ranking_b.csv` (D-066) |
| `python scripts/run_label_sensitivity.py` | rebuilds the labels in memory under 27 variants and reruns the frozen models on validation years: `reports/tables/label_sensitivity.csv` (D-068) |
| `python scripts/make_report_figures.py` | all report figures in `reports/figures/01_data` to `07_explain_robustness` and the index `reports/figures/README.md` |
| `python scripts/write_report_tables.py` | the LaTeX tables in `reports/report/tables/` from the result tables |
| `python scripts/plot_risk_timeline.py --target a --station <id> --column <col>` | `reports/figures/timeline_<target>_<station>_<column>.png`: one turbine's daily values with status, split and missing-value bands |

The report is `reports/report/main.tex` (it needs `refs.bib` and the logo image beside it; figures
and tables come from the commands above). Metadata only:
`python scripts/download_data.py --only Hill_of_Towie_data_dictionary.md --out data/raw/metadata`

The timeline plot uses feature values by default; pass `--scores path\to\scores.parquet`
to plot a score instead (the file must contain `StationId`, `Day` and the selected `--column`).
It builds a gap-free daily calendar, so days without labels or rows remain visible. A score can
still appear on an unlabelled day; the bands distinguish label status, data split and missing
values, and the command prints day counts and missing-value reasons alongside the image.

Check the code with `pytest` and `ruff check .`.

## Repository map

- `src/wt_predmaint/`: all reusable code (`data`, `features`, `labels`, `models`, `evaluation`,
  `explain`, `reporting`)
- `configs/protocol.yaml`: the single source of experiment settings (windows, splits, thresholds,
  model settings, seed)
- `scripts/`: command-line entry points for the steps above
- `docs/`: [`roadmap.md`](docs/project/roadmap.md) (schedule, owners, gates), [`DECISIONS.md`](docs/project/DECISIONS.md)
  (every decision, with an ID), [`TEAM_CHECKLIST.md`](docs/TEAM_CHECKLIST.md),
  [`DATA_TRAPS.md`](docs/project/DATA_TRAPS.md),
  [`FEATURES_AND_LABELS.md`](docs/FEATURES_AND_LABELS.md) (every label and feature column),
  [`BASELINE_SPLIT.md`](docs/BASELINE_SPLIT.md) (who builds which baseline, and what each must deliver),
  [`modelling_plan.md`](docs/project/modelling_plan.md) (the original plan; the roadmap and decision log
  supersede it), `reference/` (alarm code and data descriptions)
- `notebooks/`: exploration only, named `NN_owner_topic.ipynb`; outputs are never committed
- `reports/`: [`report/main.tex`](reports/report/main.tex) (final report),
  [`TEST_RESULTS.md`](reports/TEST_RESULTS.md), [`STAKEHOLDER_SUMMARY.md`](reports/STAKEHOLDER_SUMMARY.md),
  [`REVAEL_FEATURES_AND_LABELS.md`](reports/REVAEL_FEATURES_AND_LABELS.md) (reviewed labels/features
  section and traceability map), plus `figures/` and `tables/` (generated outputs)
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

Working rules: every decision is logged in `docs/project/DECISIONS.md` the same day; `configs/protocol.yaml`
changes only together with its decision entry; no results or references are invented; all randomness
is seeded from the config; the test years are scored once.

## Licence

TODO: no licence chosen yet. The team must decide before any public release.


## Additional validation-only model comparison (D-071)

[additional_models_main_protocol.ipynb](additional_models_main_protocol.ipynb) at the project root trains random forest, Extra Trees and histogram gradient boosting, with the original logistic/LightGBM controls and the shared constant/random floors. It reuses the main daily A/B labels/features, configured train/validation years, false-alarm-budget threshold and turbine-bootstrap evaluator. It does not score test years or overwrite the frozen report tables. Results are isolated under `reports/additional_models_main_protocol/<run-id>/`.

This is post-hoc exploratory validation work, not new held-out evidence. See D-071 in the decision log and the notebook's coverage audit; currently cached rows may differ from the published run. The notebook directly calls the evaluator and does not depend on the frozen model registry.
