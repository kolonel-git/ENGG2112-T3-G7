# Predicting Wind Turbine Degradation from SCADA

**ENGG2112, University of Sydney — Group 7, “AI for Good”**

We study whether 10-minute SCADA sensor readings from 21 Hill of Towie wind
turbines can indicate degradation or warn of an upcoming fault or stoppage. The
main study evaluates a normal-behaviour indicator (O1), 30-day downtime warnings
(O2), and model robustness (O3). A separate exploratory workflow predicts alarm
starts; an alarm is not necessarily a component fault, so its results are kept
separate from the main study.

**Main result:** on the held-out test years, no model was distinguishable from a
random score by PR-AUC, and the validation alert threshold did not maintain the
target false-alarm rate. We do not recommend these models for operational use.
See the [saved test results](reports/TEST_RESULTS.md) or
[plain-language summary](reports/STAKEHOLDER_SUMMARY.md).

## Start here

### For Pratul (tutor)

Please start with [`notebooks/00_combined_full_pipeline.ipynb`](notebooks/00_combined_full_pipeline.ipynb).
It is the main guided Jupyter notebook and shows how the project fits together:
each stage calls the reusable code and scripts used by the full pipeline.

For additional data visualization and exploratory studies, see the notebooks in
[`exploratory_study/`](exploratory_study/). The
[exploratory-study README](exploratory_study/README.md) links to its notebook
entry point and explains how to set up and use that workflow.

### Review the results

Start with the [stakeholder summary](reports/STAKEHOLDER_SUMMARY.md), then browse
the [test results](reports/TEST_RESULTS.md) and
[generated figures](reports/figures/README.md). These are saved in the repository;
reviewing them does not require downloading the raw data or rerunning the models.

### Run the main-study notebook

1. Follow the [quickstart](#quickstart) to set up Python and dependencies.
2. Open [`notebooks/00_combined_full_pipeline.ipynb`](notebooks/00_combined_full_pipeline.ipynb)
   and select the repository's `.venv` Python environment as its kernel.
3. Read the **Run controls** at the top, then choose **Run All**. With the default
   settings, the notebook displays saved results; it does not rebuild the data or
   rerun the frozen test evaluation.
4. To reproduce stages, enable the relevant switches in the notebook and follow
   its numbered sections. Rebuilding requires the raw input files. The held-out
   test run is already frozen; do not rerun it to tune models.

The notebook is a step-by-step front end to the maintained code in `scripts/`
and `src/`; it avoids duplicating the modelling implementation.

### Choose another workflow

| Workflow | Purpose | Guide |
|---|---|---|
| **Main study** | Daily 30-day warnings for fault-time target A and SCADA-stop target B | [Notebook](notebooks/00_combined_full_pipeline.ipynb) · [Protocol](configs/protocol.yaml) |
| **Iteration 2** | Exploratory alarm-start forecasting, visualization and classifier comparisons | [Start Here notebook](exploratory_study/Iteration2/00_START_HERE.ipynb) · [Run guide](exploratory_study/Iteration2/README.md) |
| **All-turbine temporal CV** | Separate six-hour icing-alarm experiment | [CV guide](exploratory_study/Iteration2/notebooks/06_all_turbine_cross_validation/README.md) |
| **Additional model comparison** | Validation-only classifiers using the main study's daily labels | [Notebook](additional_models_main_protocol.ipynb) |

These workflows use different targets, splits and metrics. Do not combine their
scores as if they were one experiment. Iteration 2 evaluation years have already
been inspected in exploratory work; they are not independent confirmation of the
main study's findings.

## Where things are

| Location | Contents |
|---|---|
| [`notebooks/`](notebooks/) | Main-study guided pipeline and diagnostics |
| [`src/wt_predmaint/`](src/wt_predmaint/) | Reusable data, feature, label, model, evaluation and reporting code |
| [`scripts/`](scripts/) | Main-study command-line pipeline steps |
| [`configs/protocol.yaml`](configs/protocol.yaml) | Main-study years, windows, quality rules, model settings and seed |
| [`data/`](data/) | Raw, interim and processed data; generally git-ignored |
| [`reports/`](reports/) | Saved results, figures, tables and report |
| [`docs/`](docs/) | Setup, methods, decisions and project notes |
| [`tests/`](tests/) | Synthetic software-correctness and leakage tests |
| [`exploratory_study/Iteration2/`](exploratory_study/Iteration2/) | Separate exploratory notebooks, data layout and outputs |

## Using Iteration 2 as an exploratory workspace

<details>
<summary>Setup, data, and notebook instructions</summary>

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

</details>

## Main study: problem and methods

The main study uses open SCADA data from the Hill of Towie wind farm in Scotland:
21 Siemens SWT-2.3-VS-82 turbines, with 10-minute readings from 2016 to April 2026.
It asks whether routine turbine sensor data can indicate degradation early enough
to support maintenance planning, and whether those warnings can be explained.

Each turbine records temperatures, pressures, rotor speed, wind speed and alarms.
Many alarms are routine: three codes appear in more than 99% of 30-day windows,
so an alarm is not by itself evidence that a turbine is failing.

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
```

Activate the environment for your shell:

```powershell
.\.venv\Scripts\Activate.ps1
```

```bash
source .venv/bin/activate
```

Then install the project tools:

```bash
python -m pip install -e ".[dev]"
pre-commit install
nbstripout --install                 # strips notebook outputs
```

The root notebook is the recommended way to follow the project in clearly
labelled steps. The commands below rebuild the main study from raw data. The
data are not bundled; downloading them requires about 15 GB. Put the raw monthly
CSVs in `data/raw/<year>/`, or download them:

```bash
python scripts/download_data.py --list
python scripts/download_data.py
```

Then run the pipeline in order:

<details>
<summary>Show all command-line pipeline steps</summary>

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

</details>

The report is `reports/report/main.tex` (it needs `refs.bib` and the logo image beside it; figures
and tables come from the commands above). Metadata only:
`python scripts/download_data.py --only Hill_of_Towie_data_dictionary.md --out data/raw/metadata`

The timeline plot uses feature values by default; pass `--scores path\to\scores.parquet`
to plot a score instead (the file must contain `StationId`, `Day` and the selected `--column`).
It builds a gap-free daily calendar, so days without labels or rows remain visible. A score can
still appear on an unlabelled day; the bands distinguish label status, data split and missing
values, and the command prints day counts and missing-value reasons alongside the image.

Check the code with `pytest` and `ruff check .`.

## Reference documents

The [repository map above](#where-things-are) summarizes the main folders.
For project details, see:

- [Roadmap](docs/project/roadmap.md) (schedule, owners and gates) and
  [decision log](docs/project/DECISIONS.md) (decisions and their IDs).
- [Team checklist](docs/TEAM_CHECKLIST.md) and
  [data traps](docs/project/DATA_TRAPS.md).
- [Features and labels](docs/FEATURES_AND_LABELS.md) and
  [baseline split](docs/BASELINE_SPLIT.md).
- [Original modelling plan](docs/project/modelling_plan.md); the roadmap and
  decision log supersede it.
- Alarm-code and dataset descriptions in `docs/reference/`.
- [Final report](reports/report/main.tex), [test results](reports/TEST_RESULTS.md),
  [stakeholder summary](reports/STAKEHOLDER_SUMMARY.md), and generated
  [figures](reports/figures/README.md).

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
