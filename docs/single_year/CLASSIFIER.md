# Train and visualise a pooled-turbine single-year classifier

This guide pools available turbines within one year. For training on just one
turbine, use [the single-turbine guide](SINGLE_TURBINE.md).

## Run order

Follow [general setup](../general/USAGE.md). Build the shared data through
`build_features.py` and run `check_feature_leakage.py` before interpreting model
results. Then, from the repository root with that Python environment:

```powershell
python scripts/single_year/all_turbines/train_classifier.py --year 2018
python scripts/single_year/all_turbines/plot_classifier.py --year 2018
```

Both scripts default to 2018. For a quicker trial, add `--resamples 100` to the
training command. Plot just one turbine with `--station 2304510`. The plotting
script normally produces timelines for every scored turbine. It does not retrain.

## What is trained

A separate logistic regression is fitted for each target and chronological fold.
Target A predicts future daily-fault-time episodes; target B predicts future
SCADA-stop episodes. The target is whether a new episode starts within the next
30 days, using the existing eligible-day labels. A score is generated with
`predict_proba(... )[:, 1]`.

The pipeline uses median imputation, standard scaling, and L2-regularised logistic
regression with `C=1`, balanced class weights, and a maximum of 3,000 iterations.
There is no hyperparameter search. Nonconvergence stops the run rather than
silently publishing results. Scores are useful for ranking but **are not
calibrated fault probabilities**, particularly because fitting is class weighted
and the labels exclude several types of day.

## Features actually supplied to the classifier

Candidates come from the existing `feature_columns()` definition, before labels
are joined. The exact list depends on the source sensors and training-fold
coverage. No future label/status/split column is a model input.

| Feature group | Inputs and examples |
|---|---|
| Producing-state sensor summaries | `op_*` daily means of available SCADA mean sensors during producing intervals: temperatures, pressures, wind, generator/rotor speed, and other sensors selected by the existing SCADA reader. Requires enough producing intervals. |
| Daily sensor extremes | `max_*` daily maxima of available sensor maximum columns. |
| Operating conditions | `frac_producing`, `frac_idle`, `frac_stopped`, `wind_mean`, `wind_std`, `rpm_mean`. |
| Recent trends | Trailing seven-day means and standard deviations (`w7mean_*`, `w7std_*`) of sensor summaries and the operating-condition features. |
| Alarm history | Trailing seven-day counts for configured alarm codes, tiers, and totals (`alarm_*`). No future alarms or end times are supplied. |
| Fault history | `days_since_fault_day` and `fault_days_30`, using known fault-day history up to the prediction day. These are legitimate historical predictors, not future outcomes; their usefulness depends on timely availability in deployment. |

Excluded: turbine ID, calendar date, labels/status/split, interval counts,
producing counts, masked-value fraction, outage flag, suspect-wind flag, and
boolean columns, following the existing feature selector. Turbine ID is a join
key, not a predictor. No normal-behaviour residuals or SHAP values are created.

Within each fold, columns with more than 80% missing training values or fewer
than two distinct observed training values are dropped. Change the coverage rule
with `--max-missing`. Infinities are treated as missing. All selection, imputation
medians, scaling statistics, and model weights are fitted on **training rows
only**, never calibration/test rows. Missing sensor cells are imputed; labelled
days lacking an entire feature row cause a clear error instead of silent removal.

`features.csv` is the definitive per-target, per-fold feature audit: it lists
each candidate, training missing share, distinct-value count, whether used, and
its fitted coefficient. Dropped features have no coefficient. Counts can differ
across targets/folds. A positive standardized coefficient raises the score with
other inputs fixed; correlated features make individual weights unstable and
these coefficients are not causal explanations.

The first completed local 2018 run retained the following inputs:

| Group | Number used per model |
|---|---|
| Producing-state sensor means | 62 |
| Daily sensor maxima | 35 |
| Operating-condition summaries | 6 |
| Seven-day trends | 205 |
| Alarm counts | 11–12 |
| Fault-history features | 2 |
| **Total** | **321–322** |

This is a recorded run, not a fixed feature count: rebuilding source data or
changing coverage settings may change selection. The initial run used 100
bootstrap draws for verification. Its results varied substantially across folds,
including test false-alarm rates exceeding the budget; it does not establish
reliable prediction skill. Inspect `metrics.csv` and the figures before drawing
conclusions.

## Three separate time periods

The older dummy-only runner needed calibration/test periods. Fitting a classifier
also needs a training period. This runner therefore uses a different schedule:

- Test on eligible days in **August, September, and October**, separately.
- For each test month, start a calibration block 90 days earlier. Keep calibration
  days only if their full configured label horizon ends before the test starts.
- Keep training days only if their full label horizon ends before calibration
  starts. This creates a second temporal purge.
- Only use rows within the selected year; test horizons must end before the next
  year. All turbines share date boundaries. Later folds may train on earlier
  folds' days only after their outcome horizons have matured.

The full label horizon is currently 60 days, so approximately 30 days remain for
calibration, with a 60-day train-to-calibration gap and a 60-day calibration-to-test
gap. Sparse source coverage or labels can make a fold unusable. Empty folds and
folds without both training/calibration classes are printed and recorded as
skipped. No successful folds means no new output is published.

The threshold is selected on calibration scores to meet the configured false-alarm
budget, then frozen for test. Constant and random baselines are rerun on the exact
same calibration/test rows. Compare these new same-fold results with the classifier;
the earlier `baselines_dummy_cv_2018.csv` uses different test periods.

## Outputs

Training writes to `data/processed/single_year/2018/all_turbines/` by default:

| File | Contents |
|---|---|
| `metrics.csv` | Per-target/fold/model metrics and turbine-bootstrap intervals. |
| `predictions.csv` | Eligible test-day scores, labels, alerts, thresholds, turbine IDs, folds, targets, and models. No in-sample training scores are mixed in. |
| `features.csv` | Exact feature-selection and coefficient audit. |
| `episode_detections.csv` | Fold-episode evaluability, detections, and lead times. |
| `a_fold1.joblib`, etc. | Fitted preprocessing/model, feature names, alert threshold, and target/fold/year identifiers. Only load trusted model files. |
| `run.json` | Snapshot of protocol, seed, actual date bounds, coverage rule, bootstrap draws, and skipped folds. |

Reruns replace files of the same name; unrelated files are not deleted. Files for
previously successful folds may remain if those folds are skipped on a later
run; use `run.json` to identify the current folds, or select a fresh `--output-dir`.
These are fold-specific research models, not a final full-year deployment model.

## Figures

PNG files are saved to `reports/figures/single_year/2018/all_turbines/` by default:

| Figure | How to use it |
|---|---|
| Fold comparison | Compare average precision, episode recall, lead time, and false alarms with confidence bounds. Check actual false alarms against the calibration budget. |
| Fold diagnostics | Precision–recall curve, episode-recall/false-alarm tradeoff, and daily confusion matrix for logistic regression. Crosses mark the frozen operating thresholds. Test threshold sweeps are descriptive, not a way to tune on test outcomes. |
| Turbine timeline | Shows scores, threshold, alerts, episode starts, and preceding warning windows for each fold. Only eligible test days are scored; gaps are not healthy predictions. Green shading is a warning window, not the episode duration. |
| Feature weights | Top 20 absolute standardized coefficients for each fold and target. |
| Lead-time distribution | Warning times for detected fold-episode observations. Undetected episodes are absent, so always check recall too. |

For metric definitions, see [evaluation metrics](EVALUATION.md#metrics). An episode
can have scored warning days in adjacent folds and therefore appear in both;
do not sum fold episode counts as distinct annual faults. Outputs are per-fold
evaluations, not pooled annual performance. These experiments use the same
turbines within one year and do not establish transfer to new years or turbines.

To use custom folders, pair training's `--output-dir` with plotting's `--input-dir`.
The plotter reads the protocol snapshot saved by the training run, not today's
configuration. `--output-dir` on the plotter selects the figure folder.
