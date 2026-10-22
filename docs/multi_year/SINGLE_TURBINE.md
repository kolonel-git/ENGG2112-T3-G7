# One-command multi-year experiment for one turbine

For the step-by-step interactive version, start with
[the fault-prediction notebook](../../notebooks/01_single_turbine_fault_prediction.ipynb).
It separates processing, training, validation, and evaluation, with tables and
plots visible inline. **Target A only is now the default**, for both the notebook
and this multi-year CLI workflow.

## Run everything

Install dependencies as described in [general setup](../general/USAGE.md), then
run this file from the repository root or with your IDE's Run button:

```powershell
python scripts/multi_year/single_turbine/main.py
```

Default turbine: **2304510**. The script runs raw-data preparation, feature/label
construction, classifier training, held-out evaluation, and plotting in order.
All stages use the same Python interpreter. A failed stage stops subsequent
stages and returns a nonzero exit status.

## Which years are used?

| Purpose | Default years |
|---|---|
| Fit classifiers, select features, fit imputation/scaling | 2018–2021 |
| Choose alert thresholds | 2022 |
| Evaluate unseen days | 2023–2025 |

All eight years contribute to the experiment, but the classifiers are not fitted
on test years. Training on all years and scoring those same days would not
measure future prediction. These are fixed chronological holdouts, not
within-year CV. Test labels are never used for fitting or threshold selection.
The full 60-day label horizon is purged at train/calibration/test boundaries
using the existing label builder; the 30-day forecast target is unchanged.

This experiment overrides splits in a private copy of the protocol. It does
not edit `configs/protocol.yaml` or the original multi-year experiment. The
currently available 2019 raw files are included here; earlier documentation of
absent 2019 tables described the source inventory at that earlier time.

Change the station or splits with flags, or edit the defaults at the top of main:

```powershell
python scripts/multi_year/single_turbine/main.py --station 2304511
python scripts/multi_year/single_turbine/main.py --train-end 2022 --calibration-year 2023
```

The calibration year must immediately follow the last training year and leave
at least one later test year. Training/calibration each require both classes.
Missing required year/table CSVs cause an explicit error rather than silently
shortening the experiment. Sources must be extracted CSVs, including nested
folders under `data/raw/<year>/`. This workflow does not download files.

## Data preparation and caching

Only the selected turbine is retained while CSVs are read in chunks. Existing
cleaning, sentinel masks, daily aggregates, rolling features, configured alarm
codes, and target A's daily-fault-time episode definition are reused. Monthly/year boundary
duplicates follow the shared cleaning rules. The raw CSVs remain unchanged.

Coverage flags use one expected turbine for this isolated experiment, rather
than expecting all 21 farm turbines. Annual suspect-wind and quality flags are
not model features. Seven-day windows are trailing; fault-history features use
information available through the prediction day. Labelled days with no feature
row are explicitly counted in `coverage.csv` and excluded from scoring, rather
than silently imputing an entirely absent day.

Per-year caches store station-only daily aggregates, summaries, alarm events,
and stop intervals. They are reused when source paths, sizes, modification
times, preparation version, and relevant configuration match. They are not
content-hash verification. Features and labels are rebuilt across the years so
rolling histories and stop episodes can cross year boundaries correctly.

- Normal rerun: verifies source cache metadata, rebuilds features/labels, refits,
  evaluates, and plots.
- `--rebuild`: forces re-reading raw CSVs instead of reusing annual caches.
- `--skip-prepare`: explicitly trusts existing preparation; validates station and
  saved configuration but does not check raw source changes. Useful for plotting
  and model development after a successful preparation run.
- `--work-dir path`: places `prepared/`, `results/`, and `figures/` together in a
  custom experiment folder. Otherwise, use the default folders below.

## Models compared

| Model/reference | What it does |
|---|---|
| Logistic regression | Balanced classes, L2 penalty, C=1; training-only median imputation and standard scaling. |
| Random forest | 300 trees, maximum depth 6, minimum leaf size 10, balanced subsample weights, fixed seed. Same training-selected inputs and preprocessing. No hyperparameter search. |
| High sensor rule | Maximum temperature/pressure deviation above training medians, divided by training interquartile ranges. This is a simple relative high-reading reference, not a manufacturer safety limit. |
| Constant | Gives every day score zero. |
| Random uniform | Gives independent reproducible random scores. |

Target A predicts a new recorded fault-time episode within 30 days. Every scorer
uses exactly the same calibration and test days for that target. Thresholds
meet the false-alarm budget on calibration and are frozen for test; the actual
test rate can exceed that budget. Logistic/forest outputs are risk rankings,
not validated probabilities. The sensor-rule score is unbounded and is not a
probability at all.

Inputs are producing-state sensor means, daily maxima, operating conditions,
seven-day trends, configured alarm counts, and past fault history, as described
in [the feature guide](../single_year/CLASSIFIER.md#features-actually-supplied-to-the-classifier).
Selection drops constant inputs and inputs more than 80% missing on training
rows only. The exact selection, coefficients, forest impurity importances, and
sensor-rule membership are recorded in `features.csv`. Importances are not
causal explanations and can be affected by correlated inputs.

## Outputs

For the defaults:

```text
data/interim/multi_year/2018_2025/turbine_2304510/
    cache/<year>/                  # metadata-verified annual station data
    features.parquet
    a_labels.parquet
    a_episodes.parquet
    label_summary.csv
    preparation.json

data/processed/multi_year/2018_2025/turbine_2304510/
    metrics.csv                    # overall and per-test-year evaluation
    predictions.csv                # held-out daily scores, labels, thresholds, alerts
    episode_detections.csv
    features.csv
    coverage.csv
    *.joblib                       # classifiers/rule with preprocessing and thresholds
    run.json

reports/figures/multi_year/2018_2025/turbine_2304510/
    PNG figures
```

Reruns replace same-named experiment outputs and caches; unrelated files and
single-year experiments are not removed. Changing splits with the same station
and overall year range reuses that output location. Use a different work folder
to keep separate experiments. A failed run may leave earlier outputs, so check
that the main command completed successfully before reading results.

Previous target-B files from older experiments may still exist; the default run
does not read them or produce target-B results. The notebook's dedicated output
subfolder contains target-A results only. Legacy target-B code remains available
as an explicit CLI option for future work; it is outside this notebook workflow.

## Reading the metrics

Start with `period=test_all` in `metrics.csv`, which evaluates all held-out days
together. Other rows give the same frozen model's performance on each individual
test year. Do not average their average precision or median lead times to
obtain pooled metrics. An episode near a year boundary can be eligible in two
annual slices, so do not sum annual episode counts as unique faults.

| Metric | Interpretation |
|---|---|
| `pr_auc` | Average precision: ranking performance; compare against prevalence and random scores. |
| `recall` | Fraction of evaluable episodes detected within their 30-day warning window. |
| `false_alarms_per_turbine_month` | Alerts on negative days divided by scored turbine-months. Compare with the budget of 1. |
| `median_lead_days` | Median earliest advance warning among detected episodes; missed episodes do not contribute. |
| `daily_precision`, `daily_recall`, `daily_f1` | Row-level classification measures at the frozen threshold; these differ from episode recall. |
| `true_positive_days`, `false_positive_days`, `false_negative_days`, `true_negative_days` | Daily confusion-matrix counts. |
| `prevalence`, `n_rows`, `n_positive_rows`, `n_episodes` | Class balance, scored exposure, and episode support needed to interpret results. |
| `test_has_both_classes` | False means that year's classification ranking/false-alarm evidence is limited. |

Undefined values are blank rather than zero. Between-turbine bootstrap intervals
are omitted because this experiment contains only one turbine. See the shared
[evaluation reference](../single_year/EVALUATION.md#metrics) for formulas.

PNG plots include model comparisons, precision–recall and recall/false-alarm
curves, confusion matrices, feature weights, yearly risk timelines, and lead-time
distributions. Test threshold sweeps are descriptive, not instructions to tune
using test outcomes. Gaps in timelines are unscored days, not healthy predictions.

## Individual stages (optional)

`prepare_data.py`, `train_classifiers.py`, and `plot_results.py` can also run
separately. Use their `--help` options for required folder arguments. The main
file supplies those arguments automatically and is the recommended entry point.
