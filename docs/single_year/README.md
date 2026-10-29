# Single-year experiments

Use this workflow to evaluate constant and random dummy baselines within a
selected calendar year. The default is **2018**. These reference scorers do not
train on sensor features.

## Setup and prerequisites

Follow [general setup](../general/USAGE.md). Run the shared preparation commands
in the [multi-year workflow](../multi_year/WORKFLOW.md) through
`build_labels.py`, with raw data available for your selected year. Those builders
scan available years; the year selector below only controls baseline evaluation.
The single-year runner does not require features or the main multi-year baseline
results, nor does it require separate validation/test years.

## Run from the repository root

```powershell
python scripts/single_year/run_year_baselines.py
```

Choose another year or reduce the bootstrap draws for a quicker experiment:

```powershell
python scripts/single_year/run_year_baselines.py --year 2018 --resamples 100
```

Alternatively, change `YEAR` near the top of the script. The command-line value
overrides that default. Use your IDE's configured Python environment.

## Outputs and interpretation

Results go to `reports/tables/baselines_dummy_cv_<year>.csv`. Rerunning replaces
that selected output. The main `baselines_dummy.csv` is separate.

Three chronological folds calibrate thresholds on earlier days and test on later
days, using all available turbines. The full configured label horizon is purged
to separate periods. Empty folds are skipped explicitly.

Read [the evaluation reference](EVALUATION.md) for every CSV column, fold dates,
metrics, confidence intervals, and limitations. This experiment does not measure
generalisation to unseen years or held-out turbines and does not replace the
main multi-year evaluation.
