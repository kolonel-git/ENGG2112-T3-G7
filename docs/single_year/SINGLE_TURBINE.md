# One turbine within one year

This experiment fits logistic regression using only the selected turbine's data.
Training, missing-value handling, scaling, threshold calibration, test scores,
and episode evaluation all use that turbine. This is different from merely
plotting one turbine's predictions from a model trained on the whole farm.

## Run

After the shared data preparation through `build_features.py`, run from the
repository root in the environment with project dependencies:

```powershell
python scripts/single_year/single_turbine/train_classifier.py
python scripts/single_year/single_turbine/plot_classifier.py
```

Both default to **2018** and **turbine 2304510**. To choose another:

```powershell
python scripts/single_year/single_turbine/train_classifier.py --year 2018 --station 2304521
python scripts/single_year/single_turbine/plot_classifier.py --year 2018 --station 2304521
```

Alternatively edit `STATION` in both entry-point files. Pass `--year` to change
the year. Plotting reads the saved run; it does not train a model.

## Folds and limitations

The same three chronological test months (August, September, October) and full
60-day horizon purges described in [classifier training](CLASSIFIER.md) apply.
These are forward-in-time folds, not shuffled days. Each needs nonempty train,
calibration, and test periods, plus both classes in training and calibration.
Unusable folds are skipped and their reasons recorded in `run.json`. The script
does not shorten the label horizon or borrow another turbine's rows to fill gaps.

In the current 2304510/2018 data, **only the August fold is viable for each
target**. September and October calibration contain positives only, so they
are skipped. This run therefore produces one chronological holdout result per
target, not a complete three-fold CV estimate.

The viable fold has only 31 training days and 27 eligible test days for each
target. All 27 test labels are positive. Average precision of 1 and zero false
alarms in that window do **not** demonstrate a good classifier: there are no
negative test days on which to assess false alerts. The console reports
single-class test windows and `test_has_both_classes` records their support.
The selected features also outnumber training days substantially, making this
a small-data experiment rather than evidence of reliable performance.

Turbine-bootstrap intervals are omitted (blank `_lo`/`_hi` cells): resampling
one turbine always selects the same data and cannot estimate between-turbine
uncertainty. Fold plots show point estimates without those bounds. Undefined
recall/lead time or absent positive labels remain undefined as in the shared
metric definitions; scores are not calibrated probabilities.

## Organised outputs

For the defaults:

- Models, metrics, exact feature audit, daily predictions, episode detections,
  and run metadata: `data/processed/single_year/2018/turbine_2304510/`.
- PNG plots: `reports/figures/single_year/2018/turbine_2304510/`.
- Pooled-turbine runs: the separate `all_turbines/` folder under the same year.

Rerunning replaces same-named files. Existing outputs from folds skipped on a
later run may remain; consult the manifest for currently completed folds or use
a new `--output-dir`. A plot run with `--input-dir` validates that the saved
model run was trained on the requested turbine, rather than presenting pooled
training as a single-turbine experiment.

Changing the selected turbine can change the number of viable folds. More data
within the year or an explicitly revised evaluation design may be needed for
useful CV. The shared [metric reference](EVALUATION.md) and
[feature summary](CLASSIFIER.md#features-actually-supplied-to-the-classifier)
still apply, but exact retained features are specific to this smaller training set.
