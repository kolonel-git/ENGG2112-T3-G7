# Binary prediction of any alarm start

Use the existing `Data_Parquets_10min` dataset. The source already has aligned 10-minute rows, `sensorObserved`, and `alarmOccurred` (new starts of **any** code). Do not rerun Excel conversion/alignment simply to use this experiment.

1. **`01_any_alarm_binary_forecast.ipynb`** — validate the data, show coverage/missingness, plot sensors against alarm starts, construct features and targets, then train logistic regression, random forest and XGBoost. Final testing is opt-in.
2. **`02_review_saved_results.ipynb`** — inspect the latest completed run, figures, metrics and feature importance without retraining.
3. **`03_predict_with_saved_models.ipynb`** — load the validation-selected models and produce three yes/no decisions from a two-hour sensor history.

Install `../../requirements.txt` in your selected Python kernel. The notebooks discover the repository from their location; there are no user-specific drive paths. Experiment settings are in the outer repository's `configs/short_horizon_alarm.json`. Reusable logic lives in `src/wt_predmaint/short_horizon.py`. The default config supports this local flattened folder and the original `iteration2/Iteration2` layout.

## Exact forecasting question

For each physical turbine and each 10-minute decision time `t`:

- Inputs: the 12 completed sensor rows in `[t-120 minutes, t)`. At 12:00 use rows stamped 10:00 through 11:50. This deliberately excludes the row at the decision time, avoiding current-interval availability ambiguity.
- Features: mean, standard deviation, minimum, maximum, last reading and last-minus-first change for every training-schema `wtc_` sensor. No station ID, alarm fields, future readings or future-imputed values are predictors.
- `y10`, `y20`, `y30`: 1 if any recorded alarm starts in `[t,t+10)`, `[t,t+20)` or `[t,t+30)` minutes; otherwise 0. Starts at the right endpoint are excluded. These are **cumulative horizons**, not separate 10–20 and 20–30-minute bands.
- A new start still counts while other alarms are active. This is not an alarm-free-only or icing-only model. Selected preprocessing codes do not restrict the target.
- All 12 source intervals must be observed; individually missing sensors are retained and imputed from training medians. Windows crossing a file-year boundary, data gap, or split boundary are excluded. Unknown year-end labels are never converted into negatives.
- One binary classifier per horizon per model family. Separate fitted scores need not be monotonic across horizons even though the true cumulative labels are nested.

Source timestamps are timezone-naive and interpreted consistently with the supplied dataset's UTC convention. Source `alarmOccurred` labels are trusted as exported: this workflow cannot recover omitted raw alarm records or prove that an empty alarm interval had complete source logging.

## Development and evaluation

Default: all 21 physical turbines (2304510–2304530), training 2016–2022, validation 2023, final test 2024–2025. 2019 is absent. Auxiliary station 91 is excluded. This evaluates future periods for known turbines; it does not establish performance on unseen turbines.

Both the whole 120-minute feature history and the whole 30-minute maximum forecast must remain within their assigned split. Thus train, validation and test do not share underlying history/target intervals at boundaries. Median imputation, scaling and class weights are fitted on training rows only. Logistic regression and random forest use class weights; XGBoost's positive weight uses the training negative/positive ratio. No SMOTE, label-based evaluation sampling or random temporal split is used.

For a bounded first run, uniform priority sampling retains at most 120,000 training rows and 50,000 validation/test rows each. Every eligible row in a split has equal inclusion probability, independent of labels. This is an **all-turbine sampled-window experiment**, not fitting/scoring all millions of windows. `sampling.json` records full eligible populations, positive counts and sample sizes. `data_quality.csv` and prevalence figures use every eligible development window. Set caps to `null` in JSON for all rows if sufficient memory is available.

Validation F1 chooses each model's threshold; validation average precision chooses a model per horizon. All fits, thresholds and choices are saved before any test data is opened. The test evaluator refuses to repeat testing in the same output folder. These years have been inspected in earlier team work, so do not describe them as historically untouched. Reusing test results for tuning would invalidate the intended holdout comparison.

Report accuracy, balanced accuracy, precision, recall, F1, average precision, ROC-AUC, confusion counts and per-turbine test metrics alongside a constant training-prevalence baseline. Accuracy alone can conceal class imbalance. Metrics and predictions are for **overlapping forecast windows**, not independent events; no event recall, statistical confidence interval or live deployment claim is implied. Class-weighted model scores are not calibrated probabilities.

Outputs live in a unique timestamp/configuration-hash directory under `reports/any_alarm_binary/`. It contains configuration, data audit, sampling disclosure, figures, frozen choices, model bundles, feature importance and sampled predictions. Never load untrusted joblib files. Data and the older alarm-group notebooks are not altered.

## Documentation references

- [scikit-learn: time-series evaluation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html)
- [scikit-learn: precision–recall and average precision](https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html)
- [XGBoost parameters: histogram trees and class weights](https://xgboost.readthedocs.io/en/stable/parameter.html)
