# Iteration2 machine-learning conclusions

**Review date:** 1 November 2026. **Evidence:** 15 saved `metrics.csv` files and their `configuration.json` files in [reports](reports/). This is a summary of existing results, not a new training run.

## Overall conclusion

Iteration2 provides a working pipeline for preparing turbine data and evaluating alarm forecasts. The saved experiments show some ability to rank alarm-related windows above ordinary operation, but **do not yet establish a reliable operational early-warning system**. Low precision, missed positive windows and very small numbers of alarm events remain important limitations.

The strongest practical precision/recall trade-off among the illustrative runs below is the **6-hour untwisting-cables (10105) LightGBM run**: AP **0.107597**, precision **17.52%**, recall **22.29%**, and F1 **0.1962**. It still produces **2,105 false-positive windows** and misses **1,558 positive windows**. These are window counts, not independent warnings or events. This observation does not establish a universally best model or a deployable warning policy.

## What was evaluated

- All completed saved configurations reviewed here use **turbine 2304511 only**. The findings do not demonstrate generalisation across the wind farm or to unseen turbines.
- Models use the preceding **one hour** of sensor readings, summarised using rolling mean, standard deviation, minimum and maximum. Predictions are issued every **10 minutes**.
- Depending on the run, a positive target means an individual alarm or any member of the icing group **8230 / 8234 / 8235 / 8236** starts within the selected horizon.
- Available pre-2023 data is used for training, 2023 for validation, and 2024–2025 for testing. Split boundaries are purged using the lookback/horizon; there is no random temporal split.
- Missing feature values are imputed using training-set medians. Model inputs exclude alarm fields, timestamps and turbine identifiers. Windows during an already-active target/group are excluded.
- Models are constant prevalence, fixed random-weight score, logistic regression, LightGBM and random forest. Validation F1 chooses each nonconstant model's threshold; the constant baseline uses 0.5.
- The metric labelled `PR-AUC (AP)` is **average precision**, not trapezoidal integration. Compare it with positive prevalence. Precision, recall and confusion-matrix counts use the validation-selected threshold.

## Illustrative test results

These are selected comparisons from the saved reports, not a new model-selection exercise. The table does not claim that one model wins every metric. AP and F1 are shown on a 0–1 scale; prevalence, precision and recall are percentages. TP/FP/FN count prediction windows.

| Target / horizon / inputs | Model | Test AP | Positive prevalence | Precision | Recall | F1 | TP / FP / FN |
|---|---|---:|---:|---:|---:|---:|---|
| [3130 / 10 min / all](reports/alarm_pitch_lubrication_3130_10min_aligned/metrics.csv) | Logistic regression | 0.002451 | 0.14% | 0.31% | 2.90% | 0.0057 | 4 / 1273 / 134 |
| [8000 / 10 min / all](reports/alarm_8000_10min_aligned/metrics.csv) | Logistic regression | 0.205140 | 0.00% | 3.85% | 20.00% | 0.0645 | 1 / 25 / 4 |
| [Icing / 10 min / all](reports/alarm_icing_8230-8234-8235-8236_10min_aligned/metrics.csv) | LightGBM | 0.009072 | 0.00% | 0.00% | 0.00% | 0.0000 | 0 / 35 / 4 |
| [Icing / 6 h / all](reports/alarm_icing_8230-8234-8235-8236_360min_aligned/metrics.csv) | LightGBM | 0.022732 | 0.14% | 7.42% | 11.81% | 0.0912 | 17 / 212 / 127 |
| [Icing / 1 day / all](reports/alarm_icing_8230-8234-8235-8236_1440min_aligned/metrics.csv) | Random forest | 0.025793 | 0.56% | 3.88% | 3.47% | 0.0366 | 20 / 496 / 556 |
| [Icing / 1 week / all](reports/alarm_icing_8230-8234-8235-8236_10080min_aligned/metrics.csv) | Random forest | 0.061083 | 3.22% | 7.06% | 9.67% | 0.0816 | 315 / 4145 / 2941 |
| [10105 / 1 h / gear pressure only](reports/alarm_untwisting_cables_10105_60min_selected_fad20fe311/metrics.csv) | LightGBM | 0.018215 | 0.33% | 3.20% | 24.40% | 0.0566 | 81 / 2447 / 251 |
| [10105 / 6 h / all](reports/alarm_untwisting_cables_10105_360min_aligned/metrics.csv) | LightGBM | 0.107597 | 1.96% | 17.52% | 22.29% | 0.1962 | 447 / 2105 / 1558 |

## Findings by alarm target

### Pitch lubrication: 3130

The aligned all-feature 10-minute run has **138 positive windows out of 102,144 test windows** (about **0.1351%** prevalence). Logistic regression achieves AP **0.002451**, modestly above prevalence, but catches only **4** positive windows with **1,273** false positives. LightGBM also catches 4, with 1,326 false positives. Random forest catches only 1, with 279 false positives.

These results show weak short-horizon warning performance. A score above a random-ranking baseline is not enough when precision is around 0.3% and most positive windows are missed.

Feature reduction was not a consistent improvement. In the [four hydraulic/hub-pressure sensor run](reports/alarm_3130_10min_selected_949e6ef3bf/metrics.csv), random forest reaches **97.83% recall** but produces **91,281 false-positive windows**, with precision only **0.1477%**. High recall alone therefore does not indicate a useful predictor. The gear-pressure-only run also does not establish a better overall warning policy than the full-feature run.

### High wind speed: 8000

The saved 10-minute run has only **2 validation positive windows** and **5 test positive windows**. Logistic regression's test AP is **0.205140**, but at the selected threshold it catches **1 of 5** positive windows and produces **25** false positives (precision **3.85%**, recall **20%**). LightGBM catches none and produces 151 false positives; random forest issues no positive warnings at its selected threshold.

The high AP is based on exceptionally few positives and should not be treated as stable evidence of dependable forecasting. Threshold selection using two validation positives is particularly fragile. A high ROC-AUC here does not remove the warning-precision problem.

### Icing group: 8230, 8234, 8235, 8236

At the 10-minute horizon, the full-feature run has only **4 positive test windows**. LightGBM has AP **0.009072** and ROC-AUC approximately **0.9969**, yet catches **zero** positive test windows at the validation-selected threshold. This demonstrates why score ranking and actual warning decisions must be examined separately.

Longer horizons produce more positive windows, but not necessarily more independent alarm events:

- **6 hours:** LightGBM AP **0.022732**, precision **7.42%**, recall **11.81%**; 17 true-positive and 212 false-positive windows.
- **1 day:** random forest AP **0.025793**, precision **3.88%**, recall **3.47%**. Logistic regression has the better F1 of these two in this run (**0.0561** versus **0.0366**), illustrating that the best AP need not give the best thresholded decisions.
- **1 week:** random forest AP **0.061083**, precision **7.06%**, recall **9.67%**. Positive prevalence is now **3.22%**, versus about **0.00392%** at 10 minutes. The larger raw AP cannot by itself demonstrate improved forecasting: the prediction task and prevalence have changed.

Using gear pressure alone weakens the saved icing ranking results compared with the same-horizon all-feature experiments. For LightGBM, 6-hour AP falls from **0.022732** to **0.003964**; 10-minute AP falls from **0.009072** to approximately **0.000096**. This supports retaining a broader sensor set for further investigation, not assuming that a single pressure measurement captures the relevant precursors.

The grouped model predicts whether **any** member starts. These metrics do not establish performance for each individual icing code, nor identify which member will trigger.

### Untwisting cables: 10105

The saved all-feature 6-hour results show a more favourable ranking and warning trade-off than the other illustrative operational points:

- **LightGBM:** AP **0.107597**, precision **17.52%**, recall **22.29%**, F1 **0.1962**.
- **Random forest:** AP **0.097081**, precision **12.99%**, recall **32.62%**, F1 **0.1858**.
- **Logistic regression:** AP **0.052840**, precision **7.14%**, recall **31.17%**.

LightGBM favours precision, whereas random forest catches more positive windows at a higher false-positive burden. Both remain limited: neither catches a majority of positive windows.

The saved gear-pressure-only 1-hour LightGBM run has AP **0.018215**, precision **3.20%**, and recall **24.40%**. It differs in **both horizon and sensor set**, so the difference from the 6-hour full-feature run cannot be attributed to either factor alone.

## Baselines and interpretation

The constant baseline's AP equals prevalence and its ROC-AUC is 0.5; with the fixed threshold it warns on none of these rare-positive datasets. It supplies a ranking floor, not a useful alarm system.

The random-weight score can obtain high recall by warning on almost everything. For example, in the 6-hour 10105 run it catches 1,878 positive windows but produces **93,971** false positives. This is why recall must be read alongside precision and false-positive counts.

None of the score plots should be read as calibrated real-world probabilities. Class weighting changes learned score distributions, and the random projection is not a calibrated probability model. Training-period plots are in-sample and cannot establish future performance.

## Limits on these conclusions

1. **Overlapping windows are correlated.** One event can generate up to 36 positive 6-hour windows, 144 one-day windows, or 1,008 one-week windows. Larger positive-window counts are not equivalent to larger independent event samples.
2. **Raw AP is not directly comparable across horizons or targets.** Compare against each run's prevalence and consider useful lead time and warning burden.
3. **The test period has been inspected across multiple experiments.** Treat these comparisons as exploratory. A fresh untouched final holdout or a predeclared temporal evaluation is needed for final claims; no confidence intervals or significance tests are supplied by these CSVs.
4. **All reviewed runs concern one turbine.** More training notebooks and prepared turbine files do not imply that those turbines have been trained or evaluated.
5. **Event/data quality matters.** Exact duplicate alarm records remain retained by preprocessing, missing/zero durations have limited activity handling, and some sensors are missing. These can affect labels, counts and eligibility. Sensor imputation does not repair missing alarm records.
6. **Some report folders are historical or duplicated.** `alarm_3130_10min` predates the aligned dataset and uses a slightly different eligible sample. `alarm_3130_10min_aligned` and `alarm_pitch_lubrication_3130_10min_aligned` have matching reported classification results; they are not independent replications. Selected-feature hashes identify recorded sensor selections, not independent repeats.
7. **A notebook or empty output folder is not a completed experiment.** Only folders with saved metrics are summarised. In particular, no completed individual 8235 result is inferred from its empty report folder, and the available files do not establish results for every alarm/horizon/feature combination now represented by notebooks.

## Priorities for the next iteration

- Audit target-event availability, exact duplicates and durations; preserve a documented, consistent label policy.
- Expand training and evaluation to more turbines, with explicit future-time and unseen-turbine tests where appropriate.
- Evaluate **event-level detection**, useful lead time and merged warning episodes, alongside the current per-window metrics.
- Select thresholds on validation data using an explicit operational false-warning budget or miss/false-warning trade-off, rather than F1 alone.
- Compare sensor subsets at the **same horizon and split**, and compare horizons against their own baselines. Freeze the experiment plan before accessing a final holdout.

**Conclusion:** the saved results support continuing model and dataset development, especially investigating the 10105 signal and broader sensor inputs. They do not yet support dependable turbine-wide alarm forecasting or deployment claims.

## Saved-run inventory

Linked files are the numerical evidence used for this summary. Configurations beside each CSV record the target, turbine, horizon and selected features. Empty report directories are excluded. No models or datasets were modified to produce this document.

| Saved run | Horizon | Inputs | Test windows / positive windows |
|---|---|---|---:|
| [alarm_3130_10min](reports/alarm_3130_10min/metrics.csv) | 0 days 00:10:00 | All sensors | 102,129 / 138 |
| [alarm_3130_10min_aligned](reports/alarm_3130_10min_aligned/metrics.csv) | 0 days 00:10:00 | All sensors | 102,144 / 138 |
| [alarm_3130_10min_selected_949e6ef3bf](reports/alarm_3130_10min_selected_949e6ef3bf/metrics.csv) | 0 days 00:10:00 | wtc_HydPress_mean, wtc_HubPresA_mean, wtc_HubPresB_mean, wtc_HubPresC_mean | 102,144 / 138 |
| [alarm_3130_10min_selected_fad20fe311](reports/alarm_3130_10min_selected_fad20fe311/metrics.csv) | 0 days 00:10:00 | wtc_GearPres_mean | 102,144 / 138 |
| [alarm_8000_10min_aligned](reports/alarm_8000_10min_aligned/metrics.csv) | 0 days 00:10:00 | All sensors | 102,138 / 5 |
| [alarm_icing_8230-8234-8235-8236_10080min_aligned](reports/alarm_icing_8230-8234-8235-8236_10080min_aligned/metrics.csv) | 7 days 00:00:00 | All sensors | 101,094 / 3,256 |
| [alarm_icing_8230-8234-8235-8236_10min_aligned](reports/alarm_icing_8230-8234-8235-8236_10min_aligned/metrics.csv) | 0 days 00:10:00 | All sensors | 102,101 / 4 |
| [alarm_icing_8230-8234-8235-8236_10min_selected_fad20fe311](reports/alarm_icing_8230-8234-8235-8236_10min_selected_fad20fe311/metrics.csv) | 0 days 00:10:00 | wtc_GearPres_mean | 102,101 / 4 |
| [alarm_icing_8230-8234-8235-8236_1440min_aligned](reports/alarm_icing_8230-8234-8235-8236_1440min_aligned/metrics.csv) | 1 days 00:00:00 | All sensors | 101,958 / 576 |
| [alarm_icing_8230-8234-8235-8236_360min_aligned](reports/alarm_icing_8230-8234-8235-8236_360min_aligned/metrics.csv) | 0 days 06:00:00 | All sensors | 102,066 / 144 |
| [alarm_icing_8230-8234-8235-8236_360min_selected_fad20fe311](reports/alarm_icing_8230-8234-8235-8236_360min_selected_fad20fe311/metrics.csv) | 0 days 06:00:00 | wtc_GearPres_mean | 102,066 / 144 |
| [alarm_icing_8230-8234-8235-8236_60min_selected_fad20fe311](reports/alarm_icing_8230-8234-8235-8236_60min_selected_fad20fe311/metrics.csv) | 0 days 01:00:00 | wtc_GearPres_mean | 102,096 / 24 |
| [alarm_pitch_lubrication_3130_10min_aligned](reports/alarm_pitch_lubrication_3130_10min_aligned/metrics.csv) | 0 days 00:10:00 | All sensors | 102,144 / 138 |
| [alarm_untwisting_cables_10105_360min_aligned](reports/alarm_untwisting_cables_10105_360min_aligned/metrics.csv) | 0 days 06:00:00 | All sensors | 102,060 / 2,005 |
| [alarm_untwisting_cables_10105_60min_selected_fad20fe311](reports/alarm_untwisting_cables_10105_60min_selected_fad20fe311/metrics.csv) | 0 days 01:00:00 | wtc_GearPres_mean | 102,090 / 332 |
