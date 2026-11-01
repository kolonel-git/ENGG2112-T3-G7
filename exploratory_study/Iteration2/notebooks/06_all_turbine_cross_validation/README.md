# All-turbine temporal cross-validation

Open **01_cv_8200_family_all_turbines_6hour.ipynb** after preparing the regular 10-minute data. It forecasts the next six hours from the prior sensor hour for the selected icing codes **8230, 8234, 8235 and 8236 only**. All eligible sensor turbines are included automatically; alarm-only IDs are audited and excluded from modelling.

Development: available 2016–2023. Outer evaluations: 2021, 2022, 2023. Each has an earlier calibration year separate from model fitting. Both features and labels are purged at date boundaries. Final refitting uses available 2016–2023 and thresholds from development out-of-fold predictions. The final learned champion and per-family models are frozen before any test access.

**2024 and 2025 are disabled by default:** `RUN_RESERVED_TEST = False`. Run All develops/fits the models without opening these years. Enable the reserved step only once the protocol is final. These years were inspected in earlier project work, so do not claim they have never been seen historically.

All positive windows are kept. The default retains at most 250,000 negatives per training stage, allocated across every eligible turbine-year, with inverse-probability/class-balancing weights for model fitting. Calibration and evaluation use every eligible row. This is all-turbine training, not all-negative-row training. Set `MAX_TRAIN_NEGATIVES = None` for the latter; that needs substantially more RAM and CPU time. Imputation/scaling are fitted to sampled training rows only. The first run builds per-turbine/year feature caches; later runs reuse unchanged caches.

The notebook compares five model families and a small parameter search, reports AP, precision, recall, F1/F2, confusion matrices, per-turbine results, event recall, lead time and false-warning episodes. Optional threshold policy `false_warning_budget` constrains both false-warning episodes and warning fraction; default policy is `max_f1` on inner calibration only.

`cv_pipeline.py` implements the workflow. Tests are in `tests/test_cv_pipeline.py`; run from this folder with `python -m unittest discover -s tests -v`. Reports go under `Iteration2/reports/cross_validation_8200_family_6hour/<run-id>/`; derived features under `Iteration2/Data_Parquets_CV/`. Original data and previous training notebooks are preserved.


For a standalone review, use [evaluate_cross_validation_results.ipynb](../../evaluate_cross_validation_results.ipynb) at the Iteration2 root. It reads saved development metrics/predictions only and does not retrain or evaluate reserved years.

The default target is now restricted to `ALARM_CODES = ['8230', '8234', '8235', '8236']`. Other codes (including 8210) do not affect the target episodes. Changing the code list produces a new run ID and cache signature, so previous broad 8200-family reports are preserved. Restart the kernel and Run All to retrain this new target; do not interpret older broad-family results as restricted-code results.
