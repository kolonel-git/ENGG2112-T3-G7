# Multi-year experiments

The [target-A notebook](../../notebooks/01_single_turbine_fault_prediction.ipynb)
is the interactive walkthrough. Multi-year scripts now default to target A only.

`single_turbine/` contains a complete station-isolated workflow over 2018–2025:

```powershell
python scripts/multi_year/single_turbine/main.py
```

The main runs `prepare_data.py`, `train_classifiers.py`, and `plot_results.py`
sequentially using the same interpreter. Defaults: turbine 2304510, fit on
2018–2021, calibrate on 2022, test on 2023–2025. It does not fit on the test years.

See [the multi-year single-turbine guide](../../docs/multi_year/SINGLE_TURBINE.md)
for options, caching, outputs, models, and metric interpretation.
