# Single-year scripts

| Folder | What runs |
|---|---|
| `single_turbine/` | Train, calibrate, and test using one turbine only. Defaults: turbine 2304510, year 2018. |
| `all_turbines/` | Pool the available turbines in a single year for classifier or dummy-only evaluation. |

Start with one turbine, from the repository root:

```powershell
python scripts/single_year/single_turbine/train_classifier.py
python scripts/single_year/single_turbine/plot_classifier.py
```

Pass `--station` and `--year` to both commands to change the experiment. The
single-turbine entry points reuse the shared implementation in `all_turbines/`
with the turbine filter applied before fitting and threshold calibration.

Outputs are grouped by year and scope:

```text
data/processed/single_year/2018/
    all_turbines/
    turbine_2304510/
reports/figures/single_year/2018/
    all_turbines/
    turbine_2304510/
```

See [single-turbine instructions](../../docs/single_year/SINGLE_TURBINE.md) for
fold limitations and [the classifier guide](../../docs/single_year/CLASSIFIER.md)
for model details.
