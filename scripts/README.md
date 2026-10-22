# Script scopes

Run commands from the repository root with the project's Python environment.
See [the usage guide](../docs/general/USAGE.md) for setup and full run order.

| Location | Scope |
|---|---|
| `multi_year/single_turbine/` | One-command preparation, classifier/reference fitting, held-out evaluation, and plots over 2018–2025 for one turbine. Start with `main.py`. |
| `single_year/single_turbine/` | Training and plots restricted to one turbine, default 2304510 in 2018. |
| `single_year/all_turbines/` | Pooled-turbine within-year classifier, dummy scores, and plots. Default year: 2018. |
| Top-level `build_*.py` | Shared data preparation across available years. These are pipeline steps, not model-training scripts. |
| `run_dummy_baselines.py` | Main constant/random evaluation using configured validation and test years. |
| `check_feature_leakage.py` | Checks selected turbine-days for future-data leakage in feature calculations. |
| `download_data.py` | Downloads original dataset files. |
| `data_preparation/` | Separate Excel export and workbook-cleaning workflow. |

The single-year command is:

```powershell
python scripts/single_year/all_turbines/run_year_baselines.py --year 2018
```

It uses all available turbines within time-separated folds, not held-out turbines.
Cross-turbine split helpers exist in `src/wt_predmaint/evaluation/splits.py`, but
there is currently no dedicated held-out-turbine runner in this scripts folder.
Neither dummy runner trains predictive models. The separate single-year classifier
workflow is documented in [the classifier guide](../docs/single_year/CLASSIFIER.md).

For the CSV columns, metrics, fold design, and confidence intervals, see
[single-year evaluation](../docs/single_year/EVALUATION.md).
