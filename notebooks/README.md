# Start here: fault-prediction notebook

Open [01_single_turbine_fault_prediction.ipynb](01_single_turbine_fault_prediction.ipynb)
in VS Code or Jupyter. It contains the single-turbine 2018–2025 workflow using
**target A only**, with executed tables and plots already visible.

1. Select the Python environment with the project dependencies as the notebook
   kernel (in VS Code: **Select Kernel → Python Environments → .venv**).
2. Review the experiment settings: turbine 2304510, train 2018–2021,
   validate 2022, test 2023–2025.
3. Choose **Run All**, or run the cells in order from the top. First-time raw
   processing can take several minutes; unchanged annual caches are reused.

The four main sections are data processing, training, validation, and evaluation.
Training and threshold selection appear in separate cells. The notebook compares
logistic regression and random forest with a high-sensor rule, constant scores,
and random scores. It explains the daily-fault-time episode definition before
building labels, then shows coverage, class balance, features, metrics and graphs.

Notebook exports are isolated in:

- `data/processed/multi_year/2018_2025/turbine_2304510/target_a_notebook/`
- `reports/figures/multi_year/2018_2025/turbine_2304510/target_a_notebook/`

Change the year/turbine in the settings cell and rerun from the top. Change
`PLOT_YEAR` and `PLOT_MODEL` to inspect a different held-out year or classifier's
risk timeline. Use fresh held-out data to evaluate changes made after inspecting
test performance. Scores are not calibrated fault probabilities.

The code reuses shared data/model helpers while showing the workflow directly;
it does not require running multiple scripts manually. For terminal use, the
[one-command guide](../docs/multi_year/SINGLE_TURBINE.md) remains available and
also defaults to target A only.
