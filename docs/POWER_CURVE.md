# B3: healthy wind/RPM power-curve proxy

B3 is a trained baseline, unlike H0 (`healthy.py`). It learns expected **generator
RPM** from primary wind speed on healthy historical training intervals. It is a
proxy for underperformance, not a measured power curve or a calibrated fault
probability.

## Run

From the repository root, using the installed project environment:

```powershell
python scripts/run_b3_power_curve.py
```

This fits separate target-specific curves and evaluates A and B on validation,
with the constant/random floors and turbine-bootstrap confidence intervals.
It does not evaluate test years. Use `--target a` for A only.

The ordinary runner and integrated notebook also find B3 in the registry:

```powershell
python scripts/run_baselines.py --target a --baselines B3_power_curve
python scripts/check_feature_leakage.py --b3 --target a --stations 1 --days 2
python -m pytest tests/test_power_curve.py -v
```

In `notebooks/02_healthy_row_diagnostics_and_baseline_evaluation.ipynb`, set
`MODEL_IDS = ["B3_power_curve"]` and select `PLOT_MODEL = "B3_power_curve"` in the
last code cell. Restart the kernel if it was opened before registration.

## Definition (D-031)

- Pooled fleet curve; 0.5 m/s wind bins; median healthy generator RPM in each bin.
- H0 chooses target-specific healthy training intervals; no validation/test data
  sets the curve or tolerance. Stop runs are excluded from fitting by H0.
- Working wind range: 4â€“25 m/s inclusive, an explicit provisional operating range,
  not a manufacturer-verified limit. Bins require at least 30 training intervals.
- Linear interpolation between supported bin centres, including internal gaps;
  no extrapolation outside the outer centres. Unsupported intervals are missing.
- Tolerance: two population standard deviations of signed healthy training
  residuals (expected RPM minus observed RPM). Values are not tuned on validation.
- Outage, suspect-wind, missing/nonfinite wind/RPM intervals are excluded from fit
  and daily scoring. Stop intervals remain eligible at scoring time: their RPM
  shortfall is part of the desired signal. No future stop-run labels enter scores.
- Day follows interval start (`TimeStamp - 10min`). Daily statistics include mean
  signed shortfall and share of valid working-range intervals above tolerance.
- Score is the trailing calendar-window mean of that share, using
  `labels.feature_window_days` (7) and `features.window_min_days` (4). Calendar
  gaps remain missing, not compressed into observed-day windows. Counts are
  combined before rolling, including year boundaries.
- Insufficient evidence gives score 0 to satisfy the finite baseline contract,
  with `b3_available=False`. A score of zero must therefore not be interpreted
  as proof of health. Availability is exported separately. The previous days
  can still support a valid rolling score when the current day has no intervals.
- `score(rows)` looks up causal features by turbine/day; it never uses labels,
  status or other rows in the scoring batch. Fit caches train/validation SCADA scores only after curve/tolerance are fixed.
  Test-year SCADA is loaded lazily only when explicitly scored; the runner
  requires frozen test evaluation. Neither operation updates fitted statistics.
- An excluded turbine in a runner sensitivity variant contributes no healthy
  fitting examples because H0 is restricted to the runner's training keys.

## Outputs

Dedicated B3 command writes:

- `reports/tables/b3_validation_{a,b}.csv`: metrics, confidence bounds and floors.
- `reports/tables/b3_curve_{a,b}.csv`: fitted bin medians and interval counts.
- `reports/tables/b3_fit_{a,b}.csv`: healthy interval counts and RPM tolerance.
- `reports/tables/b3_availability_{a,b}.csv`: evidence availability per turbine/year.
- `reports/figures/b3_curve_{a,b}.png`: fitted curve plot.
- `data/interim/features/b3_daily_{a,b}.parquet`: daily diagnostics and scores.
- `data/interim/scores/{a,b}_all_B3_power_curve_val.parquet`: plotting inputs.

Generated outputs replace files with the same names. Source tables are preserved.
The existing raw-data preparation scripts must have been run first.

## Initial validation, 2026-10-22

Using configured training years 2016/2017/2018/2020 and validation 2021/2022,
all available turbines; test years were not evaluated:

| Target | Healthy fit intervals | Tolerance RPM | AP | Episode recall | Lead days | False-alert days/turbine-month | Random episode recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| A | 1,186,879 | 181.98 | 0.4546 | 0.2153 | 21 | 0.9989 | 0.7799 |
| B | 938,075 | 170.70 | 0.4999 | 0.3021 | 21 | 0.9965 | 0.7604 |

These results do **not** demonstrate useful early detection: recall is below the
random reference at the same budget. Full intervals are in the CSVs. Do not tune
against test years to improve these numbers.

Validation: 107 repository tests pass; changed Python files pass Ruff. The
real-data B3 truncation check passed two sampled turbine-days (station 2304511).
Synthetic tests cover curve recovery, injected underperformance, held-out
perturbation, score contract, midnight assignment, calendar gaps and missing data.
Repository-wide Ruff still reports existing findings outside this change,
including the previously created notebook.
