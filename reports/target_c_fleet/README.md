# Target C (fleet) results (D-071)

Alarm-group forecasts 1, 6 and 24 hours ahead on all 21 turbines, project split (train 2016-2018 and 2020,
validation 2021-2022, test 2023 to April 2026, test scored once).

- `validation.csv`, `test.csv`: one row per alarm group x horizon x model (PR-AUC with turbine-bootstrap
  interval; episode recall, lead time and realised false-alert rate at budgets of 1 and 30 false alert-hours
  per turbine-month).
- `frozen_thresholds.json`: thresholds chosen on validation and applied unchanged to test.
- `scores/`, `models/`, `rows/`: intermediate files (git-ignored).

Reproduce from the raw CSVs in `data/raw/`:

    python scripts/target_c_fleet_build.py --step all          # alarms, cleaned 10-min parquet, hourly features
    python scripts/target_c_fleet_run.py --phase fit           # train, validation tables, frozen thresholds
    python scripts/target_c_fleet_run.py --phase test          # single test scoring

Both scripts are resumable: re-run until they print "all done".
