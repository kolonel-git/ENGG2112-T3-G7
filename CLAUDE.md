# CLAUDE.md

## Project
ENGG2112 Group 7. Predicting wind turbine degradation from Hill of Towie SCADA (21 Siemens
SWT-2.3-VS-82 turbines, 10-min data, CC-BY-4.0). Tables: tblSCTurTemp, tblSCTurbine, tblSCTurPress, tblAlarmLog, tblDailySummary (raw files in `data/raw`, 2016 to 2026-04).
- O1: degradation indicator (normal-behaviour model, predicted-minus-observed gap)
- O2: fault/downtime prediction. Targets: fault-time episodes from `tblDailySummary.TimeErr` (A) and SCADA-derived stops (B). Alarm codes (12 documented ones, list in `configs/protocol.yaml`) are features, not targets. Windows: positive 30 days, 30-60-day exclusion, 7-day feature window (D-003, D-005)
- O3: robustness (leave-one-turbine-out, missing sensors, upgrades)
- Models: ridge/logistic baselines, power-curve threshold baseline, class-weighted LightGBM, optional autoencoder, SHAP
- Metrics: PR-AUC, recall at 1 false alarm per turbine-month, median lead time

## Folder map
- `src/wt_predmaint/{data,features,labels,models,evaluation,explain}`: all reusable logic
- `notebooks/`: exploration only, `NN_owner_topic.ipynb`
- `configs/protocol.yaml`: single source of experiment settings
- `data/{raw,interim,processed}`: git-ignored; never commit
- `reports/{figures,tables}`, `tests/`, `scripts/`
- `docs/`: `roadmap.md` (schedule, owners, gates), `DECISIONS.md` (decision log), `DATA_TRAPS.md`, `modelling_plan.md`, `reference/` (teammate-supplied alarm and data descriptions)

## Coding rules
- Python only, with type hints.
- Alarm labels come from the raw `tblAlarmLog` via `src/wt_predmaint/data/alarms.py`, never from the `Alarmcode`/`AlarmEvents` columns of the processed workbooks (they keep ~0.2% of alarms).
- No data leakage across train/val/test boundaries (fit scalers, imputers, thresholds on train only; split by time/year per protocol).
- All randomness seeded from `seed` in `configs/protocol.yaml`.
- Every experiment reads `configs/protocol.yaml`; no hard-coded windows, years or budgets.
- Reusable logic in `src/`, not notebooks. Add tests for new logic. Run `ruff check .` and `pytest`.
- Undecided values stay `TODO`; do not pick them silently. Record every decision in `docs/DECISIONS.md` with an ID the same day, and change `configs/protocol.yaml` only together with its entry.
- Model on raw tables via `src/wt_predmaint/data/`, never on the processed workbooks. SCADA `TimeStamp` is the interval end (UTC); join alarms with ceil to 10 min. `tblDailySummary` rows are stamped at the start of the day. Use the cleaned tables in `data/interim/clean/` (`scripts/build_clean_data.py`); flags `n_masked`, `outage_month`, `suspect_wind` are kept, never dropped, in the data.
- O2 labels come from `scripts/build_labels.py` (D-019): rows that are in an episode, in the 30-60 day zone, purged at a split boundary, censored or poorly observed carry no label and must not be used for training or scoring.
- Fit anything (scalers, imputers, normal-behaviour models, thresholds) on train years only; the test years are scored once per frozen configuration.
- **Never invent results or references.** Report only what was actually computed; cite only sources actually verified.
