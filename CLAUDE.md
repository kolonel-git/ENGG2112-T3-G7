# CLAUDE.md

## Project
ENGG2112 Group 7. Predicting wind turbine degradation from Hill of Towie SCADA (21 Siemens
SWT-2.3-VS-82 turbines, 10-min data, CC-BY-4.0). Tables: tblSCTurTemp, tblSCTurbine, tblSCTurPress, tblAlarmLog.
- O1: degradation indicator (normal-behaviour model, predicted-minus-observed gap)
- O2: alarm/downtime prediction (30-day positive window, 30-60-day exclusion, 7-day feature window)
- O3: robustness (leave-one-turbine-out, missing sensors, upgrades)
- Models: ridge/logistic baselines, power-curve threshold baseline, class-weighted LightGBM, optional autoencoder, SHAP
- Metrics: PR-AUC, recall at 1 false alarm per turbine-month, median lead time

## Folder map
- `src/wt_predmaint/{data,features,labels,models,evaluation,explain}`: all reusable logic
- `notebooks/`: exploration only, `NN_owner_topic.ipynb`
- `configs/protocol.yaml`: single source of experiment settings
- `data/{raw,interim,processed}`: git-ignored; never commit
- `reports/{figures,tables}`, `docs/`, `tests/`, `scripts/`

## Coding rules
- Python only, with type hints.
- No data leakage across train/val/test boundaries (fit scalers, imputers, thresholds on train only; split by time/year per protocol).
- All randomness seeded from `seed` in `configs/protocol.yaml`.
- Every experiment reads `configs/protocol.yaml`; no hard-coded windows, years or budgets.
- Reusable logic in `src/`, not notebooks. Add tests for new logic. Run `ruff check .` and `pytest`.
- Undecided values stay `TODO`; do not pick them silently. Record decisions in `docs/DECISIONS.md`.
- **Never invent results or references.** Report only what was actually computed; cite only sources actually verified.
