# Roadmap (6 weeks)

Owners: Parvez = data acquisition and cleaning; Revael = features and labelling;
Kanav = modelling and tuning; Farhan = evaluation harness and cost analysis;
Cyrus = ethics review and report integration.

- **Wk0 Setup** (all): clone, install, download data, fill TODOs in `configs/protocol.yaml`. Lead: Kanav (repo), Parvez (data download).
- **Wk1 Explore** (Parvez, all): load the four tables, log findings in `DATA_TRAPS.md`, cleaning functions in `src/data`.
- **Wk2 Baselines and evaluation harness** (Kanav, Farhan): ridge/logistic and power-curve baselines; PR-AUC, recall at budget, lead time.
- **Wk3 Labels and O1** (Revael, Kanav): alarm labels and windows, normal-behaviour model, degradation indicator.
- **Wk4 Main models** (Kanav, Revael): class-weighted LightGBM, optional autoencoder, tuning.
- **Wk5 Robustness and SHAP** (Farhan, Kanav): leave-one-turbine-out, missing sensors, upgrades, SHAP, cost analysis.
- **Wk6 Write-up** (Cyrus, all): ethics review, figures/tables, report integration.

TODO: confirm exact dates and per-task owners.
