# icing_8230_8234_8235_8236

Each horizon folder contains the same three notebook roles:

- `01_train_all_features.ipynb`: all sensor features.
- `02_train_selected_features.ipynb`: editable sensor subset.
- `03_train_selected_features_variant.ipynb`: preserved alternate copy; identical settings give the same experiment/report directory.

Horizon folders: `01_horizon_10min`, `02_horizon_1hour`, `03_horizon_6hours`, `04_horizon_1day`, `05_horizon_1week`.

All use the previous hour of sensor inputs and predict every 10 minutes. Each notebook can run independently after preprocessing. Edit settings and Run All to train. Existing notebooks were preserved; newly created notebooks have not been trained.
