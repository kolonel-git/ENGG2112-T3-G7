# pitch_lubrication_3130

Each horizon folder contains the same three notebook roles:

- `01_train_all_features.ipynb`: all sensor features.
- `02_train_selected_features.ipynb`: editable sensor subset.
- `03_train_selected_features_variant.ipynb`: preserved alternate copy; identical settings give the same experiment/report directory.

Horizon folders: `01_horizon_10min`, `04_horizon_1hour`, `05_horizon_6hours`, `06_horizon_1day`, `07_horizon_1week`.

All use the previous hour of sensor inputs and predict every 10 minutes. Each notebook can run independently after preprocessing. Edit settings and Run All to train. Existing notebooks were preserved; newly created notebooks have not been trained.


## Prediction horizon order

- `01_horizon_10min/`
- `02_horizon_20min/`
- `03_horizon_30min/`
- `04_horizon_1hour/`
- `05_horizon_6hours/`
- `06_horizon_1day/`
- `07_horizon_1week/`

Each horizon contains all-features, selected-features, and selected-features variant notebooks. The sensor lookback remains two hours. New horizons require independent training.
