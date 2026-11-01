# 05_model_training

Choose an alarm-set folder and horizon. Current configurations all use the icing alarm group. Each model notebook is independent.

See [the pipeline guide](../README.md) for the full run order.


## Alarm-specific training folders

- `icing_8230_8234_8235_8236/` — existing grouped icing notebooks.
- `pitch_lubrication_3130/` — independent pitch-lubrication experiments.
- `untwisting_cables_10105/` — independent untwisting-cable experiments.

All three alarm-set folders contain matching horizon folders and training notebooks. Each alarm set is trained independently; existing icing settings are unchanged.

Pitch-lubrication (3130) and untwisting-cables (10105) now mirror all five icing horizon folders, including the all-feature and selected-feature variants available at each horizon. The new notebooks have not been trained.

Every alarm group now includes all three notebook roles in each of the five horizon folders: all features, selected features, and the selected-feature variant. New notebooks have not been trained.


## Prediction horizon order

- `01_horizon_10min/`
- `02_horizon_20min/`
- `03_horizon_30min/`
- `04_horizon_1hour/`
- `05_horizon_6hours/`
- `06_horizon_1day/`
- `07_horizon_1week/`

Each horizon contains all-features, selected-features, and selected-features variant notebooks. The sensor lookback remains two hours. New horizons require independent training.
