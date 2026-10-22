"""Training-only sensor references and daily classifier counts shared by notebooks/CLIs."""

import numpy as np

from wt_predmaint.data.turbine_export import FEATURES


def sensor_rule_fit(train, columns):
    """Fit a relative high-temperature/pressure reference on training data only."""
    sensors = {f"op_{c}" for table in ("tblSCTurTemp", "tblSCTurPress") for c in FEATURES[table]}
    columns = [c for c in columns if c in sensors]
    if not columns:
        raise ValueError("No temperature/pressure inputs available for the sensor rule")
    values = train[columns].replace([np.inf, -np.inf], np.nan)
    median = values.median()
    scale = (values.quantile(0.75) - values.quantile(0.25)).replace(0, np.nan)
    columns = scale.dropna().index.tolist()
    if not columns:
        raise ValueError("No varying temperature/pressure inputs for the sensor rule")
    return {"features": columns, "median": median[columns], "iqr": scale[columns]}


def sensor_rule_score(rule, frame):
    values = frame[rule["features"]].replace([np.inf, -np.inf], np.nan)
    return ((values.fillna(rule["median"]) - rule["median"]) / rule["iqr"]).max(axis=1).to_numpy()


def daily_metrics(frame, threshold):
    alert = frame["score"].to_numpy() > threshold
    positive = frame["label"].to_numpy() == 1
    tp, fp = int((alert & positive).sum()), int((alert & ~positive).sum())
    fn, tn = int((~alert & positive).sum()), int((~alert & ~positive).sum())
    return {"true_positive_days": tp, "false_positive_days": fp,
            "false_negative_days": fn, "true_negative_days": tn,
            "daily_precision": tp / (tp + fp) if tp + fp else np.nan,
            "daily_recall": tp / (tp + fn) if tp + fn else np.nan,
            "daily_f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else np.nan}
