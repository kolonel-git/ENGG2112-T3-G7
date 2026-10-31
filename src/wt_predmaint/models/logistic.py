"""B4: class-weighted logistic-regression baseline.

The model uses only the feature families assigned to B4 in ``docs/BASELINE_SPLIT.md``.  Both the
preprocessor and classifier are fitted on labelled training rows.  Missing feature values are
handled by :class:`TrainOnlyPreprocessor`; an unexpected non-finite probability from the fitted
classifier falls back to the neutral score 0.5 rather than leaking a missing score to the runner.

``C`` is deliberately not a constructor argument.  It is selected outside this class on validation,
then frozen per target in ``configs/protocol.yaml`` as required by the baseline contract.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from wt_predmaint.config import require
from wt_predmaint.features.prepare import TrainOnlyPreprocessor
from wt_predmaint.models.base import require_train_only

DAY_STATE_COLUMNS = (
    "frac_producing",
    "frac_idle",
    "frac_stopped",
    "wind_mean",
    "wind_std",
    "rpm_mean",
)
FAULT_HISTORY_COLUMNS = ("days_since_fault_day", "fault_days_30")
FORBIDDEN_METADATA_COLUMNS = {
    "StationId",
    "Day",
    "label",
    "status",
    "split",
    "outage_month",
    "suspect_wind",
    "n_intervals",
    "n_producing",
    "masked_frac",
}


def b4_feature_columns(table: pd.DataFrame, cfg: dict[str, Any]) -> list[str]:
    """Return B4's candidate columns, in their existing table order.

    The rolling prefix follows the configured feature window rather than assuming seven days.
    Alarm names are enumerated from the configured codes and tiers, so an unrelated column merely
    beginning with ``alarm_`` cannot enter the model.
    """
    window = int(cfg["labels"]["feature_window_days"])
    alarm_specs = cfg["labels"]["alarm_codes"]
    alarm_names = {f"alarm_{int(spec['code'])}_w{window}" for spec in alarm_specs}
    alarm_names.update(
        f"alarm_tier{tier}_w{window}" for tier in {int(spec["tier"]) for spec in alarm_specs}
    )
    alarm_names.add(f"alarm_total_w{window}")
    direct_names = {*DAY_STATE_COLUMNS, *FAULT_HISTORY_COLUMNS, *alarm_names}
    rolling_prefixes = (f"w{window}mean_", f"w{window}std_")

    selected = []
    for column in table.columns:
        if column in direct_names:
            selected.append(column)
            continue
        for prefix in rolling_prefixes:
            if column.startswith(prefix):
                source = column.removeprefix(prefix)
                if source not in FORBIDDEN_METADATA_COLUMNS:
                    selected.append(column)
                break
    return selected


class LogisticBaseline:
    """B4 logistic model with train-fitted preparation and balanced class weights."""

    def __init__(self, target: str) -> None:
        if target not in {"a", "b"}:
            raise ValueError("target must be 'a' or 'b'")
        self.name = "B4_logistic"
        self.target = target

    def fit(self, train: pd.DataFrame, cfg: dict[str, Any]) -> LogisticBaseline:
        require_train_only(train)
        work = train.copy()
        y = work["label"].astype(int)
        if set(y.unique()) != {0, 1}:
            raise ValueError("B4 logistic training rows must contain both label classes 0 and 1")

        columns = b4_feature_columns(work, cfg)
        if not columns:
            raise ValueError("no B4 feature columns were found")

        prep_cfg = cfg["preprocessing_p0"]
        raw_clip = prep_cfg["clip_quantiles"]
        clip_quantiles = (
            None
            if raw_clip is None
            else (float(raw_clip[0]), float(raw_clip[1]))
        )
        self.preprocessor_ = TrainOnlyPreprocessor(
            columns,
            min_observed_share=float(prep_cfg["min_observed_share"]),
            add_missing_indicators=bool(prep_cfg["add_missing_indicators"]),
            scale=bool(prep_cfg["scale"]),
            clip_quantiles=clip_quantiles,
        )
        matrix = self.preprocessor_.fit_transform(work)

        model_cfg = cfg["baseline_b4"]
        self.selected_c_ = float(require(cfg, f"baseline_b4.selected_c.{self.target}"))
        self.solver_ = str(model_cfg["solver"])
        self.max_iter_ = int(model_cfg["max_iter"])
        self.tolerance_ = float(model_cfg["tolerance"])
        self.nonfinite_score_fallback_ = float(model_cfg["nonfinite_score_fallback"])
        if not np.isfinite(self.selected_c_) or self.selected_c_ <= 0:
            raise ValueError("baseline_b4.selected_c must be finite and positive")
        if self.max_iter_ <= 0:
            raise ValueError("baseline_b4.max_iter must be positive")
        if not np.isfinite(self.tolerance_) or self.tolerance_ <= 0:
            raise ValueError("baseline_b4.tolerance must be finite and positive")
        if not 0 <= self.nonfinite_score_fallback_ <= 1:
            raise ValueError("baseline_b4.nonfinite_score_fallback must be in [0, 1]")

        self.class_counts_ = y.value_counts().sort_index().to_dict()
        self.model_ = LogisticRegression(
            C=self.selected_c_,
            penalty="l2",
            class_weight="balanced",
            solver=self.solver_,
            max_iter=self.max_iter_,
            tol=self.tolerance_,
            random_state=int(cfg["seed"]),
        )
        self.model_.fit(matrix, y)
        return self

    def score(self, rows: pd.DataFrame) -> pd.Series:
        """Positive-class probabilities, one finite float64 value per input row."""
        if not hasattr(self, "model_"):
            raise RuntimeError("fit must be called before score")
        matrix = self.preprocessor_.transform(rows.copy())
        positive = np.flatnonzero(self.model_.classes_ == 1)
        if len(positive) != 1:
            raise RuntimeError("the fitted classifier has no unique positive class")
        probability = self.model_.predict_proba(matrix)[:, int(positive[0])].astype(
            np.float64, copy=False
        )
        probability[~np.isfinite(probability)] = self.nonfinite_score_fallback_
        return pd.Series(probability, index=rows.index, dtype="float64")

    def coefficient_table(self, top_n: int) -> pd.DataFrame:
        """Top positive and negative standardised-feature coefficients.

        Coefficients are associations with the fitted log odds, not evidence of causation.
        ``top_n`` applies separately to the positive and negative directions.
        """
        if not hasattr(self, "model_"):
            raise RuntimeError("fit must be called before coefficients are available")
        if top_n <= 0:
            raise ValueError("top_n must be positive")

        coefficients = pd.DataFrame(
            {
                "feature": self.preprocessor_.get_feature_names_out(),
                "coefficient": self.model_.coef_[0].astype(np.float64, copy=False),
            }
        )
        positive = (
            coefficients.loc[coefficients["coefficient"] > 0]
            .sort_values(["coefficient", "feature"], ascending=[False, True])
            .head(top_n)
            .assign(direction="positive")
        )
        negative = (
            coefficients.loc[coefficients["coefficient"] < 0]
            .sort_values(["coefficient", "feature"], ascending=[True, True])
            .head(top_n)
            .assign(direction="negative")
        )
        result = pd.concat([positive, negative], ignore_index=True)
        result["rank"] = result.groupby("direction").cumcount() + 1
        result["abs_coefficient"] = result["coefficient"].abs()
        result.insert(0, "target", self.target.upper())
        result.insert(1, "C", self.selected_c_)
        return result[
            [
                "target",
                "C",
                "direction",
                "rank",
                "feature",
                "coefficient",
                "abs_coefficient",
            ]
        ]
