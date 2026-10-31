"""Train-only feature preparation for predictive models (P0).

The preprocessor deliberately receives an explicit, ordered list of model columns.  It learns
column selection, optional clipping, median imputation and scaling from labelled training rows
only, then reuses those frozen values for every row it transforms.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype

from wt_predmaint.models.base import KEYS, LABEL_COLUMNS, require_train_only

MISSING_SUFFIX = "__missing"
FORBIDDEN_COLUMNS = {*KEYS, *LABEL_COLUMNS}


class TrainOnlyPreprocessor:
    """Prepare numeric features without learning anything from validation or test rows.

    Missing indicators, when requested, are frozen to columns that contain at least one missing
    value during fitting.  Numeric features are returned first in their supplied order, followed
    by unscaled binary indicators in the same source-column order.
    """

    def __init__(
        self,
        columns: Sequence[str],
        *,
        min_observed_share: float,
        add_missing_indicators: bool,
        scale: bool = True,
        clip_quantiles: tuple[float, float] | None = None,
    ) -> None:
        if isinstance(columns, str):
            raise TypeError("columns must be a sequence of column names, not a string")
        self.columns = list(columns)
        if not self.columns:
            raise ValueError("columns must contain at least one feature name")
        if not all(isinstance(column, str) for column in self.columns):
            raise TypeError("every feature column name must be a string")

        duplicates = list(dict.fromkeys(c for c in self.columns if self.columns.count(c) > 1))
        if duplicates:
            raise ValueError(f"feature column names must be unique; duplicates: {duplicates}")
        forbidden = sorted(FORBIDDEN_COLUMNS.intersection(self.columns))
        if forbidden:
            raise ValueError(f"metadata columns cannot be model inputs: {forbidden}")

        if not isinstance(min_observed_share, (int, float)) or isinstance(
            min_observed_share, bool
        ):
            raise TypeError("min_observed_share must be a number")
        if not np.isfinite(float(min_observed_share)) or not 0 < min_observed_share <= 1:
            raise ValueError("min_observed_share must be finite and in (0, 1]")
        if not isinstance(add_missing_indicators, bool):
            raise TypeError("add_missing_indicators must be a bool")
        if not isinstance(scale, bool):
            raise TypeError("scale must be a bool")

        if clip_quantiles is not None:
            if not isinstance(clip_quantiles, tuple) or len(clip_quantiles) != 2:
                raise TypeError("clip_quantiles must be a (lower, upper) tuple or None")
            lower, upper = clip_quantiles
            if any(
                isinstance(value, bool) or not isinstance(value, (int, float))
                for value in (lower, upper)
            ):
                raise TypeError("clip quantiles must be numbers")
            if not all(np.isfinite(float(value)) for value in (lower, upper)):
                raise ValueError("clip quantiles must be finite")
            if not 0 <= lower < upper <= 1:
                raise ValueError("clip quantiles must satisfy 0 <= lower < upper <= 1")
            clip_quantiles = (float(lower), float(upper))

        self.min_observed_share = float(min_observed_share)
        self.add_missing_indicators = add_missing_indicators
        self.scale = scale
        self.clip_quantiles = clip_quantiles
        self._is_fitted = False

    @staticmethod
    def _numeric_frame(rows: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
        """Copy ``columns`` as float64, treating all non-finite values as missing."""
        missing = [column for column in columns if column not in rows.columns]
        if missing:
            raise KeyError(f"rows are missing fitted feature columns: {missing}")

        duplicated = [column for column in columns if int((rows.columns == column).sum()) != 1]
        if duplicated:
            raise ValueError(f"feature columns must occur exactly once in rows: {duplicated}")

        non_numeric = [
            column
            for column in columns
            if is_bool_dtype(rows[column].dtype) or not is_numeric_dtype(rows[column].dtype)
        ]
        if non_numeric:
            raise TypeError(f"feature columns must be numeric and not boolean: {non_numeric}")

        frame = rows.loc[:, list(columns)].astype("float64").copy()
        return frame.where(np.isfinite(frame), np.nan)

    def fit(self, train: pd.DataFrame) -> TrainOnlyPreprocessor:
        """Learn preparation values from labelled training rows only."""
        require_train_only(train)
        self._is_fitted = False
        if train.empty:
            raise ValueError("cannot fit preprocessing on an empty training frame")

        self.input_columns_ = list(self.columns)
        numeric = self._numeric_frame(train, self.input_columns_)
        self.observed_share_ = numeric.notna().mean().astype("float64")

        self.dropped_sparse_ = [
            column
            for column in self.input_columns_
            if self.observed_share_[column] < self.min_observed_share
        ]
        sufficiently_observed = [
            column for column in self.input_columns_ if column not in self.dropped_sparse_
        ]
        self.n_unique_ = numeric[sufficiently_observed].nunique(dropna=True).astype("int64")
        self.dropped_constant_ = [
            column for column in sufficiently_observed if self.n_unique_[column] <= 1
        ]
        self.retained_columns_ = [
            column
            for column in sufficiently_observed
            if column not in self.dropped_constant_
        ]
        if not self.retained_columns_:
            raise ValueError("no feature columns remain after sparse and constant-column removal")

        values = numeric.loc[:, self.retained_columns_]
        missing = values.isna()
        self.indicator_source_columns_ = (
            [column for column in self.retained_columns_ if missing[column].any()]
            if self.add_missing_indicators
            else []
        )
        self.indicator_columns_ = [
            f"{column}{MISSING_SUFFIX}" for column in self.indicator_source_columns_
        ]
        collisions = sorted(set(self.retained_columns_).intersection(self.indicator_columns_))
        if collisions:
            raise ValueError(f"missing-indicator names collide with feature columns: {collisions}")

        if self.clip_quantiles is None:
            self.clip_lower_ = None
            self.clip_upper_ = None
        else:
            lower, upper = self.clip_quantiles
            self.clip_lower_ = values.quantile(lower).astype("float64")
            self.clip_upper_ = values.quantile(upper).astype("float64")
            values = values.clip(lower=self.clip_lower_, upper=self.clip_upper_, axis="columns")

        self.medians_ = values.median().astype("float64")
        prepared = values.fillna(self.medians_)
        self.means_ = prepared.mean().astype("float64")
        scales = prepared.std(ddof=0).astype("float64")
        self.scales_ = scales.where(np.isfinite(scales) & scales.ne(0), 1.0)
        self.feature_names_ = [*self.retained_columns_, *self.indicator_columns_]
        self._is_fitted = True
        return self

    def _require_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError("TrainOnlyPreprocessor must be fitted before use")

    def transform(self, rows: pd.DataFrame) -> pd.DataFrame:
        """Apply the frozen train values and return a finite float64 matrix."""
        self._require_fitted()
        values = self._numeric_frame(rows, self.retained_columns_)
        missing = values.isna()

        if self.clip_lower_ is not None and self.clip_upper_ is not None:
            values = values.clip(
                lower=self.clip_lower_, upper=self.clip_upper_, axis="columns"
            )
        values = values.fillna(self.medians_)
        if self.scale:
            values = (values - self.means_) / self.scales_

        out = values.astype("float64")
        for source, indicator in zip(
            self.indicator_source_columns_, self.indicator_columns_, strict=True
        ):
            out[indicator] = missing[source].to_numpy(dtype="float64")
        out = out.loc[:, self.feature_names_]
        if not np.isfinite(out.to_numpy(dtype="float64")).all():
            raise ValueError("prepared features must be finite")
        return out

    def fit_transform(self, train: pd.DataFrame) -> pd.DataFrame:
        """Fit on ``train`` and transform the same rows."""
        return self.fit(train).transform(train)

    def get_feature_names_out(self) -> list[str]:
        """Return a copy of the frozen output feature names."""
        self._require_fitted()
        return list(self.feature_names_)
