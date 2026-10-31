"""Report Revael's train-only P0 preprocessing on train and validation rows.

The script deliberately does not read or transform test years. It writes the columns retained by
P0 and before/after non-finite shares to ``reports/tables``.

    python scripts/report_p0_preprocessing.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from wt_predmaint.config import load_protocol
from wt_predmaint.features.daily import feature_columns
from wt_predmaint.features.prepare import TrainOnlyPreprocessor
from wt_predmaint.models.base import attach_labels, labelled, scoring_rows

ROOT = Path(__file__).resolve().parents[1]


def make_preprocessor(columns: list[str], cfg: dict[str, Any]) -> TrainOnlyPreprocessor:
    """Build P0 from its protocol block without learning any values yet."""
    settings = cfg["preprocessing_p0"]
    raw_clip = settings["clip_quantiles"]
    clip = None if raw_clip is None else (float(raw_clip[0]), float(raw_clip[1]))
    return TrainOnlyPreprocessor(
        columns,
        min_observed_share=float(settings["min_observed_share"]),
        add_missing_indicators=bool(settings["add_missing_indicators"]),
        scale=bool(settings["scale"]),
        clip_quantiles=clip,
    )


def nonfinite_share(frame: pd.DataFrame, columns: list[str]) -> float:
    """Share of cells that are NaN or infinity."""
    values = frame.loc[:, columns].to_numpy(dtype=float)
    return float((~np.isfinite(values)).mean()) if values.size else 0.0


def column_rows(
    target: str, prep: TrainOnlyPreprocessor
) -> list[dict[str, Any]]:
    """One audit row per requested input column."""
    rows: list[dict[str, Any]] = []
    retained = set(prep.retained_columns_)
    sparse = set(prep.dropped_sparse_)
    for column in prep.input_columns_:
        if column in retained:
            reason = "kept"
        elif column in sparse:
            reason = "sparse"
        else:
            reason = "constant"
        rows.append(
            {
                "target": target.upper(),
                "column": column,
                "observed_share_train": prep.observed_share_[column],
                "n_unique_train": prep.n_unique_.get(column, np.nan),
                "kept": column in retained,
                "drop_reason": reason,
                "median": prep.medians_.get(column, np.nan),
                "mean": prep.means_.get(column, np.nan),
                "scale": prep.scales_.get(column, np.nan),
                "missing_indicator": f"{column}__missing" in prep.indicator_columns_,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interim", type=Path, default=ROOT / "data" / "interim")
    args = parser.parse_args()

    cfg = load_protocol()
    features = pd.read_parquet(args.interim / "features" / "daily_features.parquet")
    columns = feature_columns(features)
    columns_report: list[dict[str, Any]] = []
    missingness_report: list[dict[str, Any]] = []

    for target in ("a", "b"):
        labels = pd.read_parquet(args.interim / "labels" / f"{target}_labels.parquet")
        table = attach_labels(features, labels, cfg)
        prep = make_preprocessor(columns, cfg).fit(labelled(table, "train"))
        columns_report.extend(column_rows(target, prep))

        for split in ("train", "val"):
            rows = scoring_rows(table, split)
            prepared = prep.transform(rows)
            missingness_report.append(
                {
                    "target": target.upper(),
                    "split": split,
                    "n_rows": len(rows),
                    "n_input_columns": len(columns),
                    "n_retained_columns": len(prep.retained_columns_),
                    "n_output_columns": len(prepared.columns),
                    "nonfinite_share_before": nonfinite_share(rows, columns),
                    "nonfinite_share_after": nonfinite_share(
                        prepared, list(prepared.columns)
                    ),
                }
            )

    output = ROOT / "reports" / "tables"
    output.mkdir(parents=True, exist_ok=True)
    columns_frame = pd.DataFrame(columns_report)
    missingness_frame = pd.DataFrame(missingness_report)
    columns_frame.to_csv(output / "p0_columns.csv", index=False)
    missingness_frame.to_csv(output / "p0_missingness.csv", index=False)
    print(missingness_frame.round(4).to_string(index=False))
    print(f"wrote {output / 'p0_columns.csv'}")
    print(f"wrote {output / 'p0_missingness.csv'}")


if __name__ == "__main__":
    main()
