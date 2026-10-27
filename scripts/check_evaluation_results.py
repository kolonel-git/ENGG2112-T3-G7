"""Audit a validation or frozen test table using saved scores; never fit or score models."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from wt_predmaint.config import DEFAULT_PROTOCOL, load_protocol
from wt_predmaint.evaluation.audit import RESULT_KEYS, audit_results
from wt_predmaint.evaluation.runner import variant_name
from wt_predmaint.models.base import KEYS, attach_labels, labelled, scoring_rows

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=Path, default=ROOT / "reports/tables/t0_validation.csv")
    parser.add_argument("--interim", type=Path, default=ROOT / "data/interim")
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--out", type=Path, default=ROOT / "reports/tables/evaluation_audit.csv")
    args = parser.parse_args()
    cfg = load_protocol(args.protocol)
    table = pd.read_csv(args.table, float_precision="round_trip")
    feature_path = args.interim / "features/daily_features.parquet"
    features = pd.read_parquet(feature_path, columns=KEYS)
    inputs = {args.table, args.protocol, feature_path}
    frames, episodes = {}, {}
    for target, target_table in table.groupby("target"):
        label_path = args.interim / f"labels/{target.lower()}_labels.parquet"
        episode_path = args.interim / f"labels/{target.lower()}_episodes.parquet"
        inputs.update([label_path, episode_path])
        prepared = attach_labels(features, pd.read_parquet(label_path), cfg)
        episodes[target] = pd.read_parquet(episode_path)
        keys = set(target_table[RESULT_KEYS].itertuples(index=False, name=None))
        keys.update((v, t, m, "val") for v, t, m, _ in list(keys))
        for variant, _, model, split in sorted(keys):
            excluded = [] if variant == "all" else cfg["baseline_table"]["exclude_stations"]
            if variant != variant_name(excluded):
                raise ValueError(f"unknown variant {variant}")
            rows = scoring_rows(prepared.loc[~prepared.StationId.isin(excluded)], split)
            path = args.interim / f"scores/{target.lower()}_{variant}_{model}_{split}.parquet"
            inputs.add(path)
            scores = pd.read_parquet(path)
            joined = rows.merge(scores, on=KEYS, how="outer", validate="one_to_one", indicator=True)
            if not joined["_merge"].eq("both").all():
                raise ValueError(f"{path}: saved scores do not cover exactly the feature days")
            frames[(variant, target, model, split)] = labelled(joined, split)
    result = audit_results(table, frames, episodes, cfg)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out, index=False)
    manifest = {
        "scope": "Saved-result consistency only; no model fitting or scoring. G3 is not certified.",
        "sha256": {
            str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p): hashlib.sha256(
                p.read_bytes()
            ).hexdigest() for p in sorted(inputs)
        },
    }
    args.out.with_suffix(".json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Verified {len(result)} saved results (including intervals and random floors)")


if __name__ == "__main__":
    main()
