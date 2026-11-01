"""Run the two-hour-history any-alarm experiment without a notebook."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
os.environ.setdefault("MPLBACKEND", "Agg")

from wt_predmaint.short_horizon import (  # noqa: E402
    development,
    evaluate_test,
    fit_models,
    load_config,
    new_run,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO / "configs/short_horizon_alarm.json")
    parser.add_argument(
        "--evaluate-test",
        action="store_true",
        help="Evaluate 2024–2025 after all models/thresholds are frozen.",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    out = new_run(REPO, cfg)
    print(f"Output: {out}", flush=True)
    prepared = development(REPO, cfg, out)
    print(fit_models(cfg, prepared).to_string(index=False), flush=True)
    if args.evaluate_test:
        print(evaluate_test(REPO, cfg, prepared).to_string(index=False), flush=True)
    else:
        print("Development complete. Test data was not opened.", flush=True)


if __name__ == "__main__":
    main()
