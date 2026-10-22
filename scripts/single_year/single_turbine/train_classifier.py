"""Cross-validate a classifier on only one turbine within one year (default 2018)."""

from pathlib import Path
from runpy import run_path

STATION = 2304510  # Change this or pass --station; --year defaults to 2018.

if __name__ == "__main__":
    runner = Path(__file__).resolve().parents[1] / "all_turbines/train_classifier.py"
    run_path(str(runner))["main"](default_station=STATION)
