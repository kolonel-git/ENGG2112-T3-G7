"""Visualise results trained on only one turbine; does not retrain the model."""

from pathlib import Path
from runpy import run_path

STATION = 2304510  # Match training, or pass --station explicitly.

if __name__ == "__main__":
    runner = Path(__file__).resolve().parents[1] / "all_turbines/plot_classifier.py"
    run_path(str(runner))["main"](default_station=STATION)
