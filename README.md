# Predicting Wind Turbine Degradation from SCADA

ENGG2112 (University of Sydney), Group 7, "AI for Good": explainable predictive maintenance
on open wind-farm SCADA data.

## Dataset
- Hill of Towie wind farm open dataset (RES/TRIG), CC-BY-4.0, 21 Siemens SWT-2.3-VS-82 turbines, 10-minute SCADA.
- Pinned to **v2.1.0**: DOI [10.5281/zenodo.22662930](https://doi.org/10.5281/zenodo.22662930)
  (concept DOI for all versions: 10.5281/zenodo.14870021).
- Note: brief cited 10.5281/zenodo.14870023; that DOI is **v1.0.0**. Team to confirm (see `docs/DECISIONS.md`).
- Tables used: `tblSCTurTemp`, `tblSCTurbine`, `tblSCTurPress`, `tblAlarmLog`.
- TODO: add the exact citation text from the Zenodo record page.

## Quickstart
- `git clone <repo-url>` then `cd` into it
- `python -m venv .venv`; activate (`.venv\Scripts\activate` on Windows, `source .venv/bin/activate` on macOS/Linux)
- `pip install -e ".[dev]"`
- `pre-commit install` and `nbstripout --install` (strips notebook outputs; see below)
- `python scripts/download_data.py --list`, then `python scripts/download_data.py` (all years, ~15 GB) or `--years 2022 2023`

## Folder map
- `src/wt_predmaint/`: all reusable code (`data`, `features`, `labels`, `models`, `evaluation`, `explain`)
- `notebooks/`: exploration only, named `NN_owner_topic.ipynb`
- `configs/protocol.yaml`: the single experiment protocol (many values still TODO)
- `data/{raw,interim,processed}`: git-ignored data
- `reports/{figures,tables}`: generated outputs
- `docs/`: `roadmap.md`, `DECISIONS.md`, `DATA_TRAPS.md`
- `tests/`, `scripts/`

## Tests and lint
- `pytest`
- `ruff check .`

## Notebook outputs
- Outputs must not be committed. Run `nbstripout --install` once per clone (the `pre-commit` hook also strips them).

## Licence
- TODO: no licence chosen yet. Team must decide before any public release.
