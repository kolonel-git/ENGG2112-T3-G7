# Predicting Wind Turbine Degradation from SCADA

ENGG2112 (University of Sydney), Group 7, "AI for Good": explainable predictive maintenance
on open wind-farm SCADA data.

## Dataset
- Hill of Towie wind farm open dataset (RES/TRIG), CC-BY-4.0, 21 Siemens SWT-2.3-VS-82 turbines, 10-minute SCADA.
- Provisionally pinned to **v2.1.0**: DOI [10.5281/zenodo.22662930](https://doi.org/10.5281/zenodo.22662930)
  (concept DOI for all versions: 10.5281/zenodo.14870021). The brief cited 10.5281/zenodo.14870023, which is **v1.0.0** and ends before our 2025 to 2026 data; version to be confirmed (`docs/DECISIONS.md` D-004).
- Tables used: `tblSCTurTemp`, `tblSCTurbine`, `tblSCTurPress`, `tblAlarmLog`, `tblDailySummary`.
- TODO: add the exact citation text from the Zenodo record page.

## Quickstart
- `git clone <repo-url>` then `cd` into it
- `python -m venv .venv`; activate (`.venv\Scripts\activate` on Windows, `source .venv/bin/activate` on macOS/Linux)
- `pip install -e ".[dev]"`
- `pre-commit install` and `nbstripout --install` (strips notebook outputs; see below)
- Put the raw monthly CSVs in `data/raw/<year>/` (or `python scripts/download_data.py --list`, then `python scripts/download_data.py`; all years are ~15 GB for the yearly zips)
- `python scripts/build_alarm_table.py` builds `data/interim/alarm_events.parquet` from the raw alarm logs
- `python scripts/build_scada_parquet.py --workers 4` builds one parquet per SCADA table and year in `data/interim/scada/`
- `python scripts/build_clean_data.py` joins and cleans them into `data/interim/clean/` (`scada_<year>.parquet`, `daily_summary.parquet`) and writes `reports/tables/cleaning_report.csv`
- Metadata only: `python scripts/download_data.py --only Hill_of_Towie_data_dictionary.md --out data/raw/metadata`

## Folder map
- `src/wt_predmaint/`: all reusable code (`data`, `features`, `labels`, `models`, `evaluation`, `explain`)
- `notebooks/`: exploration only, named `NN_owner_topic.ipynb`
- `configs/protocol.yaml`: the single experiment protocol (many values still TODO)
- `data/{raw,interim,processed}`: git-ignored data
- `reports/{figures,tables}`: generated outputs
- `docs/`: `roadmap.md` (schedule, owners, gates), `DECISIONS.md` (decision log), `DATA_TRAPS.md`, `modelling_plan.md`, `reference/` (alarm code and processed-data descriptions)
- `tests/`, `scripts/`

## Tests and lint
- `pytest`
- `ruff check .`

## Notebook outputs
- Outputs must not be committed. Run `nbstripout --install` once per clone (the `pre-commit` hook also strips them).

## Licence
- TODO: no licence chosen yet. Team must decide before any public release.
