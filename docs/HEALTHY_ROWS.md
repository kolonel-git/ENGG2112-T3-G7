# H0: healthy rows for fitting normal behaviour

Implemented in [models/healthy.py](../src/wt_predmaint/models/healthy.py), decision
[D-030](project/DECISIONS.md). H0 supplies boolean fitting masks for B3, R1 and
R2; it is not itself a classifier implementing `Baseline.score`.

## Daily mask

```python
from wt_predmaint.models.base import attach_labels, require_train_only
from wt_predmaint.models.healthy import healthy_days, healthy_intervals

# labels is the COMPLETE label table for target a, including unlabelled days.
mask = healthy_days(labels, features, cfg, target="a")
healthy_train = attach_labels(features.loc[mask], labels, cfg)
require_train_only(healthy_train)
```

The mask is indexed exactly like `features`; no input rows are removed or
modified. Only rows satisfying all of these rules can be true:

- Within `healthy.start_year` to `healthy.end_year` (2016–2025), in configured
  `splits.train_years`, and carrying a declared train label split. A feature
  table's own split, when supplied, must also be train.
- Target-specific status `negative`, with label 0: no new episode in the next
  60 days under the existing horizon-label definition, and not currently in an episode.
- Both `outage_month` and `suspect_wind` explicitly false. Missing flags do not
  establish healthy behaviour; missing required columns raise a clear error.
- Seven preceding calendar days of known train-label history without an
  `in_episode` day. This excludes days D+1 through D+7 after an episode ends
  on D; D+8 may qualify. The current day is already excluded by its status if
  it is inside an episode.

The recovery window is configured as `healthy.recovery_days: 7`. At training
block boundaries or after missing label history, rows are conservatively excluded
until the full preceding window is known. No validation/test history is borrowed
to establish recovery. Statuses such as `zone` and `purged` may establish that
a past train day was outside an episode, but they cannot themselves be healthy.

Pass the full label history, not just negative or labelled rows: dropping
in-episode/unlabelled days before this call prevents proving recovery. A/B use
their own supplied label tables; A's fault-history feature is not used to infer
B's recovery. No episode files, raw tables or held-out data are read internally.

Training labels legitimately select normal-behaviour fitting examples; this mask
must not be used as a future-aware input to a prediction-time risk scorer.

## Ten-minute mask

```python
interval_mask = healthy_intervals(labels, features, clean_scada, cfg, target="a")
healthy_intervals_for_fit = clean_scada.loc[interval_mask]
```

The output is indexed exactly like `clean_scada`. It requires a healthy day,
clear interval quality flags, finite generator rpm and primary wind speed, and
wind at least `healthy.cut_in_wind_ms` (4.0 m/s). The 4.0 value is an explicit
operating proxy aligned with the existing stop definition, not a verified
manufacturer cut-in specification.

Whole qualifying stop runs are excluded using the shared `stop_runs` logic:
currently at least 18 consecutive 10-minute intervals with rpm below 200 and
wind at least 4.0 m/s, outside outage periods. Thresholds come from the protocol.
Runs are found on the complete clean training interval history before selecting
healthy days, including runs spanning midnight or consecutive training years.
Pass that complete history, not a preselected healthy batch.

An interval stamped 00:00 belongs to the previous day: day assignment uses
`TimeStamp - 10 minutes`. Calendar and optional declared split gates apply before
stop-run detection, so held-out intervals cannot change training selections.

## Reproduce counts

From the repository root, in the project Python environment:

```powershell
python scripts/report_healthy_rows.py
```

By default this reads shared prepared files under `data/interim/`, handles both
targets, and writes [reports/tables/healthy_rows.csv](../reports/tables/healthy_rows.csv).
Use `--targets a`, `--interim PATH`, `--config PATH`, or `--output PATH` to override.
Annual SCADA input is concatenated across training years before masking so
year-boundary stop runs are not split artificially.

The report has one row per target/turbine/year across 2016–2025, including
feature-day counts, healthy-day counts, source-file availability, interval-row
counts and healthy-interval counts. Missing annual clean files have **blank**
interval counts; zero means an available source contributed no selected intervals.
Feature-day count zero makes absent prepared feature years visible. The scope
does not make all ten years training years: configured validation/test years
always have zero kept rows.

For the currently available shared files, the script computed 2,291 healthy A
days and 1,735 healthy B days, with 247,738 and 169,163 healthy intervals
respectively. These are partial-source results: the shared clean directory only
contains 2018 and 2019 files, and 2019 is not a configured training year. Other
clean years are marked unavailable, not claimed to have been analysed. Rebuild
the shared pipeline and rerun this report when the remaining years are prepared.

Tests cover held-out and wrong-split rejection, recovery boundaries, missing
history/flags, turbine isolation, midnight/year-boundary runs, finite-value and
cut-in checks, input/index preservation, and reporting unavailable years.
# Integrated notebook (2026-10-22)

Open `notebooks/02_healthy_row_diagnostics_and_baseline_evaluation.ipynb` for all-turbine
coverage checks, H0 retention charts, registered-model validation, metric
comparisons, precision–recall curves and risk timelines. Change `MODEL_IDS`
in the configuration cell to select registered models. H0 is a training mask,
not a risk predictor; it is audited separately. Preparation and frozen test
evaluation default to off. Tables and figures go to
`reports/notebook_all_turbines/`. All code cells were executed successfully
against the current prepared data with reference models; the 11 H0 tests pass.

