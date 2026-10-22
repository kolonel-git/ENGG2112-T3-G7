# Baseline work split

Four people, four packages. Written 2026-10-22. It turns the baseline list in
[modelling_plan.md](modelling_plan.md) section 7.3 (with D-011) into work orders. Read
[FEATURES_AND_LABELS.md](FEATURES_AND_LABELS.md) and decisions D-019 to D-022 in
[DECISIONS.md](DECISIONS.md) first. The rules in [CLAUDE.md](../CLAUDE.md) apply to everything here.

Nothing below is a result. Numbers appear only once someone has computed them with the scripts.

## 1. Summary

| Person | Package | Baselines | Size |
|---|---|---|---|
| **Kanav** | Model interface and runner, then the normal-behaviour models | interface, runner, R2 (ridge normal-behaviour model) | Medium, then large |
| **Parvez** | Healthy-row definition and the power-curve proxy | H0 (healthy mask), B3 (power-curve proxy) | Large |
| **Revael** | Train-only preprocessing and the linear model | P0 (preprocessing), B4 (logistic regression) | Medium |
| **Farhan** | Rule baselines, the z-score reference and the results table | B1, B2, R1, T0 (results table) | Medium |

Cyrus (ethics and report) has no baseline package. Cyrus's input is the false-alarm and cost framing,
which the results tables feed later.

Already done, not part of the split: B0a prevalence dummy and B0b random-score floor
(`scripts/run_baselines.py`, `reports/tables/baselines_<target>.csv`).

### Order of work

```
Day 0   Kanav: interface stub (base.py)  -->  everyone codes against it
Step 1  (parallel)  Kanav: runner     Parvez: H0 healthy mask   Revael: P0 preprocessing
                    Farhan: B1, B2 (rules need only the interface)
Step 2  (parallel)  Parvez: B3        Revael: B4       Farhan: R1 (needs H0)    Kanav: R2 (needs H0)
Step 3  Farhan: T0 results table, all baselines through the runner, validation years only
Gate G2 baseline results on validation, thresholds from validation only
```

Test years (2023 to 2026-04) are scored once per frozen configuration at Gate G3, not during
development. Until then everything is evaluated on validation (2021 to 2022).

## 2. Shared contract

### 2.1 Interface (`src/wt_predmaint/models/base.py`, merged with D-023)

Every baseline is a class with this shape. Code against it from day one; the stub will not change
without a message to the group.

```python
class Baseline(Protocol):
    name: str                      # short id used in tables, e.g. "B1_recent_fault"
    def fit(self, train: pd.DataFrame, cfg: dict) -> "Baseline": ...
    def score(self, rows: pd.DataFrame) -> pd.Series: ...   # higher = riskier, index aligned to rows
```

- `train` and `rows` are feature-table rows joined to labels: `StationId`, `Day`, `label`, `status`,
  `split`, and every feature column. The target (`a` or `b`) is a constructor argument.
- `fit` sees train-year rows only. It must not read validation or test rows, from anywhere.
- `score` must work for **every** day that has features, including days with no label (D-022). Metrics
  later keep only labelled rows.
- Call `require_train_only(train)` first in `fit`, and `check_score_contract(fitted, rows)` in your tests. `models/dummy.py` shows a complete example.
- No missing scores. If a baseline cannot score a day, it returns a documented fallback value, not NaN.
- All randomness reads `seed` from `configs/protocol.yaml`.

### 2.2 How a baseline is evaluated (Kanav's runner)

The runner calls `fit` on train rows, scores validation rows, picks the threshold with
`threshold_for_budget` on validation, and writes a row with `evaluation_row`
([metrics.py](../src/wt_predmaint/evaluation/metrics.py), D-020). Until the runner is merged, owners can
call those two functions directly in a throwaway script.

Every baseline is reported for **both targets (A and B)** with: PR-AUC with interval, recall at the
budget of 1 false alarm per turbine-month, median lead time, the realised false-alarm rate, the number
of evaluable episodes, and the random-score floor next to it. Intervals bootstrap turbines (D-009).

### 2.3 Tie warning for rule baselines

A rule score such as "number of alarms in 7 days" has many tied values. Alerts are `score > threshold`
(strict), so at a tight budget a tied score can leave zero alerts and recall 0. Handle it by one of:
adding a documented deterministic tie-breaker (for example a second feature, scaled to be smaller than
one step of the main score), or reporting that the rule cannot meet the budget. Do not hide it. Record
the choice in a decision entry.

### 2.4 Decision IDs (avoid merge conflicts)

Several people will add entries to `docs/DECISIONS.md` at once. To keep IDs unique, each person uses
their own block:

| Person | IDs |
|---|---|
| Kanav | D-023 to D-029 |
| Parvez | D-030 to D-039 |
| Revael | D-040 to D-049 |
| Farhan | D-050 to D-059 |

Change `configs/protocol.yaml` only together with the entry, and add new keys under a new block named
for your package, so edits to the file do not collide.

### 2.5 Definition of done (every package)

- Code in `src/wt_predmaint/`, type hints, no hard-coded windows, years, budgets or thresholds.
- Unit tests on synthetic data, including a **fit-leakage test**: changing validation or test rows
  must not change a fitted baseline. (The feature look-ahead test is separate; see 2.6.)
- `ruff check .` and `pytest` pass.
- A script in `scripts/` that reproduces your table in `reports/tables/`.
- A decision entry for every choice not already fixed, including thresholds, variants and exclusions.
- One PR per baseline, with a title and a summary (what, why, results, things to review, how to verify,
  not included).
- Results quoted in the PR are the numbers the script printed. No invented numbers or references.

### 2.6 If you add a feature

Any new daily feature must be built from data to the end of its day only and must pass
`scripts/check_feature_leakage.py` (extend its sources if the feature needs a new one).

## 3. Kanav: interface, runner and the ridge normal-behaviour model

### 3.1 Interface stub (`models/base.py`), first
The Protocol above plus a small helper that joins the feature table to a label table for one target and
one split. Merge it before anyone else's package, in a PR of its own.

### 3.2 Runner (`evaluation/runner.py`, `scripts/run_baselines.py`; merged with D-024)
To add your baseline: write the class, add one line to `models/registry.py`, then run
`python scripts/run_baselines.py --target a --baselines <your id>`.

- Input: a list of baselines, a target, and which splits to score.
- For each baseline: fit on train, score all days with features, restrict to labelled rows for the
  metrics, pick the threshold on validation, write one row per (baseline, target, split) with
  `evaluation_row`.
- Always include the random floor and the constant score as rows.
- Output: `reports/tables/baselines_<target>.csv` and a printed table. Test is scored only with an
  explicit flag, and the script prints a warning that test is for frozen configurations.
- Tests: a trivial baseline gives hand-computable metrics; a baseline that peeks at validation labels
  is caught by the fit-leakage test.

### 3.3 R2: ridge normal-behaviour model (O1), then LightGBM
- Predict each chosen sensor from operating conditions (wind speed, rotor speed, ambient and nacelle
  temperature, with lags because temperatures respond slowly). Fit on healthy train rows only, using
  H0 from Parvez.
- Residual = observed minus predicted, smoothed over a window (choose on validation, log it),
  standardised with the train healthy residual spread.
- Output: a daily indicator per turbine and per sensor, plus residual columns that can join the feature
  table for O2 (must pass the look-ahead test).
- Targets for the models come from section 6.1 of the modelling plan (gearbox oil, generator bearings,
  main bearing, windings, transformer, hydraulic oil). Prune the list in a decision entry.
- Acceptance: plots against fault episodes with counts of how often the indicator rises before a
  fault, computed on validation. The LightGBM version follows after the baselines.

## 4. Parvez: healthy rows (H0) and the power-curve proxy (B3)

### 4.1 H0: healthy-row mask (`models/healthy.py`), first
Rows that describe normal behaviour. B3, R1 and R2 all use it, so it comes first and is a small PR.
Proposed definition (change in a decision entry if you disagree):
- Day-level mask, train years only.
- Label status is `negative`, so no episode starts in the next 60 days, and the day is not inside an
  episode.
- No episode ended in the previous N days, so recovering days are not "healthy" (N: choose and log).
- Not in an `outage_month`, and not a `suspect_wind` turbine-year (D-015, D-017).
- 10-minute version for B3: the day is healthy, and the interval has rpm and wind present, wind at or
  above cut-in and not in a stop run.
- Provide `healthy_days(labels, features, cfg, target)` and a 10-minute variant, plus a short count of
  rows kept per turbine and year (a table in `reports/tables/`).
- Test: no validation or test day can ever be marked healthy.

### 4.2 B3: power-curve proxy (D-011)
**Idea:** a healthy turbine at a given wind speed turns at a given rotor speed. Learn that curve, then
flag days where the turbine persistently runs below it. There is no measured power in our files
(D-018), so rotor speed against wind speed stands in for the power curve.

**Steps:**
1. Fit the curve on healthy train rows from the clean 10-minute data (`data/interim/clean/`): bin
   primary wind speed (bin width: choose and log, for example 0.5 m/s) and take a robust centre of
   generator rpm per bin. Decide pooled for the fleet or per turbine; the fleet is very homogeneous
   (see the modelling plan), so pooled is the default.
2. For every 10-minute interval: shortfall = curve(wind) minus observed rpm, only where wind is in the
   working range.
3. Daily features: share of intervals with shortfall above a tolerance, and the mean shortfall. The
   tolerance comes from the spread of the train healthy shortfall (for example a multiple of its
   standard deviation; log the multiple).
4. Window: trailing 7-day mean of the daily share (`labels.feature_window_days`).
5. Score = that 7-day value. Higher means more persistent underperformance.

**Rules that matter here:**
- Day assignment follows D-013 and D-021: an interval belongs to the day of its start.
- 2019 has no temperature or pressure data and is unused. The four `suspect_wind` turbine-years carry a
  bad wind signal and need an explicit rule (exclude from the fit, or from scoring, or both).
- Intervals in stop runs and in outage months must not pollute the curve.
- The new daily columns must pass `check_feature_leakage.py`.

**Deliverables:** `models/power_curve.py`, `scripts/run_b3_power_curve.py`, tests (the curve recovers a
known synthetic curve; shortfall is zero on healthy synthetic data and positive on injected
underperformance), a curve plot in `reports/figures/`, and results for A and B on validation.
**Decisions to log:** bin width, working wind range, pooled versus per turbine, tolerance multiple,
handling of suspect-wind turbine-years.

## 5. Revael: preprocessing (P0) and logistic regression (B4)

### 5.1 P0: train-only preprocessing (`features/prepare.py`), first
Shared by B4 and the later models. A fitted object with `fit(train)` and `transform(rows)`:
- **Column selection:** keep columns observed in at least a set share of train rows. About 12 of the
  110 base sensor columns are over 50% missing because they only exist in some years. Choose the share
  in a decision entry (D-021 left it open) and report which columns are dropped.
- **Constant columns:** drop columns with no variation in train.
- **Imputation:** median per column from train. Decide whether to add missing-value indicator columns.
- **Scaling:** mean and standard deviation from train, for the linear model. A flag turns scaling off
  for tree models.
- Optional: clip extreme values at train quantiles (log the quantiles).
- Output: a numeric matrix and the list of final column names; the same object applied unchanged to
  validation and test.
- Tests: statistics come from train only (changing validation rows does not change the fit); constant
  and sparse columns are dropped; transform is deterministic.
- Also report per-split missing-value shares after preparation (`reports/tables/`).

### 5.2 B4: class-weighted logistic regression
- Inputs: the prepared window features. Allowed families: `w7mean_`, `w7std_`, the alarm columns, the
  day-state columns and the fault-history columns. Not allowed: keys, quality flags and counts (see
  `feature_columns`).
- Model: `LogisticRegression` with `class_weight="balanced"` and L2 penalty. Choose `C` from a small
  grid (for example 0.01, 0.1, 1) on **validation**, log the grid and the choice. Fixed solver and
  `max_iter`, `random_state` from the seed.
- Because the columns are heavily correlated, also try an L1 or elastic-net variant only if time allows,
  and say so as an extra.
- Both targets, separately.
- Output: scores for every day; a coefficient table with the largest weights (a first read of what
  matters; it is attribution, not cause); results for A and B on validation.
- Tests: scores are higher for days drawn from the positive class on a synthetic separable set; the
  class weights are computed from train rows only.
- **Decisions to log:** column share threshold (with P0), the C grid and choice, with or without
  missing indicators.

## 6. Farhan: rule baselines, z-score reference and the results table

### 6.1 B1: recent-fault rule (`models/rules.py`)
- Score from `days_since_fault_day` (smaller means riskier) and `fault_days_30` (larger means
  riskier). Define one primary form and say why, for example: `fault_days_30`, with
  `days_since_fault_day` as the tie-breaker. Also report the simplest single-column versions in the
  table, so readers see what the combination adds.
- `days_since_fault_day` is empty before the first fault in the data: choose and log the value used
  (for example "very long ago").
- These columns come from the **target A** rule. Use them for both targets as the default, and note in
  the table that target B uses A's history. If time allows, add a stop-based history feature for B
  (extend the features, pass the look-ahead test, log it).
- No fitting. See 2.3 for ties.

### 6.2 B2: recent-alarm-count rule
- Score from the 7-day alarm columns. Variants to run and report: `alarm_total_w7`,
  `alarm_tier2_w7`, tier 1 plus 2, and one per documented code. Pick **one** primary variant by
  validation PR-AUC, and report the others next to it.
- Expect weather-driven alarms (icing, high and low wind) to make this partly a seasonal rule. Check the
  per-month pattern of the score and mention it. The 2022 drop in codes 20 and 25 affects the counts
  across years; state it.
- No fitting beyond choosing the variant. See 2.3 for ties.

### 6.3 R1: per-turbine z-score reference (O1, no model)
- For a list of sensors (for example gearbox oil, generator bearings, main bearing, windings): how far
  is the turbine's recent value from its own healthy history? Per turbine, mean and standard deviation
  of the `op_` value over its **healthy train days** (H0), then the z-score of the 7-day mean.
- Score for O2: the largest absolute z across the chosen sensors, or their mean (choose and log).
- This reference is naive on purpose (it ignores wind and load). It is the bar the ridge
  normal-behaviour model R2 must clear. Report both on the same plots.
- Turbines or years with too few healthy days need a documented fallback (for example the fleet value).

### 6.4 T0: the results table (`scripts/make_baseline_table.py`)
- One table, rows = baseline x target, columns = PR-AUC with interval, recall at the budget with
  interval, median lead time, realised false-alarm rate, number of evaluable episodes, prevalence.
- Always includes the constant and random floors, so every number sits next to what chance gives
  (D-020).
- Validation years only for now. A second table version without turbine 2304515 (positive on 97% of its
  validation days) for sensitivity.
- A short read-out in `reports/` for the report: which baselines beat the random floor, by how much,
  and where they do not.

## 7. What "good" looks like at Gate G2

- Every baseline above runs through the runner for targets A and B on validation.
- The table shows each result with its interval and the random floor beside it.
- Thresholds come from validation only. Test is untouched.
- Each package has a PR with tests and decision entries, and `ruff check .` and `pytest` pass.
- Known weak spots are written down: rule baselines with ties, the proxy for power, the high prevalence
  (37% to 50%), autocorrelated rows, and the fact that A and B are not independent.

## 8. Open points that affect more than one package

- Missingness cutoff for columns (P0 and B4, Revael).
- The healthy-row definition (H0, Parvez) feeds B3, R1 and R2.
- Whether to exclude the four `suspect_wind` turbine-years and turbine 2304515 per model (open decision
  in the log).
- Target B has no stop-based history feature yet (B1).
