# B1: recent-fault baseline

B1 asks whether recent recorded faults warn of another fault episode starting in the next 30 days.
One prediction is made at the end of each turbine-day. This is a ranking rule, not a probability
model. Both targets use target A's past fault history, as required by BASELINE_SPLIT section 6.1.

## Validation results, 2026-10-22

The fixed combined rule has higher PR-AUC point estimates than the floors, but detects far fewer
distinct episodes at the required false-alarm budget. B1 is a completed benchmark, not evidence of a
useful warning system. PR-AUC uncertainty intervals overlap those of the random floor; no statistical
superiority claim is made. Test years were not scored.

| Target | Score | PR-AUC | Episode recall | Median lead, days | False alarms / turbine-month |
|---|---|---:|---:|---:|---:|
| A | Constant | 0.470 | 0.0% | — | 0.000 |
| A | Random | 0.465 | 78.0% | 20.0 | 0.999 |
| A | **Combined B1 (primary)** | **0.559** | **22.5%** | **27.0** | **0.999** |
| A | Fault count only | 0.533 | 16.3% | 28.0 | 0.499 |
| A | Recency only | 0.528 | 38.8% | 21.0 | 0.807 |
| B | Constant | 0.492 | 0.0% | — | 0.000 |
| B | Random | 0.483 | 76.0% | 21.0 | 1.000 |
| B | **Combined B1 (primary)** | **0.550** | **21.9%** | **27.5** | **0.990** |
| B | Fault count only | 0.530 | 19.3% | 28.0 | 0.972 |
| B | Recency only | 0.531 | 35.9% | 16.0 | 0.972 |

Primary B1's 95% turbine bootstrap intervals:

- A: PR-AUC 0.482–0.647; episode recall 17.7–27.3%; 47 of 209 evaluable episodes detected.
- B: PR-AUC 0.462–0.655; episode recall 14.7–29.9%; 42 of 192 evaluable episodes detected.

Full metrics and intervals: [target A](tables/b1_a.csv), [target B](tables/b1_b.csv).
Lead time describes detected episodes only; it does not account for episodes missed.

Why the metrics differ: PR-AUC ranks individual turbine-days, while recall counts distinct episodes
at one chosen threshold. For A, combined B1 alerts on 541 positive days but detects only 47 episodes;
random alerts on 262 positive days and detects 163 episodes. Both have 318 false-alert days. Thus
B1's alerts are more concentrated around fewer episodes. For B, the corresponding figures are
412 positive-alert days / 42 episodes for B1, versus 262 / 146 for random. These counts come from
saved scores joined to labelled validation days at the stored thresholds.

The count-only rule cannot spend the whole false-alarm budget because of ties (particularly A,
0.499 against a budget of 1). The combined tie-breaker improves this, but does not remove repeated
scores or the poor episode recall. Recency alone has the best B1 episode recall, still below random.
The primary variant was fixed before evaluation and is not changed after seeing these numbers.

Sensitivity without turbine 2304515 reaches the same practical conclusion:

| Target | Primary B1 PR-AUC | Primary B1 recall | Random recall | Evaluable episodes |
|---|---:|---:|---:|---:|
| A | 0.531 | 20.5% | 77.4% | 195 |
| B | 0.516 | 16.9% | 74.6% | 177 |

Full sensitivity results: [A](tables/b1_a_excl_2304515.csv),
[B](tables/b1_b_excl_2304515.csv). Thresholds are reselected on the reduced validation set, as in
the shared runner. This is a sensitivity analysis, not a replacement main result.

Missing-history fallbacks apply on 301 of 14,954 validation feature days. Among metric rows, A has
none missing, while B has 65 of 8,827 days missing both inputs. These B days are scored at zero;
no rows are silently dropped. See [coverage](tables/b1_history_coverage.csv) and
[sensitivity coverage](tables/b1_history_coverage_excl_2304515.csv).

## Fixed rules (D-050)

| Registered name | Score | Role |
|---|---|---|
| `B1_recent_fault` | `fault_days_30 + 0.5 / (1 + days_since_fault_day)` | Primary |
| `B1_fault_count` | `fault_days_30` | Single-column comparison |
| `B1_fault_recency` | `1 / (1 + days_since_fault_day)` | Single-column comparison |

The combined rule ranks count first, then recency. A more recent fault cannot outweigh an extra
fault day. The weight is fixed in `configs/protocol.yaml`, before inspecting validation results.
Missing recency contributes zero (no known previous fault); missing count also contributes zero.
Unknown history is not evidence of health. Coverage files report how often these fallbacks apply.
Matching count and recency can still produce ties. Alerts use the existing strict `score > threshold`
rule: identical scores are not broken with random noise, and a zero-alert outcome is reported honestly.

## Reproduce

Run from the repository root with the project installed in a Python environment. With the existing
prepared daily features and A/B label/episode parquet files under `data/interim/`:

```powershell
python scripts/run_b1_recent_fault.py
python scripts/run_b1_recent_fault.py --exclude 2304515
```

An alternate prepared-data root can be supplied with `--interim PATH`. The script calls the shared
runner, includes the constant and deterministic random floors, fits on labelled training rows,
scores all validation feature days, chooses the threshold on labelled validation rows, and computes
metrics only on labelled rows. It never scores test years. Confidence intervals use the configured
1,000 turbine bootstrap draws. The sensitivity run excludes turbine 2304515 throughout.

Outputs:

- `reports/tables/b1_a.csv` and `b1_b.csv`: validation metrics, intervals and random comparisons.
- `reports/tables/b1_history_coverage.csv`: missing history on all and labelled validation days.
- Files with `_excl_2304515` suffixes: separate sensitivity results.
- `data/interim/scores/`: daily validation scores, including days excluded from metrics.

To rebuild the prepared data from the original year folders, point `$rawRoot` at their parent:

```powershell
$rawRoot = "../Cyrus Chow's files - DATA"
python scripts/build_scada_parquet.py --raw $rawRoot --workers 2
python scripts/build_alarm_table.py --raw $rawRoot
python scripts/build_clean_data.py --raw $rawRoot
python scripts/build_labels.py
python scripts/build_features.py
python scripts/check_feature_leakage.py
```

SCADA conversion defaults to protocol years, keeping 2019 unused even though this supplied source
has 2019 sensor files. Use a fresh interim directory if an earlier experiment already converted 2019:
the conversion script does not delete existing files. The daily-summary loader retains its existing
history semantics. No source CSV is moved, changed or committed.

## Verification

`pytest` and `ruff check .` cover score ordering, missing inputs, train-only fitting, label/status and
batch independence, registry integration, and a known-answer episode evaluation including tied
scores. The feature builder and its look-ahead check are unchanged.

Read episode recall beside the random floor: many days can issue an alert in a 30-day warning
window, so high recall alone is not evidence of predictive value. PR-AUC, lead time, false alarms,
episode counts and uncertainty intervals are all included in the CSVs.

Completed checks: 89 tests passed; Ruff and diff whitespace checks passed. The real-data look-ahead
check passed on 20 seeded turbine-days across four turbines. Rebuilt cleaning, label and feature
summary CSVs are identical to the committed versions. Runtime: Python 3.12.14, pandas 3.0.6,
NumPy 2.5.4, scikit-learn 1.9.1; seed 42. To reload a threshold from CSV for an exact strict-boundary
comparison, use `pd.read_csv(path, float_precision="round_trip")`.
