## Completed validation run

The notebook was executed on the current main-project cached features and labels. All fitted-model results below use the main false-alert budget and evaluator. No test-year scores were produced.

| Target | Model | AP | 95% turbine-bootstrap AP interval | Episode recall | False-alert days / turbine-month |
|---|---|---:|---|---:|---:|
| A | B0a_constant | 0.4704 | [0.4108, 0.5378] | 0.000 | 0.000 |
| A | B0b_random | 0.4649 | [0.4045, 0.5348] | 0.780 | 0.999 |
| A | B4_logistic | 0.4466 | [0.3769, 0.5523] | 0.234 | 0.999 |
| A | L1_lgbm | 0.4495 | [0.3918, 0.5400] | 0.139 | 0.999 |
| A | X_RF | 0.4105 | [0.3427, 0.5065] | 0.206 | 0.999 |
| A | X_ExtraTrees | 0.4141 | [0.3470, 0.5069] | 0.258 | 0.999 |
| A | X_HistGB | 0.4288 | [0.3703, 0.5188] | 0.120 | 0.999 |
| B | B0a_constant | 0.4924 | [0.4293, 0.5625] | 0.000 | 0.000 |
| B | B0b_random | 0.4830 | [0.4188, 0.5575] | 0.760 | 1.000 |
| B | B4_logistic | 0.4495 | [0.3717, 0.5325] | 0.245 | 1.000 |
| B | L1_lgbm | 0.4517 | [0.3785, 0.5364] | 0.240 | 1.000 |
| B | X_RF | 0.4509 | [0.3825, 0.5252] | 0.234 | 1.000 |
| B | X_ExtraTrees | 0.4662 | [0.4011, 0.5364] | 0.344 | 1.000 |
| B | X_HistGB | 0.4322 | [0.3644, 0.5154] | 0.161 | 1.000 |

For these fixed settings, none of the three additional models exceeds the paired random-score baseline in AP point estimate. This is a development finding, not a claim about every possible setting of these model families. Do not tune on the previously inspected test years to reverse this finding.

[Full validation metrics](validation_metrics.csv) and [supplementary window metrics](validation_window_metrics.csv).
