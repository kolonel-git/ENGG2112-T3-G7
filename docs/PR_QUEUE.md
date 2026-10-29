# Pull request queue, in merge order

Last updated 2026-10-29. `gh` is not installed, so each PR is opened in the browser from the compare
link; nobody pushes to `main`. Repository: `kolonel-git/ENGG2112-T3-G7`. `origin/main` was at
`57b0d7b` when this queue was written.

## What happened to the earlier queue

`origin/main` had committed unresolved merge-conflict markers (Parvez's merges of 2026-10-28 and
10-29), so it did not load or pass its tests: `configs/protocol.yaml`, `models/healthy.py` and
`tests/test_healthy.py` were broken, the baseline tables were overwritten with three rows, and
`scripts/build_scada_parquet.py` used an undefined `ROOT`. Kanav's three stacked branches
(`kanav/nbm`, `kanav/exploratory-models`, `kanav/label-sensitivity`, 29 commits) are now merged with
`origin/main` in one branch, `kanav/sync-main`, which also repairs those problems. **Do not open the
three old PRs.** `kanav/sync-main` replaces them.

## To merge, in this order

| # | Branch | Into | State | Notes |
|---|--------|------|-------|-------|
| 1 | `kanav/sync-main` | `main` | Pushed; open it first | Everything of Kanav's (NBM, LightGBM, the single test run, SHAP and robustness, four exploratory rounds D-064 to D-068, the restructured report, figures F36 to F39) merged with `origin/main`. Repairs the conflict markers on `main`, the duplicate `healthy:` block, the baseline tables and the `ROOT` bug, and brings `ruff` back to clean. Merge with a **merge commit**, not squash. CI must be green. |
| 2 | `farhan-evaluation-cost-review` | `main` | Needs an update by Farhan | Merge the new `main` first. Its **D-055 collides** with ours: renumber to D-069 (next free ID after D-068). Resolve README, `docs/project/DECISIONS.md` and `docs/project/roadmap.md`, rerun `scripts/check_evaluation_results.py` and regenerate the evaluation section so it includes the LightGBM, R2 and test results. |
| 3 | `revael/features-labels-report` | `main` | Needs an update by Revael | Docs only. Merge the new `main`; trivial conflicts in README and `docs/project/roadmap.md` (the docs were moved to `docs/project/`). |
| 4 | `revael/simple-alarm-forecast` | `main` | Review first | Separate experiment (`src/wt_predmaint/experiments/`, three commits of 2026-10-29), based on the broken `main`. Ask Revael to merge the new `main`, run `ruff check .` and `pytest`, and say whether it belongs in the report. It does not change any of our results. |

## Safeguards for `main`

- GitHub, Settings, Branches: require a pull request and a passing CI run before merging, and block
  direct pushes (CI already runs `ruff check .` and `pytest`, but nothing stopped the push).
- Before merging anything, run `git grep -n -E "^(<<<<<<<|>>>>>>>) "` and `ruff check .` locally.
- Scripts that write tables (for example `scripts/run_dummy_baselines.py`) must not overwrite the full
  `reports/tables/baselines_*.csv` files; write to their own file.

## Do not merge

`Parvez/main`, `main_P` and `Data_Formatting_P` (older copies; their useful content is already in
`main`), `revael/p0-b4-baseline` and `Farhan` (already in `main`), and `test`. Delete them after the
team agrees.

## Not yet created (needed before submission)

- **Parvez:** cleaning-layer sign-off; dataset version (D-004); repository licence and citation text
  (README TODOs); try to fetch the AeroUp dates and data dictionary from Zenodo.
- **Cyrus:** ethics section and report integration. The red items in `reports/report/main.tex` list
  what still needs confirming.
- **Final report integration PR:** after the sections above arrive, one PR that merges them into
  `reports/report/main.tex`.

## Compare links

- #1: https://github.com/kolonel-git/ENGG2112-T3-G7/compare/main...kanav/sync-main?expand=1
- #2: https://github.com/kolonel-git/ENGG2112-T3-G7/compare/main...farhan-evaluation-cost-review?expand=1
- #3: https://github.com/kolonel-git/ENGG2112-T3-G7/compare/main...revael/features-labels-report?expand=1
- #4: https://github.com/kolonel-git/ENGG2112-T3-G7/compare/main...revael/simple-alarm-forecast?expand=1

## Farhan update, 2026-10-29

Main `6535030` includes #19 and #17. PR #18 now merges that main, preserves
Kanav's D-055 and records Farhan's review as D-069. Evaluation/cost and F27
include the frozen models and test export. Published-table checks pass; original
score reproduction remains pending Kanav's git-ignored artefacts. See
`docs/EVALUATION_REVIEW.md`. The older queue states above are historical.
