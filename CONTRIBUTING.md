# Contributing

1. Never push to `main`. All changes go through a pull request.
2. Make one branch per task, named `owner/short-topic` (e.g. `kanav/lgbm-baseline`).
3. `git pull` on `main`, then `git switch -c owner/short-topic`.
4. Keep PRs small (one task, ideally under ~300 changed lines).
5. Reusable logic goes in `src/wt_predmaint/`; notebooks are for exploration (`NN_owner_topic.ipynb`).
6. Run `pytest` and `ruff check .` before pushing; CI runs the same.
7. Open a PR with the template filled in and request one reviewer.
8. Reviewer approves or comments within ~2 days; author merges after approval.
9. Never commit data, credentials or notebook outputs.
10. Log non-obvious choices in `docs/DECISIONS.md`; log dataset surprises in `docs/DATA_TRAPS.md`.
