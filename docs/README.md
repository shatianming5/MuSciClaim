# docs/

Documentation for the project: plan, experiment definitions, verification logs, and any design notes.

## Contents

| Path | Role | Why it exists |
| --- | --- | --- |
| `docs/README.md` | Docs index | Explains what each document is for |
| `docs/plan.md` | Canonical evaluation/implementation plan | Single source of truth for what the repo is building and why |
| `docs/experiment.md` | Experiment matrix | Runnable commands and expected artifacts for common tracks |

## How To Use

Recommended reading order:
1. `docs/plan.md`: explains *what* is being validated and *why* (capability axioms, required ablations).
2. `docs/experiment.md`: translates the plan into runnable experiments (`musciclaim-eval` commands) and
   expected artifact locations.

If you are integrating a new model:
- Update `configs/models.yaml` with the new checkpoint and a pinned `revision`.
- Run the closest experiment in `docs/experiment.md` (text-only or VLM).
- Use `scores/<run_id>/report.md` as the fastest “single-page” summary.
