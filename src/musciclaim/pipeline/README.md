# musciclaim/pipeline/

The evaluation runner: run matrix expansion, inference loop, artifact writing, and reproducibility checks.

## Contents

| Path | Role | Why it exists |
| --- | --- | --- |
| `pipeline/README.md` | Pipeline index | Documents runner components and artifact writing |
| `pipeline/__init__.py` | Package marker | Keeps the pipeline modules importable as a package |
| `pipeline/runner.py` | Main runner | Orchestrates the evaluation loop and writes `predictions.jsonl` |
| `pipeline/matrix.py` | Run matrix builder | Turns configs into explicit runs (model x condition x prompt mode) |
| `pipeline/metadata.py` | Run metadata capture | Records environment and provenance for auditability |
| `pipeline/helpers.py` | Runner utilities | `utc_run_id`, `condition_flags`, `max_new_tokens`, JSONL writer |
| `pipeline/repro.py` | Reproducibility checks | Compares repeated prediction files for determinism audits (A5) |
| `pipeline/error_slices.py` | Error slice reports | Writes high-risk misclassification slices for human review |

## How To Use

You typically use the pipeline via the CLI:

```bash
musciclaim-eval --run-config configs/run.yaml --models-config configs/models.yaml
```

Key outputs written by the runner:
- `runs/<run_id>/run_metadata.json`: environment + config provenance
- `runs/<run_id>/<model>/<condition>/<prompt_mode>/predictions.jsonl`: primary prediction artifacts
- `runs/<run_id>/repro_report.json`: determinism comparisons (only if repeats enabled)

Design notes:
- The run matrix is expanded in `src/musciclaim/pipeline/matrix.py`.
- The runner enforces strict JSON schemas and performs one format-fix retry on invalid outputs.
