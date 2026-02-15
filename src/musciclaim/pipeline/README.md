# musciclaim/pipeline/

The evaluation runner: run matrix expansion, inference loop, artifact writing, and reproducibility checks.

## Contents

| Path | Role | Why it exists |
| --- | --- | --- |
| `src/musciclaim/pipeline/README.md` | Pipeline index | Documents runner components and artifact writing |
| `src/musciclaim/pipeline/__init__.py` | Package marker | Keeps the pipeline modules importable as a package |
| `src/musciclaim/pipeline/matrix.py` | Run matrix builder | Turns configs into explicit runs (model x condition x prompt mode) |
| `src/musciclaim/pipeline/runner.py` | Main runner | Executes runs and writes `predictions.jsonl` |
| `src/musciclaim/pipeline/metadata.py` | Run metadata capture | Records environment and provenance for auditability |

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
