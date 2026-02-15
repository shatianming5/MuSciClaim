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
