# musciclaim/

Core library for loading MuSciClaims, running model inference, enforcing strict JSON outputs, and computing metrics.

## Contents

| Path | Role | Why it exists |
| --- | --- | --- |
| `src/musciclaim/README.md` | Package index | Explains modules and how they fit together |
| `src/musciclaim/__init__.py` | Package entrypoint | Exposes version and keeps imports stable |
| `src/musciclaim/config.py` | YAML config loader | Makes runs reproducible and explicit |
| `src/musciclaim/schema.py` | Core enums + dataclasses | Defines stable contracts (conditions, prompt modes, records) |
| `src/musciclaim/parse.py` | Strict JSON parser | Enforces machine-checkable outputs with clear failure modes |
| `src/musciclaim/panels.py` | Panel normalization | Makes localization evaluation robust to formatting noise |
| `src/musciclaim/cli/` | CLI entrypoints | Runs the pipeline and aggregation from the terminal |
| `src/musciclaim/data/` | Dataset + images | Loads MuSciClaims and handles figure retrieval/preprocessing |
| `src/musciclaim/models/` | Model adapters | Abstracts HF text/VLM models behind one interface |
| `src/musciclaim/pipeline/` | Runner + metadata | Executes runs and writes auditable artifacts |
| `src/musciclaim/metrics/` | Metrics + aggregation | Computes Macro-F1, synergy, flip-rate, and localization |
| `src/musciclaim/prompts/` | Prompt templates | Central place for strict JSON schemas and retry prompts |

## How To Use

Most usage should go through the CLI (`musciclaim-eval`), but the core runner is also callable from
Python for integration tests or notebooks.

Programmatic example:

```python
from pathlib import Path

from musciclaim.config import load_models_config, load_run_config
from musciclaim.pipeline.runner import run_evaluation

run_cfg = load_run_config("configs/run.yaml")
models_cfg = load_models_config("configs/models.yaml")

run_id = run_evaluation(
    run_cfg=run_cfg,
    models_cfg=models_cfg,
    limit=10,
    run_id="example_run",
    repo_root=Path("."),
)

print("run_id:", run_id)
```
