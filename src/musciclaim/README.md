# musciclaim/

Core library for loading MuSciClaims, running model inference, enforcing strict JSON outputs, and computing metrics.

## Contents

| Path | Role | Why it exists |
| --- | --- | --- |
| `src/musciclaim/README.md` | Package index | Explains modules and how they fit together |
| `src/musciclaim/__init__.py` | Package entrypoint | Exposes version and keeps imports stable |
| `src/musciclaim/config/` | **Config package** | Typed dataclasses + YAML loaders for models and run settings |
| `src/musciclaim/schema/` | **Schema package** | Core enums (`Decision`, `Condition`, `PromptMode`) + dataclasses (`PredictionRecord`, etc.) |
| `src/musciclaim/parse.py` | Strict JSON parser | Enforces machine-checkable outputs with clear failure modes |
| `src/musciclaim/panels.py` | Panel normalization | Makes localization evaluation robust to formatting noise |
| `src/musciclaim/cli/` | CLI entrypoints | Runs the pipeline and aggregation from the terminal |
| `src/musciclaim/data/` | Dataset + images | Loads MuSciClaims and handles figure retrieval/preprocessing |
| `src/musciclaim/models/` | Model adapters | Abstracts HF text/VLM models behind one interface; shared `torch_utils` |
| `src/musciclaim/pipeline/` | Runner + helpers | Evaluation loop, reproducibility checks, error slices |
| `src/musciclaim/metrics/` | Metrics + aggregation | Classification, synergy, flip-rate, localization, discovery, reports |
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
