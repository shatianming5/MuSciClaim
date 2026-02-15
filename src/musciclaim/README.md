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
