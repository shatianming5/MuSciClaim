# musciclaim/models/

Model adapters behind a small interface so the pipeline is model-agnostic.

## Contents

| Path | Role | Why it exists |
| --- | --- | --- |
| `src/musciclaim/models/README.md` | Models index | Documents adapters and how to add new ones |
| `src/musciclaim/models/__init__.py` | Package marker | Keeps the model modules importable as a package |
| `src/musciclaim/models/base.py` | Adapter protocol | Single interface for text-only and VLM models |
| `src/musciclaim/models/dummy.py` | Deterministic adapter | Enables smoke tests without GPUs or external downloads |
| `src/musciclaim/models/hf_text.py` | HF text adapter | Runs text-only transformers models |
| `src/musciclaim/models/hf_vlm.py` | HF VLM adapter | Runs multimodal transformers models (image + text) |
| `src/musciclaim/models/factory.py` | Adapter factory | Central place to map config -> adapter instance |
