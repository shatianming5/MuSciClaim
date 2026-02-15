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

## How To Use

### Choosing an adapter

In `configs/models.yaml`, set `adapter` and `modality` consistently:
- `adapter: dummy` + `modality: text` for smoke tests
- `adapter: hf_text` + `modality: text` for text-only Transformers models
- `adapter: hf_vlm` + `modality: vlm` for multimodal Transformers models

The runner uses `modality` to decide which input conditions are valid (e.g., it skips `full`/`f_only`
for text-only models).

### Adding a new adapter

1. Create a new module under `src/musciclaim/models/` (e.g., `my_provider.py`).
2. Implement the `ModelAdapter` interface from `src/musciclaim/models/base.py`:
   - `info` (provenance)
   - `supports_images`
   - `generate(...) -> GenerationResult`
3. Register it in `src/musciclaim/models/factory.py` (map a new `adapter` string to your class).
4. Update `configs/models.example.yaml` with a commented template, so usage is obvious.

Design guideline:
- Keep adapters thin. Anything evaluation-specific should live in the runner/metrics, not in the
  adapter.
