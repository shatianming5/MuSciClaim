# configs/

YAML configs for models and runs. These are the primary knobs for reproducible evaluations.

## Contents

| Path | Role | Why it exists |
| --- | --- | --- |
| `configs/README.md` | Config index | Explains config files and reproducibility conventions |
| `configs/models.example.yaml` | Example model configuration | Template you copy to `configs/models.yaml` |
| `configs/run.example.yaml` | Example run configuration | Template you copy to `configs/run.yaml` |
| `configs/models.yaml` | Your model IDs and revisions (local) | Keeps model provenance explicit and reproducible (gitignored by default) |
| `configs/run.yaml` | Your run matrix and IO settings (local) | Controls dataset revision, preprocessing, and evaluation matrix (gitignored by default) |

## How To Use

1. Create local config files (these are gitignored by default):

```bash
cp configs/models.example.yaml configs/models.yaml
cp configs/run.example.yaml configs/run.yaml
```

2. Edit `configs/models.yaml` to point at real models.

Fields (per model key like `ours` / `base`):
- `adapter`: `dummy` | `hf_text` | `hf_vlm`
- `repo_id`: Hugging Face model ID (required for `hf_text`/`hf_vlm`)
- `revision`: tag or commit SHA (strongly recommended for reproducibility)
- `modality`: `text` | `vlm` (must match the adapter)
- `device`: e.g. `cpu`, `cuda`, `mps`
- `dtype`: `float32` | `float16` | `bfloat16` (depends on your hardware/model)

Examples:

Dummy (no external model downloads):
```yaml
ours:
  adapter: dummy
  repo_id: null
  revision: null
  modality: text
  device: cpu
  dtype: float32
```

Text-only Transformers model:
```yaml
ours:
  adapter: hf_text
  repo_id: "YOUR_TEXT_MODEL_ID"
  revision: "YOUR_SHA_OR_TAG"
  modality: text
  device: cuda
  dtype: bfloat16
```

Multimodal (VLM) Transformers model:
```yaml
ours:
  adapter: hf_vlm
  repo_id: "YOUR_VLM_MODEL_ID"
  revision: "YOUR_SHA_OR_TAG"
  modality: vlm
  device: cuda
  dtype: float16
```

3. Edit `configs/run.yaml` to match your model modality and desired diagnostics.

Important keys:
- `dataset.revision`: pin a dataset SHA if you need strict reproducibility.
- `preprocessing.resize_max_side`: set if your VLM cannot ingest large figures.
- `matrix.conditions`:
  - Text-only track: `[c_only, claim_only]`
  - VLM track: `[full, c_only, f_only, claim_only]`
- `matrix.prompt_modes`: typically `[D, R]` (decision-only vs short reasoning).
- `matrix.include_panels_run`: set `true` for VLM localization (`PANELS` mode).
- `matrix.reproducibility_repeats`: set `> 1` to write `runs/<run_id>/repro_report.json`.

Example: text-only track (no images):
```yaml
matrix:
  conditions: [c_only, claim_only]
  prompt_modes: [D, R]
  include_panels_run: false
  reproducibility_repeats: 2
```

Example: VLM track (ablations + localization):
```yaml
matrix:
  conditions: [full, c_only, f_only, claim_only]
  prompt_modes: [D, R]
  include_panels_run: true
  reproducibility_repeats: 2
```

## Common Pitfalls

- If you set `adapter: hf_vlm`, you must set `modality: vlm`. Otherwise image conditions will be skipped.
- If you get OOMs: reduce `preprocessing.resize_max_side`, reduce `max_new_tokens_*`, or use a smaller model.
