# MuSciClaim

Reproducible evaluation and diagnostics for **MuSciClaims**: can a model truly audit scientific claims by **seeing figures + reading captions**, rather than guessing from text?

This repo implements the end-to-end pipeline described in `docs/plan.md`:
- 3-way claim verification: `SUPPORT` / `CONTRADICT` / `NEUTRAL`
- Input ablations to prove cross-modal necessity (Full, C-only, F-only, Claim-only)
- Evidence localization via panel prediction
- Epistemic sensitivity via base-claim flips (SUPPORT -> CONTRADICT)
- Determinism / auditability via strict JSON outputs and reproducible artifacts

## Contents

| Path | Role | Why it exists |
| --- | --- | --- |
| `README.md` | Project entrypoint | First-stop documentation and the end-to-end workflow diagram |
| `pyproject.toml` | Python packaging | Dependencies, CLI entrypoints, and tooling configuration |
| `LICENSE` | License | Makes redistribution/usage terms explicit |
| `.gitignore` | Git hygiene | Keeps outputs and local environment files out of version control |
| `.editorconfig` | Editor hygiene | Enforces consistent whitespace/newline rules across editors |
| `plan.md` | Pointer doc | Links to the canonical plan in `docs/plan.md` |
| `docs/` | Documentation | Canonical plan and future experiment/verification docs |
| `configs/` | YAML configs | Explicit, reproducible model and run settings |
| `src/` | Source root | Python package source tree (uses the `src/` layout) |
| `src/musciclaim/` | Core library | Keeps evaluation logic reusable and testable |
| `scripts/` | Dev utilities | Lightweight checks and maintenance helpers |
| `tests/` | Unit tests | Guards strict parsing/metrics contracts |
| `runs/` | Raw artifacts | Per-run JSONL predictions and metadata (gitignored) |
| `scores/` | Aggregates | Summary tables derived from `runs/` (gitignored) |
| `analysis/` | Error slices | Human-readable failure cases (gitignored) |

## Workflow

```mermaid
flowchart TD
  A["Load HF dataset + lock revision"] --> B["Preflight checks"]
  B --> C["Download/cache figures"]
  C --> D["Build run matrix: Full/C-only/F-only/Claim-only + prompt modes"]
  D --> E["Model inference via adapter: VLM or text-only"]
  E --> F["Strict JSON parse + one format-fix retry"]
  F --> G["Write predictions.jsonl + run_metadata.json"]
  G --> H["Compute metrics: Macro-F1, bias, synergy, flip-rate, localization"]
  H --> I["Write summary.csv + error slices"]
```

## Quickstart

1. Create a venv and install:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev,hf]'
```

2. Copy example configs and fill model IDs:

```bash
cp configs/models.example.yaml configs/models.yaml
cp configs/run.example.yaml configs/run.yaml
```

3. Run a small smoke eval (limits to 10 examples):

```bash
musciclaim-eval --run-config configs/run.yaml --models-config configs/models.yaml --limit 10
```

4. Inspect artifacts:
- `runs/<run_id>/<model>/<condition>/<prompt_mode>/predictions.jsonl`
- `runs/<run_id>/run_metadata.json`
- `runs/<run_id>/repro_report.json` (if reproducibility repeats are enabled)
- `scores/<run_id>/summary.csv`

## Configure Models

MuSciClaim is model-agnostic via adapters. Edit `configs/models.yaml` and choose an adapter per model
key (typically `ours` and optionally `base`).

### Dummy (smoke testing)

```yaml
ours:
  adapter: dummy
  repo_id: null
  revision: null
  modality: text
  device: cpu
  dtype: float32
```

### Text-only Transformers model

```yaml
ours:
  adapter: hf_text
  repo_id: "YOUR_TEXT_MODEL_ID"
  revision: "YOUR_SHA_OR_TAG"
  modality: text
  device: cuda
  dtype: bfloat16
```

### Multimodal (VLM) Transformers model

```yaml
ours:
  adapter: hf_vlm
  repo_id: "YOUR_VLM_MODEL_ID"
  revision: "YOUR_SHA_OR_TAG"
  modality: vlm
  device: cuda
  dtype: float16
```

If you only want to evaluate one model, you can omit the `base:` block (paired significance will
then be skipped).

## Pick The Right Run Matrix

Edit `configs/run.yaml`:
- For text-only models, use `matrix.conditions: [c_only, claim_only]`.
- For VLM models, use `matrix.conditions: [full, c_only, f_only, claim_only]` and keep
  `matrix.include_panels_run: true` for localization.

## Common Commands

Run end-to-end evaluation:
```bash
musciclaim-eval --run-config configs/run.yaml --models-config configs/models.yaml
```

Recompute metrics from an existing run directory:
```bash
musciclaim-metrics --run-id <run_id>
```

Run documentation hygiene checks:
```bash
musciclaim-check-readmes
musciclaim-check-docstrings
```

## Example: Full Text-Only Run (Not Smoke)

This is a concrete, non-smoke example that runs on the full MuSciClaims test split (1,515 examples)
using real Hugging Face models (text-only track: `c_only` + `claim_only`).

1. Put this into `configs/models.yaml` (pinned revisions for reproducibility):

```yaml
ours:
  adapter: hf_text
  repo_id: "HuggingFaceTB/SmolLM2-1.7B-Instruct"
  revision: "31b70e2e869a7173562077fd711b654946d38674"
  modality: text
  device: cpu  # change to cuda/mps if available
  dtype: float32  # on GPU: bfloat16/float16 is typically faster

base:
  adapter: hf_text
  repo_id: "HuggingFaceTB/SmolLM2-360M-Instruct"
  revision: "a10cc1512eabd3dde888204e902eca88bddb4951"
  modality: text
  device: cpu
  dtype: float32
```

2. Ensure `configs/run.yaml` uses a text-only matrix:

```yaml
matrix:
  conditions: [c_only, claim_only]
  prompt_modes: [D, R]
  include_panels_run: false
  reproducibility_repeats: 2
```

3. Run the full evaluation (no `--limit`):

```bash
musciclaim-eval \
  --run-config configs/run.yaml \
  --models-config configs/models.yaml \
  --run-id exp_full_text_smollm2
```

4. Read results:
- `scores/exp_full_text_smollm2/report.md`
- `scores/exp_full_text_smollm2/summary.csv`

## Artifact Contract

### Predictions JSONL
Each line is a single example prediction record. Fields include:
- IDs: `claim_id`, `base_claim_id`
- Gold: `label_gold`, `panels_gold`
- Pred: `label_pred`, `panels_pred`, `reasoning` (optional)
- Flags: `invalid_output`, `invalid_input_image`, `invalid_panels`, `truncated`
- Perf: `latency_ms`, `tokens_in`, `tokens_out`
- Image meta: `orig_w`, `orig_h`, `new_w`, `new_h`, `resize_ratio`, `filepath`
- Provenance: `model_id`, `model_revision`, `dataset_revision`, `run_id`

### Summary CSV
`Macro-F1`, per-class P/R/F1, high-risk confusion counts, support bias, synergy, flip-rate, localization set-F1, and invalid rates.

### Additional Score Artifacts
- `scores/<run_id>/confusions.json`: confusion matrices keyed by `model/prompt_mode/condition`.
- `scores/<run_id>/report.md`: one-page summary table with key metrics and significance (if available).
- `scores/<run_id>/significance.json`: paired bootstrap CIs for Ours-vs-Base (if both are present).
- `scores/<run_id>/leakage_audit.json`: overlap audit (only if you pass a training `paper_id` list).

## Documentation
- Implementation/evaluation plan: `docs/plan.md`
- Folder-level docs: each directory contains a `README.md` describing purpose + contents.
