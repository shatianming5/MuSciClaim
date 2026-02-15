# Experiment Spec (MuSciClaims)

This document is a runnable experiment matrix for `musciclaim-eval`.

## Conventions

- Conditions:
  - `full`: figure + caption + claim (VLM only)
  - `c_only`: caption + claim
  - `f_only`: figure + claim (VLM only)
  - `claim_only`: claim only
- Prompt modes:
  - `D`: decision-only JSON
  - `R`: short reasoning + decision JSON
  - `PANELS`: panels + reasoning + decision JSON (localization)

## Exp-001: Pipeline Smoke (Dummy Models)

Purpose:
- Verify the pipeline runs end-to-end without GPUs.

Command:
```bash
musciclaim-eval \
  --run-config configs/run.yaml \
  --models-config configs/models.yaml \
  --limit 10 \
  --run-id exp_001_smoke
```

Expected artifacts:
- `runs/exp_001_smoke/run_metadata.json`
- `runs/exp_001_smoke/repro_report.json` (if repeats enabled)
- `runs/exp_001_smoke/<model>/<condition>/<prompt_mode>/predictions.jsonl`
- `scores/exp_001_smoke/summary.csv`
- `scores/exp_001_smoke/confusions.json`
- `scores/exp_001_smoke/significance.json` (if `ours` and `base` are both configured)

## Exp-002: Text-Only Track (Ours vs Base)

Purpose:
- Run the text-only evaluation track (no images).
- Produce ablation evidence: `c_only` vs `claim_only`.

Setup:
- In `configs/models.yaml`, set both `ours` and `base` to `adapter: hf_text`.
- Ensure `configs/run.yaml` includes `conditions: [c_only, claim_only]`.

Command:
```bash
musciclaim-eval \
  --run-config configs/run.yaml \
  --models-config configs/models.yaml \
  --run-id exp_002_text
```

Primary outputs:
- `scores/exp_002_text/summary.csv`
- `scores/exp_002_text/significance.json`

## Exp-003: VLM Track (Ours vs Base)

Purpose:
- Run the full multimodal evaluation with ablations and localization.

Setup:
- In `configs/models.yaml`, set both `ours` and `base` to `adapter: hf_vlm` and `modality: vlm`.
- Ensure `configs/run.yaml` includes `conditions: [full, c_only, f_only, claim_only]`.
- Ensure `include_panels_run: true`.

Command:
```bash
musciclaim-eval \
  --run-config configs/run.yaml \
  --models-config configs/models.yaml \
  --run-id exp_003_vlm
```

Primary outputs:
- `scores/exp_003_vlm/summary.csv`
- `scores/exp_003_vlm/significance.json`
- Localization: `localization_f1` columns and `scores/exp_003_vlm/confusions.json`

## Optional: Leakage Audit (paper_id overlap)

If you have a newline-delimited list of training `paper_id` values, you can compute overlap metrics:

```bash
musciclaim-metrics \
  --run-id exp_003_vlm \
  --training-paper-ids-file path/to/training_paper_ids.txt
```

This produces:
- `scores/<run_id>/leakage_audit.json`

## Exp-004: Full Text-Only Example (Concrete Models, Not Smoke)

Purpose:
- Run a full (no `--limit`) text-only evaluation on the full MuSciClaims test split.

Setup:
- Install HF extras: `pip install -e '.[hf]'`
- Configure `configs/models.yaml`:

```yaml
ours:
  adapter: hf_text
  repo_id: "HuggingFaceTB/SmolLM2-1.7B-Instruct"
  revision: "31b70e2e869a7173562077fd711b654946d38674"
  modality: text
  device: cpu
  dtype: float32

base:
  adapter: hf_text
  repo_id: "HuggingFaceTB/SmolLM2-360M-Instruct"
  revision: "a10cc1512eabd3dde888204e902eca88bddb4951"
  modality: text
  device: cpu
  dtype: float32
```

- Configure `configs/run.yaml` matrix:

```yaml
matrix:
  conditions: [c_only, claim_only]
  prompt_modes: [D, R]
  include_panels_run: false
  reproducibility_repeats: 2
```

Command:
```bash
musciclaim-eval \
  --run-config configs/run.yaml \
  --models-config configs/models.yaml \
  --run-id exp_004_full_text_smollm2
```

Expected artifacts:
- `runs/exp_004_full_text_smollm2/run_metadata.json`
- `runs/exp_004_full_text_smollm2/<model>/<condition>/<prompt_mode>/predictions.jsonl`
- `scores/exp_004_full_text_smollm2/summary.csv`
- `scores/exp_004_full_text_smollm2/report.md`
