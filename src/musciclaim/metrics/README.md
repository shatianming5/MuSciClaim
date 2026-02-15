# musciclaim/metrics/

Metrics for A1–A4: classification, localization, synergy, support bias, and flip-rate sensitivity.

## Contents

| Path | Role | Why it exists |
| --- | --- | --- |
| `src/musciclaim/metrics/README.md` | Metrics index | Documents metric modules and summary outputs |
| `src/musciclaim/metrics/__init__.py` | Package marker | Keeps the metrics modules importable as a package |
| `src/musciclaim/metrics/bootstrap.py` | Paired bootstrap utilities | Computes paired confidence intervals for metric differences |
| `src/musciclaim/metrics/classification.py` | Macro-F1 / per-class / confusion | Core task metrics |
| `src/musciclaim/metrics/localization.py` | Set-F1 for panels | Quantifies evidence localization |
| `src/musciclaim/metrics/sensitivity.py` | Flip-rate by base_claim_id | Tests epistemic sensitivity |
| `src/musciclaim/metrics/diagnostics.py` | Bias + synergy | Diagnoses cross-modal necessity and support bias |
| `src/musciclaim/metrics/significance.py` | Ours vs Base significance | Produces paired bootstrap CIs for Macro-F1 and CONTRADICT-F1 |
| `src/musciclaim/metrics/leakage.py` | Leakage audit helpers | Optional overlap checks using paper_id lists |
| `src/musciclaim/metrics/aggregate.py` | Summary writer | Produces `summary.csv` from predictions |

## How To Use

### Recompute metrics from an existing run

If you already have `runs/<run_id>/**/predictions.jsonl`, you can regenerate score artifacts without
re-running inference:

```bash
musciclaim-metrics --run-id exp_003_vlm
```

This writes:
- `scores/<run_id>/summary.csv`
- `scores/<run_id>/confusions.json`
- `scores/<run_id>/report.md`
- `scores/<run_id>/significance.json` (only if both `ours` and `base` exist)

### Programmatic aggregation

```python
from pathlib import Path
from musciclaim.metrics.aggregate import aggregate_run

summary_csv = aggregate_run(
    run_id="exp_003_vlm",
    runs_root=Path("runs"),
    scores_root=Path("scores"),
)
print("wrote:", summary_csv)
```
