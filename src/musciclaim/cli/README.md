# musciclaim/cli/

Command-line entrypoints that wire configs to the library pipeline.

## Contents

| Path | Role | Why it exists |
| --- | --- | --- |
| `src/musciclaim/cli/README.md` | CLI index | Documents available commands and their roles |
| `src/musciclaim/cli/__init__.py` | Package marker | Keeps the CLI modules importable as a package |
| `src/musciclaim/cli/eval.py` | `musciclaim-eval` CLI | Runs inference + metrics and writes reproducible artifacts |
| `src/musciclaim/cli/metrics.py` | `musciclaim-metrics` CLI | Recomputes metrics from existing `predictions.jsonl` |
| `src/musciclaim/cli/check_readmes.py` | `musciclaim-check-readmes` CLI | Enforces documentation hygiene |
| `src/musciclaim/cli/check_docstrings.py` | `musciclaim-check-docstrings` CLI | Enforces "What/Why" docstring markers for public classes/methods |

## How To Use

First install the package (editable install recommended):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev,hf]'
```

### `musciclaim-eval`

Runs the full end-to-end pipeline (dataset load, image download/cache, inference, strict parsing,
artifact writing, and aggregation).

```bash
musciclaim-eval --run-config configs/run.yaml --models-config configs/models.yaml
```

Common flags:
- `--limit 10`: smoke runs
- `--run-id my_run`: stable run IDs for comparisons
- `--training-paper-ids-file path/to/ids.txt`: enables leakage audit outputs
- `--bootstrap-iters 2000 --bootstrap-seed 1337`: paired significance settings

### `musciclaim-metrics`

Recomputes `scores/<run_id>/summary.csv` from existing `runs/<run_id>/**/predictions.jsonl` without
re-running the model.

```bash
musciclaim-metrics --run-id exp_003_vlm
```

### Hygiene checks

```bash
musciclaim-check-readmes
musciclaim-check-docstrings
```

Tip: run `--help` for any CLI command:
```bash
musciclaim-eval --help
```
