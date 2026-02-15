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
