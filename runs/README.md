# runs/

Run artifacts (JSONL predictions and metadata). This directory is gitignored except for this README.

## Naming

- `runs/<run_id>/run_metadata.json`
- `runs/<run_id>/repro_report.json` (only if reproducibility repeats are enabled)
- `runs/<run_id>/<model>/<condition>/<prompt_mode>/predictions.jsonl`
- `runs/<run_id>/<model>/<condition>/<prompt_mode>/predictions_repeat_<k>.jsonl` (repeat runs for A5)

## Contents

| Path | Role | Why it exists |
| --- | --- | --- |
| `runs/.gitkeep` | Keeps directory in git | Output dirs are gitignored but documented |
| `runs/README.md` | Output contract | Makes artifacts discoverable and reproducible |

## How To Use

1. After running `musciclaim-eval`, list available run IDs:

```bash
ls -1 runs
```

2. Inspect one prediction file:

```bash
head -n 3 runs/<run_id>/<model>/<condition>/<prompt_mode>/predictions.jsonl
```

Each line is a JSON object. For example, pretty-print a single line:

```bash
python3 -c 'import json,sys; print(json.dumps(json.loads(sys.stdin.readline()), indent=2))' \\
  < runs/<run_id>/<model>/<condition>/<prompt_mode>/predictions.jsonl
```

3. Check determinism outputs (if enabled):

```bash
cat runs/<run_id>/repro_report.json
```
