# scores/

Aggregated metrics (CSV/JSON) derived from `runs/`. This directory is gitignored except for this README.

## Naming

- `scores/<run_id>/summary.csv`
- `scores/<run_id>/confusions.json`
- `scores/<run_id>/report.md`
- `scores/<run_id>/significance.json` (paired bootstrap CIs if `ours` and `base` exist)
- `scores/<run_id>/leakage_audit.json` (only if a training `paper_id` list is provided)

## Contents

| Path | Role | Why it exists |
| --- | --- | --- |
| `scores/.gitkeep` | Keeps directory in git | Output dirs are gitignored but documented |
| `scores/README.md` | Output contract | Explains where summary tables live |

## How To Use

1. After running `musciclaim-eval`, open the one-page report:

```bash
cat scores/<run_id>/report.md
```

2. Inspect the CSV summary:

```bash
head -n 5 scores/<run_id>/summary.csv
```

3. Recompute metrics without rerunning inference:

```bash
musciclaim-metrics --run-id <run_id>
```
