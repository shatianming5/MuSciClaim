# analysis/

Human-readable error slices and qualitative analysis derived from `runs/`. This directory is gitignored except for this README.

## Naming

- `analysis/<run_id>/<model>/error_slices_D.md`

## Contents

| Path | Role | Why it exists |
| --- | --- | --- |
| `analysis/.gitkeep` | Keeps directory in git | Output dirs are gitignored but documented |
| `analysis/README.md` | Output contract | Explains where error slices and notes live |

## How To Use

After running `musciclaim-eval`, look for high-risk failure slices (e.g., CONTRADICT -> SUPPORT):

```bash
ls -1 analysis/<run_id>/
cat analysis/<run_id>/<model>/error_slices_D.md
```

These slices are intentionally small and are meant for fast, human review.
