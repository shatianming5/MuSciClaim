# src/

Source code for the MuSciClaim Python package (installed as `musciclaim`).

## Contents

| Path | Role | Why it exists |
| --- | --- | --- |
| `src/README.md` | Source index | Explains what lives under `src/` |
| `src/musciclaim/` | Main library package | Keeps evaluation logic reusable and testable |

## How To Use

This repo uses the standard Python `src/` layout. Install in editable mode:

```bash
pip install -e '.[dev,hf]'
```

Then use the CLI entrypoints (recommended) or import `musciclaim` from Python.
