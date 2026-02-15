# scripts/

Developer utilities and CLI helpers (lint, checks, one-off maintenance).

## Contents

| Path | Role | Why it exists |
| --- | --- | --- |
| `scripts/README.md` | Scripts index | Explains developer utilities included in this repo |
| `scripts/check_readmes.py` | README enforcement | Ensures every required directory has an English README with a contents section |
| `scripts/check_docstrings.py` | Docstring enforcement | Ensures public classes and methods include required "What/Why" docstring markers |

## How To Use

Run the checks locally (from repo root):

```bash
python3 scripts/check_readmes.py
python3 scripts/check_docstrings.py
```

If you have the package installed in a venv, you can also use the CLI entrypoints:

```bash
musciclaim-check-readmes
musciclaim-check-docstrings
```
