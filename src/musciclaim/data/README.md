# musciclaim/data/

Dataset loading, preflight checks, and image retrieval/preprocessing.

## Contents

| Path | Role | Why it exists |
| --- | --- | --- |
| `src/musciclaim/data/README.md` | Data index | Documents dataset loading and image handling |
| `src/musciclaim/data/__init__.py` | Package marker | Keeps the data modules importable as a package |
| `src/musciclaim/data/hf_dataset.py` | HF dataset loader | Loads MuSciClaims with version recording and field checks |
| `src/musciclaim/data/images.py` | Image download/open/resize | Keeps visual preprocessing minimal and auditable |
