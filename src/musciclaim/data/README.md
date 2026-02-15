# musciclaim/data/

Dataset loading, preflight checks, and image retrieval/preprocessing.

## Contents

| Path | Role | Why it exists |
| --- | --- | --- |
| `src/musciclaim/data/README.md` | Data index | Documents dataset loading and image handling |
| `src/musciclaim/data/__init__.py` | Package marker | Keeps the data modules importable as a package |
| `src/musciclaim/data/hf_dataset.py` | HF dataset loader | Loads MuSciClaims with version recording and field checks |
| `src/musciclaim/data/images.py` | Image download/open/resize | Keeps visual preprocessing minimal and auditable |

## How To Use

### Via the CLI

The runner downloads figures on demand and caches them under `io.cache_dir` from `configs/run.yaml`.
You generally do not need to call this module directly:

```bash
musciclaim-eval --run-config configs/run.yaml --models-config configs/models.yaml
```

### Programmatic loading

```python
from musciclaim.config import DatasetConfig
from musciclaim.data.hf_dataset import load_musciclaims

examples, prov = load_musciclaims(
    cfg=DatasetConfig(repo_id="StonyBrookNLP/MuSciClaims", revision=None, split="test"),
    limit=5,
)
print(prov)
print(examples[0])
```

### Image caching behavior

For each example with a figure, the runner calls `huggingface_hub.hf_hub_download` and stores the
result in `io.cache_dir`. This makes repeated runs faster and avoids re-downloading figures.
