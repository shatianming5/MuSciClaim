# musciclaim/config/

Typed configuration dataclasses and YAML loaders for MuSciClaim.

Formerly a single `config.py` module; split into a package for clarity.

## Contents

| Path | Role | Why it exists |
| --- | --- | --- |
| `config/__init__.py` | Re-export hub | Preserves `from musciclaim.config import X` compatibility |
| `config/dataclasses.py` | All config dataclasses | `ModelSpec`, `RunConfig`, `DatasetConfig`, `IOConfig`, etc. |
| `config/loader.py` | YAML parsing | `load_models_config()` and `load_run_config()` |

## Import Examples

```python
# These all work exactly as before:
from musciclaim.config import ModelSpec, RunConfig
from musciclaim.config import load_models_config, load_run_config

# You can also import from sub-modules directly:
from musciclaim.config.dataclasses import PreprocessingConfig
from musciclaim.config.loader import load_run_config
```
