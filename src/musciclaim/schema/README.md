# musciclaim/schema/

Core enums and frozen dataclasses that define the evaluation contracts.

Formerly a single `schema.py` module; split into a package for clarity.

## Contents

| Path | Role | Why it exists |
| --- | --- | --- |
| `schema/__init__.py` | Re-export hub | Preserves `from musciclaim.schema import X` compatibility |
| `schema/enums.py` | `Decision`, `PromptMode`, `Condition` | Strongly-typed labels, prompt modes, and ablation conditions |
| `schema/dataclasses.py` | `EvalExample`, `PredictionRecord`, etc. | Stable record types consumed by runner, metrics, and audits |

## Import Examples

```python
# These all work exactly as before:
from musciclaim.schema import Decision, PromptMode, Condition
from musciclaim.schema import PredictionRecord, EvalExample

# You can also import from sub-modules directly:
from musciclaim.schema.enums import Decision
from musciclaim.schema.dataclasses import PredictionRecord
```
