# tests/

Unit tests for parsing, panel normalization, and metrics. The test suite is intentionally small: it validates the strict contracts that keep the evaluation pipeline auditable.

## Contents

| Path | Role | Why it exists |
| --- | --- | --- |
| `tests/README.md` | Test index | Explains the test coverage and intent |
| `tests/test_parse.py` | Output parsing tests | Prevents silent schema drift and weak JSON validation |
| `tests/test_panels.py` | Panel normalization tests | Keeps localization evaluation consistent |
| `tests/test_metrics.py` | Metrics tests | Ensures Macro-F1, localization, and flip-rate computations are correct |
| `tests/test_bootstrap.py` | Bootstrap tests | Ensures paired bootstrap CI helper behaves as expected |

## How To Use

Run the unit tests from the repo root:

```bash
pytest -q
```

If you are using the repo venv explicitly:

```bash
.venv/bin/pytest -q
```
