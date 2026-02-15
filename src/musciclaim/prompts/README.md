# musciclaim/prompts/

Prompt templates with strict JSON output requirements.

## Contents

| Path | Role | Why it exists |
| --- | --- | --- |
| `src/musciclaim/prompts/README.md` | Prompts index | Documents prompt schema variants and strict JSON rules |
| `src/musciclaim/prompts/__init__.py` | Package marker | Keeps the prompt modules importable as a package |
| `src/musciclaim/prompts/templates.py` | Prompt builder | Centralizes schema prompts and retry prompts |

## How To Use

The prompt mode controls the strict JSON schema the model must return:
- `D`: decision only
  - `{"decision":"SUPPORT"}`
- `R`: short reasoning + decision
  - `{"reasoning":"...","decision":"NEUTRAL"}`
- `PANELS`: panels + reasoning + decision (localization)
  - `{"figure_panels":["Panel A"],"reasoning":"...","decision":"CONTRADICT"}`

If the runner fails to parse JSON, it can issue one "format-fix retry" prompt (enabled by
`policies.invalid_json_retry` in `configs/run.yaml`).
