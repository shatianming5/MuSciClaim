"""Pipeline utility functions.

What it does:
    Provides small, stateless helpers shared across the evaluation runner.

Why it exists:
    Keeps the main runner module focused on orchestration logic.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from musciclaim.config import RunConfig
from musciclaim.schema import Condition, PromptMode


def utc_run_id() -> str:
    """Generate a UTC-timestamped run identifier."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def ensure_dir(path: Path) -> None:
    """Create directory tree if it does not exist."""
    path.mkdir(parents=True, exist_ok=True)


def condition_flags(cond: Condition) -> tuple[bool, bool]:
    """Return (figure_provided, caption_provided) for a given condition."""

    if cond == Condition.FULL:
        return True, True
    if cond == Condition.C_ONLY:
        return False, True
    if cond == Condition.F_ONLY:
        return True, False
    if cond == Condition.CLAIM_ONLY:
        return False, False
    raise ValueError(f"Unknown condition: {cond}")


def max_new_tokens(cfg: RunConfig, pm: PromptMode) -> int:
    """Pick the appropriate max_new_tokens for *pm*."""
    if pm == PromptMode.D:
        return cfg.inference.decoding.max_new_tokens_decision
    return cfg.inference.decoding.max_new_tokens_reasoning


def write_jsonl_line(fp, obj: dict[str, Any]) -> None:
    """Append one JSON object as a single line."""
    fp.write(json.dumps(obj, ensure_ascii=True) + "\n")
