"""Prediction file discovery and JSONL reading.

What it does:
    Locates ``predictions.jsonl`` files inside a run directory and loads them
    into plain dicts.

Why it exists:
    Discovery logic is reused by both the aggregation pipeline and the
    metrics CLI; keeping it separate avoids duplication.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ALLOWED_LABELS = ["SUPPORT", "NEUTRAL", "CONTRADICT"]
ALLOWED_SET = set(ALLOWED_LABELS)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file into a list of dicts."""
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def discover_predictions(run_dir: Path) -> list[tuple[str, str, str, Path]]:
    """Return (model_name, condition, prompt_mode, path) tuples."""

    out: list[tuple[str, str, str, Path]] = []
    for path in run_dir.rglob("predictions.jsonl"):
        rel = path.relative_to(run_dir)
        parts = rel.parts
        if len(parts) < 4:
            continue
        model, condition, prompt_mode = parts[0], parts[1], parts[2]
        out.append((model, condition, prompt_mode, path))
    out.sort(key=lambda x: (x[0], x[1], x[2]))
    return out


def flag_rate(records: list[dict[str, Any]], flag: str) -> float:
    """Fraction of *records* where *flag* is truthy."""
    if not records:
        return 0.0
    return sum(1 for r in records if bool(r.get(flag))) / float(len(records))
