"""Reproducibility checking utilities.

What it does:
    Decides whether a run spec should be repeated for determinism audits (A5)
    and compares repeated prediction files for agreement.

Why it exists:
    Determinism checks are integral to the evaluation contract but orthogonal
    to the main inference loop.
"""

from __future__ import annotations

import json
from itertools import zip_longest
from pathlib import Path

from musciclaim.config import ModelSpec
from musciclaim.schema import Condition, PromptMode


def repro_key(*, model_name: str, condition: Condition, prompt_mode: PromptMode) -> str:
    """Build a stable key for reproducibility reporting."""
    return f"{model_name}/{condition.value}/{prompt_mode.value}"


def should_repro_check(
    *,
    model_spec: ModelSpec,
    spec_condition: Condition,
    spec_mode: PromptMode,
) -> bool:
    """Return True if this run should be repeated for determinism checks (A5)."""

    if spec_mode not in {PromptMode.D, PromptMode.R}:
        return False

    if model_spec.modality == "vlm":
        return spec_condition == Condition.FULL

    return spec_condition == Condition.C_ONLY


def compare_prediction_files(*, baseline: Path, other: Path) -> tuple[float, list[str]]:
    """Compare two prediction JSONL files and return (agreement_rate, inconsistent_claim_ids)."""

    fields = [
        "label_pred",
        "panels_pred",
        "reasoning",
        "invalid_output",
        "invalid_panels",
        "raw_text",
    ]

    total = 0
    ok = 0
    inconsistent: list[str] = []

    with baseline.open("r", encoding="utf-8") as f0, other.open("r", encoding="utf-8") as f1:
        for l0, l1 in zip_longest(f0, f1):
            if l0 is None or l1 is None:
                inconsistent.append("__LENGTH_MISMATCH__")
                break

            if not l0.strip() and not l1.strip():
                continue

            r0 = json.loads(l0)
            r1 = json.loads(l1)

            total += 1
            k0 = tuple(r0.get(f) for f in fields)
            k1 = tuple(r1.get(f) for f in fields)
            if k0 == k1:
                ok += 1
            else:
                inconsistent.append(str(r0.get("claim_id")))

    rate = (ok / total) if total else 1.0
    return rate, inconsistent
