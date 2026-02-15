"""Epistemic sensitivity metrics (flip-rate)."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass


@dataclass(frozen=True)
class SensitivityMetrics:
    """Flip-rate metrics grouped by base_claim_id.

    What it does:
        Summarizes how often predictions flip between SUPPORT and CONTRADICT variants, plus a
        breakdown of transition types.

    Why it exists:
        Vetting requires sensitivity to perturbations; flip-rate exposes models that default to
        SUPPORT.
    """

    strict_flip_rate: float
    partial_flip_rate: float
    num_pairs_total: int
    num_pairs_valid: int
    breakdown: dict[str, int]


def compute_flip_rates(*, records: list[dict], allowed_labels: set[str]) -> SensitivityMetrics:
    """Compute flip-rate on SUPPORT vs CONTRADICT variants of the same base claim."""

    by_base: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_base[str(r.get("base_claim_id"))].append(r)

    total_pairs = 0
    valid_pairs = 0

    strict_ok = 0
    partial_ok = 0

    breakdown: dict[str, int] = defaultdict(int)

    for base_id, items in by_base.items():
        del base_id

        support = next((x for x in items if x.get("label_gold") == "SUPPORT"), None)
        contra = next((x for x in items if x.get("label_gold") == "CONTRADICT"), None)
        if support is None or contra is None:
            continue

        total_pairs += 1

        ps = support.get("label_pred")
        pc = contra.get("label_pred")
        if ps in allowed_labels and pc in allowed_labels:
            valid_pairs += 1
            breakdown[f"{ps}->{pc}"] += 1

            if ps == "SUPPORT" and pc == "CONTRADICT":
                strict_ok += 1
            if ps == "SUPPORT" and pc in {"NEUTRAL", "CONTRADICT"}:
                partial_ok += 1
        else:
            breakdown[f"{ps}->{pc}"] += 1

    strict_rate = (strict_ok / valid_pairs) if valid_pairs else 0.0
    partial_rate = (partial_ok / valid_pairs) if valid_pairs else 0.0

    return SensitivityMetrics(
        strict_flip_rate=strict_rate,
        partial_flip_rate=partial_rate,
        num_pairs_total=total_pairs,
        num_pairs_valid=valid_pairs,
        breakdown=dict(breakdown),
    )
