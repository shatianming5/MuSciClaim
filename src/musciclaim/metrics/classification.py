"""Classification metrics for 3-class claim verification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class PerClass:
    """Per-class precision/recall/F1.

    What it does:
        Stores the per-label breakdown used by macro-averaged scores and error analysis.

    Why it exists:
        Macro-F1 alone hides failure modes (e.g., CONTRADICT collapse); per-class metrics
        expose them.
    """

    precision: float
    recall: float
    f1: float
    support: int


@dataclass(frozen=True)
class ClassificationMetrics:
    """Aggregated classification metrics.

    What it does:
        Holds macro-F1, per-class metrics, a confusion matrix, and validity counts.

    Why it exists:
        Centralizes the classification contract so reporting and significance tests stay consistent.
    """

    macro_f1: float
    per_class: dict[str, PerClass]
    confusion: dict[str, dict[str, int]]
    n_total: int
    n_valid: int


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def compute_classification_metrics(
    *,
    gold: list[str],
    pred: list[str | None],
    labels: list[str],
    valid_only: bool,
) -> ClassificationMetrics:
    """Compute Macro-F1 + per-class metrics.

    If valid_only=True, examples with pred not in `labels` (or None) are excluded.
    Otherwise, those examples count as false negatives for the gold label.
    """

    if len(gold) != len(pred):
        raise ValueError("gold/pred length mismatch")

    n_total = len(gold)

    # Determine which examples count as valid.
    valid_mask = [p in labels for p in pred]

    if valid_only:
        idxs = [i for i, ok in enumerate(valid_mask) if ok]
    else:
        idxs = list(range(n_total))

    gold_f = [gold[i] for i in idxs]
    pred_f = [pred[i] for i in idxs]

    n_valid = sum(valid_mask)

    # Confusion over known labels only; invalid preds become '__INVALID__' column.
    all_pred_labels = labels + ["__INVALID__"]

    confusion: dict[str, dict[str, int]] = {
        g: {p: 0 for p in all_pred_labels} for g in labels
    }

    for g, p in zip(gold_f, pred_f, strict=True):
        if g not in labels:
            continue
        pp = p if p in labels else "__INVALID__"
        confusion[g][pp] += 1

    per_class: dict[str, PerClass] = {}
    f1s: list[float] = []

    for lab in labels:
        tp = confusion[lab][lab]
        fp = sum(confusion[g][lab] for g in labels if g != lab)
        fn = sum(confusion[lab][p] for p in all_pred_labels if p != lab)

        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = _safe_div(2 * precision * recall, precision + recall)

        support = sum(confusion[lab].values())

        per_class[lab] = PerClass(
            precision=precision,
            recall=recall,
            f1=f1,
            support=support,
        )
        f1s.append(f1)

    macro_f1 = sum(f1s) / float(len(labels)) if labels else 0.0

    # Drop the invalid column in the public confusion matrix for readability.
    confusion_public = {g: {p: confusion[g][p] for p in labels} for g in labels}

    return ClassificationMetrics(
        macro_f1=macro_f1,
        per_class=per_class,
        confusion=confusion_public,
        n_total=n_total,
        n_valid=n_valid,
    )


def extract_gold_pred(
    *,
    records: Iterable[dict],
) -> tuple[list[str], list[str | None]]:
    """Extract gold/pred label lists from JSONL dict records."""

    gold: list[str] = []
    pred: list[str | None] = []

    for r in records:
        gold.append(r.get("label_gold"))
        pred.append(r.get("label_pred"))

    return gold, pred
