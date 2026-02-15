"""Evidence localization metrics (panel set-F1)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LocalizationMetrics:
    """Set-based localization metrics.

    What it does:
        Aggregates mean precision/recall/F1 for predicted panel sets against gold annotations.

    Why it exists:
        Localization turns claim-verification into an auditable process rather than a black-box
        label.
    """

    precision: float
    recall: float
    f1: float
    n_total: int


def _set_f1(gold: set[str], pred: set[str]) -> tuple[float, float, float]:
    if not pred:
        precision = 1.0 if not gold else 0.0
    else:
        precision = len(gold & pred) / float(len(pred))

    if not gold:
        recall = 1.0
    else:
        recall = len(gold & pred) / float(len(gold))

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2.0 * precision * recall / (precision + recall)

    return precision, recall, f1


def compute_localization_metrics(
    *,
    gold_sets: list[list[str]],
    pred_sets: list[list[str]],
) -> LocalizationMetrics:
    """Compute mean set-based localization precision/recall/F1.

    What it does:
        Evaluates overlap between gold and predicted panel sets per example, then averages.

    Why it exists:
        A vetting model should be able to point to which panels it used, not just output a label.
    """

    if len(gold_sets) != len(pred_sets):
        raise ValueError("gold/pred length mismatch")

    ps: list[float] = []
    rs: list[float] = []
    fs: list[float] = []

    for g, p in zip(gold_sets, pred_sets, strict=True):
        pr, rc, f1 = _set_f1(set(g), set(p))
        ps.append(pr)
        rs.append(rc)
        fs.append(f1)

    n = len(gold_sets)
    return LocalizationMetrics(
        precision=sum(ps) / n if n else 0.0,
        recall=sum(rs) / n if n else 0.0,
        f1=sum(fs) / n if n else 0.0,
        n_total=n,
    )
