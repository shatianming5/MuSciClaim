"""Statistical comparison between two models (paired significance).

This module produces paired bootstrap confidence intervals for key metrics.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from musciclaim.metrics.bootstrap import BootstrapCI, paired_bootstrap_ci_diff
from musciclaim.metrics.classification import compute_classification_metrics


@dataclass(frozen=True)
class PairedSignificance:
    """Paired significance results for a single condition/prompt setting.

    What it does:
        Stores paired bootstrap confidence intervals for key metric differences (ours - base).

    Why it exists:
        Turns qualitative improvements into quantified evidence with uncertainty bounds.
    """

    n_common: int
    macro_f1_e2e_diff: BootstrapCI
    contradict_f1_e2e_diff: BootstrapCI


def _align_by_claim_id(
    *,
    ours_records: list[dict],
    base_records: list[dict],
) -> tuple[list[str], list[str], list[str | None], list[str | None]]:
    """Align two record lists by claim_id.

    Returns:
        (claim_ids, gold_labels, ours_preds, base_preds)
    """

    ours_by_id = {str(r.get("claim_id")): r for r in ours_records}
    base_by_id = {str(r.get("claim_id")): r for r in base_records}

    common = sorted(set(ours_by_id.keys()) & set(base_by_id.keys()))

    gold: list[str] = []
    ours_pred: list[str | None] = []
    base_pred: list[str | None] = []

    for cid in common:
        ro = ours_by_id[cid]
        rb = base_by_id[cid]
        gold.append(str(ro.get("label_gold")))
        ours_pred.append(ro.get("label_pred"))
        base_pred.append(rb.get("label_pred"))

    return common, gold, ours_pred, base_pred


def compute_paired_significance(
    *,
    ours_records: list[dict],
    base_records: list[dict],
    labels: list[str],
    iters: int,
    seed: int,
) -> PairedSignificance:
    """Compute paired bootstrap CIs for macro-F1 and CONTRADICT F1."""

    _, gold, ours_pred, base_pred = _align_by_claim_id(
        ours_records=ours_records,
        base_records=base_records,
    )

    def macro_f1(g: list[str], p: list[str | None]) -> float:
        m = compute_classification_metrics(gold=g, pred=p, labels=labels, valid_only=False)
        return float(m.macro_f1)

    def contradict_f1(g: list[str], p: list[str | None]) -> float:
        m = compute_classification_metrics(gold=g, pred=p, labels=labels, valid_only=False)
        return float(m.per_class["CONTRADICT"].f1)

    macro_ci = paired_bootstrap_ci_diff(
        gold=gold,
        pred_a=ours_pred,
        pred_b=base_pred,
        metric_fn=macro_f1,
        iters=iters,
        seed=seed,
    )
    contra_ci = paired_bootstrap_ci_diff(
        gold=gold,
        pred_a=ours_pred,
        pred_b=base_pred,
        metric_fn=contradict_f1,
        iters=iters,
        seed=seed,
    )

    return PairedSignificance(
        n_common=len(gold),
        macro_f1_e2e_diff=macro_ci,
        contradict_f1_e2e_diff=contra_ci,
    )


def significance_to_dict(sig: PairedSignificance) -> dict:
    """Convert PairedSignificance to a JSON-serializable dict."""

    d = asdict(sig)

    # Expand nested dataclasses into plain dicts.
    d["macro_f1_e2e_diff"] = asdict(sig.macro_f1_e2e_diff)
    d["contradict_f1_e2e_diff"] = asdict(sig.contradict_f1_e2e_diff)
    return d
