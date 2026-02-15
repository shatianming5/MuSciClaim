"""Diagnostic metrics: support bias and cross-modal synergy."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SupportBiasMetrics:
    """Bias metrics focused on the tendency to predict SUPPORT.

    What it does:
        Summarizes support-rate and high-risk false-negative rates where evidence is ignored.

    Why it exists:
        MuSciClaims highlights systematic SUPPORT bias; tracking it prevents misleading conclusions.
    """

    pred_support_rate_valid: float
    fn_contradict_as_support: float
    fn_neutral_as_support: float


def compute_support_bias(
    *,
    gold: list[str],
    pred: list[str | None],
    allowed_labels: set[str],
) -> SupportBiasMetrics:
    """Compute support-bias diagnostics.

    What it does:
        Quantifies how often the model predicts SUPPORT, and how often CONTRADICT/NEUTRAL
        examples are incorrectly predicted as SUPPORT.

    Why it exists:
        MuSciClaims notes systematic bias toward SUPPORT; this makes the bias measurable.
    """

    if len(gold) != len(pred):
        raise ValueError("gold/pred length mismatch")

    valid = [p in allowed_labels for p in pred]
    denom_valid = sum(valid)
    pred_support = sum(1 for p in pred if p == "SUPPORT")
    pred_support_rate_valid = (pred_support / denom_valid) if denom_valid else 0.0

    # Treat invalid preds as "not SUPPORT" in FN rates.
    gold_contra = [i for i, g in enumerate(gold) if g == "CONTRADICT"]
    gold_neutral = [i for i, g in enumerate(gold) if g == "NEUTRAL"]

    fn_contra_as_support = (
        sum(1 for i in gold_contra if pred[i] == "SUPPORT") / len(gold_contra)
        if gold_contra
        else 0.0
    )
    fn_neutral_as_support = (
        sum(1 for i in gold_neutral if pred[i] == "SUPPORT") / len(gold_neutral)
        if gold_neutral
        else 0.0
    )

    return SupportBiasMetrics(
        pred_support_rate_valid=pred_support_rate_valid,
        fn_contradict_as_support=fn_contra_as_support,
        fn_neutral_as_support=fn_neutral_as_support,
    )


@dataclass(frozen=True)
class SynergyMetrics:
    """Cross-modal synergy computed from per-condition scores.

    What it does:
        Stores Full-vs-ablation deltas and the synergy gain `Full - max(C-only, F-only)`.

    Why it exists:
        A model can score well using a single modality; synergy quantifies true cross-modal benefit.
    """

    delta_full_caption: float | None
    delta_full_figure: float | None
    synergy: float | None


def compute_synergy(
    *,
    f1_full: float | None,
    f1_c_only: float | None,
    f1_f_only: float | None,
) -> SynergyMetrics:
    """Compute cross-modal synergy diagnostics from per-condition scores.

    What it does:
        Computes deltas (Full - CaptionOnly), (Full - FigureOnly), and synergy gain
        (Full - max(C-only, F-only)).

    Why it exists:
        A high Full score is not meaningful if a single modality achieves the same result.
    """

    if f1_full is None:
        return SynergyMetrics(delta_full_caption=None, delta_full_figure=None, synergy=None)

    d_fc = (f1_full - f1_c_only) if f1_c_only is not None else None
    d_ff = (f1_full - f1_f_only) if f1_f_only is not None else None

    best_single = None
    if f1_c_only is not None or f1_f_only is not None:
        best_single = max(x for x in [f1_c_only, f1_f_only] if x is not None)

    syn = (f1_full - best_single) if best_single is not None else None
    return SynergyMetrics(delta_full_caption=d_fc, delta_full_figure=d_ff, synergy=syn)
