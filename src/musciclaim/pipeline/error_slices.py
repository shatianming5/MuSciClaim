"""Error-slice report generation.

What it does:
    Writes a minimal Markdown report listing high-risk misclassifications
    (CONTRADICT -> SUPPORT) for quick human review.

Why it exists:
    Qualitative error slices complement numeric metrics and help detect
    systematic failure modes that averages obscure.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from musciclaim.pipeline.helpers import ensure_dir


def write_error_slices(
    *,
    run_id: str,
    analysis_root: Path,
    examples_by_claim: dict[str, Any],
    predictions_path: Path,
    model_name: str,
    prompt_mode: str,
    max_items: int,
) -> None:
    """Write a minimal error slice report for high-risk failures."""

    if not predictions_path.exists():
        return

    items: list[dict[str, Any]] = []
    with predictions_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("label_gold") == "CONTRADICT" and r.get("label_pred") == "SUPPORT":
                items.append(r)

    items = items[:max_items]
    if not items:
        return

    out_dir = analysis_root / run_id / model_name
    ensure_dir(out_dir)

    md = out_dir / f"error_slices_{prompt_mode}.md"

    lines: list[str] = []
    lines.append(f"# Error Slices ({model_name}, {prompt_mode})\n")
    lines.append("High-risk slice: **CONTRADICT -> SUPPORT**.\n")

    for r in items:
        ex = examples_by_claim.get(r.get("claim_id"))
        claim_text = getattr(ex, "claim_text", None)
        caption = getattr(ex, "caption", None)
        fig = getattr(ex, "figure_filepath", None)

        lines.append("## Case\n")
        lines.append(f"- claim_id: `{r.get('claim_id')}`\n")
        lines.append(f"- base_claim_id: `{r.get('base_claim_id')}`\n")
        lines.append(f"- gold: `{r.get('label_gold')}`\n")
        lines.append(f"- pred: `{r.get('label_pred')}`\n")
        if fig:
            lines.append(f"- figure: `{fig}`\n")
        if claim_text:
            lines.append("\n**Claim**\n\n")
            lines.append(claim_text + "\n")
        if caption:
            lines.append("\n**Caption**\n\n")
            lines.append(caption + "\n")
        if r.get("reasoning"):
            lines.append("\n**Model reasoning**\n\n")
            lines.append(str(r.get("reasoning")) + "\n")

    md.write_text("".join(lines), encoding="utf-8")
