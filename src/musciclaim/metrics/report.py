"""Markdown report generation.

What it does:
    Renders a one-page deliverable-style Markdown report from aggregated rows
    and optional significance results.

Why it exists:
    Human-readable summaries complement machine-readable CSV/JSON artifacts
    and are quicker to review.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _fmt(x: Any) -> str:
    if x is None:
        return "NA"
    if isinstance(x, float):
        return f"{x:.4f}"
    return str(x)


def _primary_condition_for_row(row: dict[str, Any]) -> str:
    """Pick the most meaningful condition for a given summary row."""

    if row.get("n_total_full", 0):
        return "full"
    if row.get("n_total_c_only", 0):
        return "c_only"
    if row.get("n_total_claim_only", 0):
        return "claim_only"
    return "full"


def write_report_md(
    *,
    out_dir: Path,
    run_id: str,
    rows: list[dict[str, Any]],
    significance: dict[str, Any],
) -> None:
    """Write a one-page Markdown report (deliverable-style)."""

    lines: list[str] = []
    lines.append(f"# MuSciClaim Report: {run_id}\n\n")

    lines.append("## Summary\n\n")
    lines.append(
        "| Model | Prompt | Primary cond | Macro-F1 (e2e) | CONTRADICT F1 | "
        "Synergy | Flip strict | Loc F1 | Invalid out |\n"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |\n")

    for r in sorted(rows, key=lambda x: (x.get("model", ""), x.get("prompt_mode", ""))):
        cond = _primary_condition_for_row(r)
        macro = r.get(f"macro_f1_e2e_{cond}")
        contra = r.get(f"f1_contradict_{cond}")
        invalid = r.get(f"invalid_output_rate_{cond}")

        cols = [
            str(r.get("model")),
            str(r.get("prompt_mode")),
            cond,
            _fmt(macro),
            _fmt(contra),
            _fmt(r.get("synergy")),
            _fmt(r.get("flip_rate_strict")),
            _fmt(r.get("localization_f1")),
            _fmt(invalid),
        ]
        lines.append("| " + " | ".join(cols) + " |\n")

    if significance:
        lines.append("\n## Paired Significance (Ours vs Base)\n\n")
        lines.append("Paired bootstrap CIs are reported for `(ours - base)`.\n\n")

        for k in sorted(significance.keys()):
            s = significance[k]
            macro = s["macro_f1_e2e_diff"]
            contra = s["contradict_f1_e2e_diff"]

            lines.append(f"### {k}\n\n")
            lines.append(f"- n_common: {s['n_common']}\n")
            lines.append(
                f"- Macro-F1 diff: {_fmt(macro['diff'])} "
                f"(95% CI [{_fmt(macro['ci95_low'])}, {_fmt(macro['ci95_high'])}])\n"
            )
            lines.append(
                f"- CONTRADICT F1 diff: {_fmt(contra['diff'])} "
                f"(95% CI [{_fmt(contra['ci95_low'])}, {_fmt(contra['ci95_high'])}])\n"
            )

    lines.append("\n## Artifacts\n\n")
    lines.append(f"- `scores/{run_id}/summary.csv`\n")
    lines.append(f"- `scores/{run_id}/confusions.json`\n")
    if significance:
        lines.append(f"- `scores/{run_id}/significance.json`\n")
    lines.append(f"- `runs/{run_id}/run_metadata.json`\n")
    lines.append(f"- `runs/{run_id}/repro_report.json` (if repeats enabled)\n")

    (out_dir / "report.md").write_text("".join(lines), encoding="utf-8")
