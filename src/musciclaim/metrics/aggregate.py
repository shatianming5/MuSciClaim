"""Aggregation and summary writing for evaluation runs."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from musciclaim.metrics.classification import compute_classification_metrics, extract_gold_pred
from musciclaim.metrics.diagnostics import compute_support_bias, compute_synergy
from musciclaim.metrics.leakage import compute_overlap_stats, read_id_set, split_by_overlap
from musciclaim.metrics.localization import compute_localization_metrics
from musciclaim.metrics.sensitivity import compute_flip_rates
from musciclaim.metrics.significance import compute_paired_significance, significance_to_dict

ALLOWED_LABELS = ["SUPPORT", "NEUTRAL", "CONTRADICT"]
ALLOWED_SET = set(ALLOWED_LABELS)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
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
        # Expected: <model>/<condition>/<prompt_mode>/predictions.jsonl
        if len(parts) < 4:
            continue
        model, condition, prompt_mode = parts[0], parts[1], parts[2]
        out.append((model, condition, prompt_mode, path))
    out.sort(key=lambda x: (x[0], x[1], x[2]))
    return out


def _flag_rate(records: list[dict[str, Any]], flag: str) -> float:
    if not records:
        return 0.0
    return sum(1 for r in records if bool(r.get(flag))) / float(len(records))


def _primary_condition_for_row(row: dict[str, Any]) -> str:
    """Pick the most meaningful condition for a given summary row."""

    if row.get("n_total_full", 0):
        return "full"
    if row.get("n_total_c_only", 0):
        return "c_only"
    if row.get("n_total_claim_only", 0):
        return "claim_only"
    return "full"


def _fmt(x: Any) -> str:
    if x is None:
        return "NA"
    if isinstance(x, float):
        return f"{x:.4f}"
    return str(x)


def _write_report_md(
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


def aggregate_run(
    *,
    run_id: str,
    runs_root: Path,
    scores_root: Path,
    training_paper_ids_file: Path | None = None,
    bootstrap_iters: int = 2000,
    bootstrap_seed: int = 1337,
) -> Path:
    """Aggregate a run directory into a single summary CSV.

    Returns:
        Path to the written summary CSV.
    """

    run_dir = runs_root / run_id
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    discovered = discover_predictions(run_dir)

    # Group: model -> prompt_mode -> condition -> records
    grouped: dict[str, dict[str, dict[str, list[dict[str, Any]]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )

    for model, condition, prompt_mode, path in discovered:
        grouped[model][prompt_mode][condition] = _read_jsonl(path)

    rows: list[dict[str, Any]] = []
    confusions: dict[str, Any] = {}

    training_ids: set[str] | None = None
    if training_paper_ids_file is not None:
        training_ids = read_id_set(training_paper_ids_file)

    leakage_audit: dict[str, Any] = {}
    significance: dict[str, Any] = {}

    for model, by_mode in grouped.items():
        for prompt_mode in ("D", "R"):
            by_cond = by_mode.get(prompt_mode, {})
            if not by_cond:
                continue

            row: dict[str, Any] = {
                "run_id": run_id,
                "model": model,
                "prompt_mode": prompt_mode,
            }

            f1_valid: dict[str, float | None] = {}

            for cond in ["full", "c_only", "f_only", "claim_only"]:
                records = by_cond.get(cond, [])
                gold, pred = extract_gold_pred(records=records)

                metrics_valid = compute_classification_metrics(
                    gold=gold,
                    pred=pred,
                    labels=ALLOWED_LABELS,
                    valid_only=True,
                )
                metrics_e2e = compute_classification_metrics(
                    gold=gold,
                    pred=pred,
                    labels=ALLOWED_LABELS,
                    valid_only=False,
                )

                row[f"macro_f1_valid_{cond}"] = metrics_valid.macro_f1 if records else None
                row[f"macro_f1_e2e_{cond}"] = metrics_e2e.macro_f1 if records else None
                row[f"n_total_{cond}"] = metrics_e2e.n_total if records else 0
                row[f"n_valid_{cond}"] = metrics_valid.n_total if records else 0
                row[f"invalid_output_rate_{cond}"] = _flag_rate(records, "invalid_output")
                row[f"invalid_input_image_rate_{cond}"] = _flag_rate(records, "invalid_input_image")
                row[f"invalid_panels_rate_{cond}"] = _flag_rate(records, "invalid_panels")
                row[f"truncated_rate_{cond}"] = _flag_rate(records, "truncated")

                # Per-class metrics (end-to-end).
                for lab in ALLOWED_LABELS:
                    suffix = f"{lab.lower()}_{cond}"
                    if records:
                        pc = metrics_e2e.per_class[lab]
                        row[f"precision_{suffix}"] = pc.precision
                        row[f"recall_{suffix}"] = pc.recall
                        row[f"f1_{suffix}"] = pc.f1
                    else:
                        row[f"precision_{suffix}"] = None
                        row[f"recall_{suffix}"] = None
                        row[f"f1_{suffix}"] = None

                # High-risk confusion counts (end-to-end).
                if records:
                    cm = metrics_e2e.confusion
                    row[f"count_contradict_to_support_{cond}"] = cm["CONTRADICT"]["SUPPORT"]
                    row[f"count_neutral_to_support_{cond}"] = cm["NEUTRAL"]["SUPPORT"]
                else:
                    row[f"count_contradict_to_support_{cond}"] = 0
                    row[f"count_neutral_to_support_{cond}"] = 0

                if records:
                    f1_valid[cond] = metrics_valid.macro_f1
                    confusions[f"{model}/{prompt_mode}/{cond}"] = metrics_e2e.confusion
                else:
                    f1_valid[cond] = None

                # Bias metrics (reported for full condition only).
                if cond == "full" and records:
                    bias = compute_support_bias(gold=gold, pred=pred, allowed_labels=ALLOWED_SET)
                    row["pred_support_rate_valid_full"] = bias.pred_support_rate_valid
                    row["fn_contradict_as_support_full"] = bias.fn_contradict_as_support
                    row["fn_neutral_as_support_full"] = bias.fn_neutral_as_support

            syn = compute_synergy(
                f1_full=f1_valid.get("full"),
                f1_c_only=f1_valid.get("c_only"),
                f1_f_only=f1_valid.get("f_only"),
            )
            row["delta_full_caption"] = syn.delta_full_caption
            row["delta_full_figure"] = syn.delta_full_figure
            row["synergy"] = syn.synergy

            # Flip-rate from full condition.
            full_recs = by_cond.get("full", [])
            if full_recs:
                sens = compute_flip_rates(records=full_recs, allowed_labels=ALLOWED_SET)
                row["flip_rate_strict"] = sens.strict_flip_rate
                row["flip_rate_partial"] = sens.partial_flip_rate
                row["flip_pairs_total"] = sens.num_pairs_total
                row["flip_pairs_valid"] = sens.num_pairs_valid
                row["flip_breakdown_json"] = json.dumps(sens.breakdown, sort_keys=True)
            else:
                row["flip_rate_strict"] = None
                row["flip_rate_partial"] = None
                row["flip_pairs_total"] = 0
                row["flip_pairs_valid"] = 0
                row["flip_breakdown_json"] = None

            # Localization from PANELS run.
            panels_recs = by_mode.get("PANELS", {}).get("full", [])
            if panels_recs:
                gold_sets = [r.get("panels_gold") or [] for r in panels_recs]
                pred_sets = [r.get("panels_pred") or [] for r in panels_recs]
                loc = compute_localization_metrics(gold_sets=gold_sets, pred_sets=pred_sets)
                row["localization_precision"] = loc.precision
                row["localization_recall"] = loc.recall
                row["localization_f1"] = loc.f1
            else:
                row["localization_precision"] = None
                row["localization_recall"] = None
                row["localization_f1"] = None

            rows.append(row)

    # Optional: leakage audit based on paper_id overlap.
    if training_ids is not None:
        for model, by_mode in grouped.items():
            for prompt_mode, by_cond in by_mode.items():
                for cond, records in by_cond.items():
                    stats = compute_overlap_stats(records=records, training_ids=training_ids)
                    overlap, non = split_by_overlap(records=records, training_ids=training_ids)

                    def _macro_f1(recs: list[dict]) -> float | None:
                        if not recs:
                            return None
                        g, p = extract_gold_pred(records=recs)
                        m = compute_classification_metrics(
                            gold=g,
                            pred=p,
                            labels=ALLOWED_LABELS,
                            valid_only=False,
                        )
                        return float(m.macro_f1)

                    leakage_audit.setdefault(model, {}).setdefault(prompt_mode, {})[cond] = {
                        "overlap_stats": {
                            "n_total": stats.n_total,
                            "n_known": stats.n_known,
                            "n_overlap": stats.n_overlap,
                            "overlap_rate_known": stats.overlap_rate_known,
                        },
                        "macro_f1_e2e_overlap": _macro_f1(overlap),
                        "macro_f1_e2e_nonoverlap": _macro_f1(non),
                    }

    # Optional: paired significance (Ours vs Base) for common settings.
    if "ours" in grouped and "base" in grouped:
        for prompt_mode in ("D", "R"):
            ours_mode = grouped["ours"].get(prompt_mode, {})
            base_mode = grouped["base"].get(prompt_mode, {})
            if not ours_mode or not base_mode:
                continue

            # Prefer 'full' when both have it; otherwise fall back to 'c_only' (text-only track).
            condition = None
            for cand in ("full", "c_only"):
                if ours_mode.get(cand) and base_mode.get(cand):
                    condition = cand
                    break
            if condition is None:
                continue

            sig = compute_paired_significance(
                ours_records=ours_mode[condition],
                base_records=base_mode[condition],
                labels=ALLOWED_LABELS,
                iters=bootstrap_iters,
                seed=bootstrap_seed,
            )

            significance[f"{condition}/{prompt_mode}"] = significance_to_dict(sig)

    scores_root.mkdir(parents=True, exist_ok=True)
    out_dir = scores_root / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_path = out_dir / "summary.csv"

    # Stable header ordering.
    fieldnames: list[str] = []
    for r in rows:
        for k in r.keys():
            if k not in fieldnames:
                fieldnames.append(k)

    with summary_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    (out_dir / "confusions.json").write_text(
        json.dumps(confusions, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    if leakage_audit:
        (out_dir / "leakage_audit.json").write_text(
            json.dumps(
                {
                    "training_paper_ids_file": str(training_paper_ids_file),
                    "models": leakage_audit,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    if significance:
        (out_dir / "significance.json").write_text(
            json.dumps(
                {
                    "bootstrap_iters": bootstrap_iters,
                    "bootstrap_seed": bootstrap_seed,
                    "paired": significance,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    _write_report_md(out_dir=out_dir, run_id=run_id, rows=rows, significance=significance)

    return summary_path
