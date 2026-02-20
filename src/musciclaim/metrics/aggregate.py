"""Aggregation and summary writing for evaluation runs."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from musciclaim.metrics.classification import compute_classification_metrics, extract_gold_pred
from musciclaim.metrics.diagnostics import compute_support_bias, compute_synergy
from musciclaim.metrics.discovery import (
    ALLOWED_LABELS,
    ALLOWED_SET,
    discover_predictions,
    flag_rate,
    read_jsonl,
)
from musciclaim.metrics.leakage import compute_overlap_stats, read_id_set, split_by_overlap
from musciclaim.metrics.localization import compute_localization_metrics
from musciclaim.metrics.report import write_report_md
from musciclaim.metrics.sensitivity import compute_flip_rates
from musciclaim.metrics.significance import compute_paired_significance, significance_to_dict


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

    grouped: dict[str, dict[str, dict[str, list[dict[str, Any]]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )

    for model, condition, prompt_mode, path in discovered:
        grouped[model][prompt_mode][condition] = read_jsonl(path)

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

            row = _build_model_row(
                run_id=run_id,
                model=model,
                prompt_mode=prompt_mode,
                by_cond=by_cond,
                by_mode=by_mode,
                confusions=confusions,
            )
            rows.append(row)

    if training_ids is not None:
        _run_leakage_audit(
            grouped=grouped,
            training_ids=training_ids,
            leakage_audit=leakage_audit,
        )

    if "ours" in grouped and "base" in grouped:
        _run_paired_significance(
            grouped=grouped,
            significance=significance,
            bootstrap_iters=bootstrap_iters,
            bootstrap_seed=bootstrap_seed,
        )

    _write_artifacts(
        scores_root=scores_root,
        run_id=run_id,
        rows=rows,
        confusions=confusions,
        leakage_audit=leakage_audit,
        significance=significance,
        training_paper_ids_file=training_paper_ids_file,
        bootstrap_iters=bootstrap_iters,
        bootstrap_seed=bootstrap_seed,
    )

    return scores_root / run_id / "summary.csv"


def _build_model_row(
    *,
    run_id: str,
    model: str,
    prompt_mode: str,
    by_cond: dict[str, list[dict[str, Any]]],
    by_mode: dict[str, dict[str, list[dict[str, Any]]]],
    confusions: dict[str, Any],
) -> dict[str, Any]:
    """Compute all metrics for one (model, prompt_mode) combination."""

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
            gold=gold, pred=pred, labels=ALLOWED_LABELS, valid_only=True,
        )
        metrics_e2e = compute_classification_metrics(
            gold=gold, pred=pred, labels=ALLOWED_LABELS, valid_only=False,
        )

        row[f"macro_f1_valid_{cond}"] = metrics_valid.macro_f1 if records else None
        row[f"macro_f1_e2e_{cond}"] = metrics_e2e.macro_f1 if records else None
        row[f"n_total_{cond}"] = metrics_e2e.n_total if records else 0
        row[f"n_valid_{cond}"] = metrics_valid.n_total if records else 0
        row[f"invalid_output_rate_{cond}"] = flag_rate(records, "invalid_output")
        row[f"invalid_input_image_rate_{cond}"] = flag_rate(records, "invalid_input_image")
        row[f"invalid_panels_rate_{cond}"] = flag_rate(records, "invalid_panels")
        row[f"truncated_rate_{cond}"] = flag_rate(records, "truncated")

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

    return row


def _run_leakage_audit(
    *,
    grouped: dict[str, dict[str, dict[str, list[dict[str, Any]]]]],
    training_ids: set[str],
    leakage_audit: dict[str, Any],
) -> None:
    """Run the leakage audit based on paper_id overlap."""

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
                        gold=g, pred=p, labels=ALLOWED_LABELS, valid_only=False,
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


def _run_paired_significance(
    *,
    grouped: dict[str, dict[str, dict[str, list[dict[str, Any]]]]],
    significance: dict[str, Any],
    bootstrap_iters: int,
    bootstrap_seed: int,
) -> None:
    """Compute paired significance between 'ours' and 'base' models."""

    for prompt_mode in ("D", "R"):
        ours_mode = grouped["ours"].get(prompt_mode, {})
        base_mode = grouped["base"].get(prompt_mode, {})
        if not ours_mode or not base_mode:
            continue

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


def _write_artifacts(
    *,
    scores_root: Path,
    run_id: str,
    rows: list[dict[str, Any]],
    confusions: dict[str, Any],
    leakage_audit: dict[str, Any],
    significance: dict[str, Any],
    training_paper_ids_file: Path | None,
    bootstrap_iters: int,
    bootstrap_seed: int,
) -> None:
    """Write all score artifacts to disk."""

    scores_root.mkdir(parents=True, exist_ok=True)
    out_dir = scores_root / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_path = out_dir / "summary.csv"

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

    write_report_md(out_dir=out_dir, run_id=run_id, rows=rows, significance=significance)
