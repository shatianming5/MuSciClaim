"""Evaluation runner.

This module executes the run matrix:
- loads dataset examples
- downloads and preprocesses images (when required)
- calls model adapters
- enforces strict JSON output schemas
- writes `predictions.jsonl` artifacts
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from itertools import zip_longest
from pathlib import Path
from typing import Any

from tqdm import tqdm

from musciclaim.config import ModelSpec, RunConfig
from musciclaim.data.hf_dataset import load_musciclaims
from musciclaim.data.images import download_hf_dataset_file, open_and_preprocess_image
from musciclaim.metrics.aggregate import aggregate_run
from musciclaim.models.factory import build_adapter
from musciclaim.panels import PanelWhitelist, normalize_panel_list
from musciclaim.parse import try_parse_model_output
from musciclaim.pipeline.matrix import build_run_matrix
from musciclaim.pipeline.metadata import collect_run_metadata, write_run_metadata
from musciclaim.prompts.templates import build_prompt
from musciclaim.schema import Condition, ImageMeta, PredictionRecord, PromptMode


def _utc_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _condition_flags(cond: Condition) -> tuple[bool, bool]:
    """Return (figure_provided, caption_provided)."""

    if cond == Condition.FULL:
        return True, True
    if cond == Condition.C_ONLY:
        return False, True
    if cond == Condition.F_ONLY:
        return True, False
    if cond == Condition.CLAIM_ONLY:
        return False, False
    raise ValueError(f"Unknown condition: {cond}")


def _max_new_tokens(cfg: RunConfig, pm: PromptMode) -> int:
    if pm == PromptMode.D:
        return cfg.inference.decoding.max_new_tokens_decision
    return cfg.inference.decoding.max_new_tokens_reasoning


def _write_jsonl_line(fp, obj: dict[str, Any]) -> None:
    fp.write(json.dumps(obj, ensure_ascii=True) + "\n")


def _repro_key(*, model_name: str, condition: Condition, prompt_mode: PromptMode) -> str:
    """Build a stable key for reproducibility reporting."""

    return f"{model_name}/{condition.value}/{prompt_mode.value}"


def _should_repro_check(
    *,
    model_spec: ModelSpec,
    spec_condition: Condition,
    spec_mode: PromptMode,
) -> bool:
    """Return True if this run should be repeated for determinism checks (A5)."""

    if spec_mode not in {PromptMode.D, PromptMode.R}:
        return False

    # Prefer the primary evidence-bearing condition per modality.
    if model_spec.modality == "vlm":
        return spec_condition == Condition.FULL

    return spec_condition == Condition.C_ONLY


def _compare_prediction_files(*, baseline: Path, other: Path) -> tuple[float, list[str]]:
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
                # Length mismatch: treat remaining examples as inconsistent.
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


def _write_error_slices(
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

    # High-risk: CONTRADICT -> SUPPORT
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
    _ensure_dir(out_dir)

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


def run_evaluation(
    *,
    run_cfg: RunConfig,
    models_cfg: dict[str, ModelSpec],
    limit: int | None,
    run_id: str | None,
    training_paper_ids_file: Path | None = None,
    bootstrap_iters: int = 2000,
    bootstrap_seed: int = 1337,
    repo_root: Path,
) -> str:
    """Run the full evaluation loop and write artifacts.

    Returns:
        The resolved run_id.
    """

    resolved_run_id = run_id or _utc_run_id()

    runs_root = (repo_root / run_cfg.io.out_root).resolve()
    scores_root = (repo_root / run_cfg.io.scores_root).resolve()
    analysis_root = (repo_root / run_cfg.io.analysis_root).resolve()

    run_dir = runs_root / resolved_run_id
    _ensure_dir(run_dir)

    # Load dataset.
    examples, prov = load_musciclaims(cfg=run_cfg.dataset, limit=limit)
    examples_by_claim = {ex.claim_id: ex for ex in examples}

    # Build run matrix and load adapters.
    run_specs = build_run_matrix(models=models_cfg, matrix=run_cfg.matrix)

    adapters = {name: build_adapter(spec=spec, name=name) for name, spec in models_cfg.items()}

    # Metadata.
    meta = collect_run_metadata(
        repo_root=repo_root,
        run_id=resolved_run_id,
        run_cfg=run_cfg,
        models_cfg=models_cfg,
    )
    meta["dataset_provenance"] = asdict(prov)
    write_run_metadata(path=run_dir / "run_metadata.json", metadata=meta)

    whitelist = PanelWhitelist.az()
    repro_report: dict[str, Any] = {}

    for spec in run_specs:
        adapter = adapters[spec.model_name]
        model_spec = models_cfg[spec.model_name]
        figure_flag, caption_flag = _condition_flags(spec.condition)

        out_dir = run_dir / spec.model_name / spec.condition.value / spec.prompt_mode.value
        _ensure_dir(out_dir)
        baseline_path = out_dir / "predictions.jsonl"

        max_new = _max_new_tokens(run_cfg, spec.prompt_mode)

        repeats = 1
        if run_cfg.matrix.reproducibility_repeats > 1 and _should_repro_check(
            model_spec=model_spec,
            spec_condition=spec.condition,
            spec_mode=spec.prompt_mode,
        ):
            repeats = run_cfg.matrix.reproducibility_repeats

        for repeat_idx in range(repeats):
            out_path = (
                baseline_path
                if repeat_idx == 0
                else out_dir / f"predictions_repeat_{repeat_idx}.jsonl"
            )

            with out_path.open("w", encoding="utf-8") as fp:
                desc = f"{spec.model_name}:{spec.condition.value}:{spec.prompt_mode.value}"
                if repeat_idx > 0:
                    desc = f"{desc}:repeat_{repeat_idx}"

                for ex in tqdm(examples, desc=desc):
                    invalid_input_image = False
                    image_meta = None
                    images: list[Any] | None = None

                    if figure_flag:
                        try:
                            img_path = download_hf_dataset_file(
                                repo_id=run_cfg.dataset.repo_id,
                                filename=ex.figure_filepath,
                                revision=prov.revision_resolved or run_cfg.dataset.revision,
                                cache_dir=run_cfg.io.cache_dir,
                            )
                            loaded = open_and_preprocess_image(
                                path=img_path,
                                preprocessing=run_cfg.preprocessing,
                            )
                            image_meta = loaded.meta
                            if loaded.image is None:
                                invalid_input_image = True
                            else:
                                images = [loaded.image]
                        except Exception:
                            invalid_input_image = True
                            image_meta = ImageMeta(
                                filepath=ex.figure_filepath,
                                orig_w=None,
                                orig_h=None,
                                new_w=None,
                                new_h=None,
                                resize_ratio=None,
                            )

                    # If an image is required but missing, record an invalid example.
                    if figure_flag and invalid_input_image:
                        rec = PredictionRecord(
                            claim_id=ex.claim_id,
                            base_claim_id=ex.base_claim_id,
                            paper_id=ex.paper_id,
                            label_gold=ex.label_gold.value,
                            label_pred=None,
                            panels_gold=ex.panels_gold,
                            panels_pred=None,
                            reasoning=None,
                            invalid_output=False,
                            invalid_input_image=True,
                            invalid_panels=False,
                            truncated=False,
                            latency_ms=None,
                            tokens_in=None,
                            tokens_out=None,
                            image_meta=image_meta,
                            model_id=adapter.info.model_id,
                            model_revision=adapter.info.model_revision,
                            dataset_revision=prov.revision_resolved,
                            run_id=resolved_run_id,
                            condition=spec.condition.value,
                            prompt_mode=spec.prompt_mode.value,
                            raw_text=None,
                        )
                        _write_jsonl_line(fp, rec.to_dict())
                        continue

                    prompt = build_prompt(
                        mode=spec.prompt_mode,
                        claim_text=ex.claim_text,
                        caption_text=ex.caption,
                        figure_provided=figure_flag,
                        caption_provided=caption_flag,
                        retry=False,
                    )

                    gen = adapter.generate(
                        prompt=prompt,
                        images=images if adapter.supports_images else None,
                        max_new_tokens=max_new,
                        temperature=run_cfg.inference.decoding.temperature,
                        top_p=run_cfg.inference.decoding.top_p,
                        timeout_s=run_cfg.inference.timeout_s,
                    )

                    parsed, _err = try_parse_model_output(
                        text=gen.text,
                        mode=spec.prompt_mode,
                    )

                    retries = 0
                    raw_text = gen.text
                    if parsed is None and run_cfg.policies.invalid_json_retry:
                        while (
                            retries < run_cfg.policies.invalid_json_max_retries
                            and parsed is None
                        ):
                            retries += 1
                            retry_prompt = build_prompt(
                                mode=spec.prompt_mode,
                                claim_text=ex.claim_text,
                                caption_text=ex.caption,
                                figure_provided=figure_flag,
                                caption_provided=caption_flag,
                                retry=True,
                            )
                            gen2 = adapter.generate(
                                prompt=retry_prompt,
                                images=images if adapter.supports_images else None,
                                max_new_tokens=max_new,
                                temperature=run_cfg.inference.decoding.temperature,
                                top_p=run_cfg.inference.decoding.top_p,
                                timeout_s=run_cfg.inference.timeout_s,
                            )
                            raw_text = gen2.text
                            parsed, _err = try_parse_model_output(
                                text=gen2.text,
                                mode=spec.prompt_mode,
                            )
                            gen = gen2

                    invalid_output = parsed is None
                    invalid_panels = False

                    label_pred = parsed.decision.value if parsed else None
                    reasoning = parsed.reasoning if parsed else None
                    panels_pred = parsed.figure_panels if parsed else None

                    if panels_pred is not None:
                        norm_panels, had_invalid = normalize_panel_list(panels_pred)
                        invalid_panels = had_invalid or (not whitelist.validate(norm_panels))
                        panels_pred = norm_panels

                    rec = PredictionRecord(
                        claim_id=ex.claim_id,
                        base_claim_id=ex.base_claim_id,
                        paper_id=ex.paper_id,
                        label_gold=ex.label_gold.value,
                        label_pred=label_pred,
                        panels_gold=ex.panels_gold,
                        panels_pred=panels_pred,
                        reasoning=reasoning,
                        invalid_output=invalid_output,
                        invalid_input_image=False,
                        invalid_panels=invalid_panels,
                        truncated=gen.truncated,
                        latency_ms=gen.latency_ms,
                        tokens_in=gen.tokens_in,
                        tokens_out=gen.tokens_out,
                        image_meta=image_meta,
                        model_id=adapter.info.model_id,
                        model_revision=adapter.info.model_revision,
                        dataset_revision=prov.revision_resolved,
                        run_id=resolved_run_id,
                        condition=spec.condition.value,
                        prompt_mode=spec.prompt_mode.value,
                        raw_text=raw_text,
                    )
                    _write_jsonl_line(fp, rec.to_dict())

            if repeat_idx > 0:
                rate, inconsistent = _compare_prediction_files(
                    baseline=baseline_path,
                    other=out_path,
                )
                key = _repro_key(
                    model_name=spec.model_name,
                    condition=spec.condition,
                    prompt_mode=spec.prompt_mode,
                )
                entry = repro_report.setdefault(key, {"repeats": repeats, "comparisons": []})
                entry["comparisons"].append(
                    {
                        "repeat": repeat_idx,
                        "agreement_rate": rate,
                        "num_inconsistent": len(inconsistent),
                        "inconsistent_claim_ids": inconsistent[:50],
                    }
                )

    if repro_report:
        (run_dir / "repro_report.json").write_text(
            json.dumps(repro_report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    # Aggregate metrics (+ optional audits).
    aggregate_run(
        run_id=resolved_run_id,
        runs_root=runs_root,
        scores_root=scores_root,
        training_paper_ids_file=training_paper_ids_file,
        bootstrap_iters=bootstrap_iters,
        bootstrap_seed=bootstrap_seed,
    )

    # Minimal error slices for quick debugging.
    for model_name in models_cfg.keys():
        # Prefer the evidence-bearing condition if present; fall back for text-only runs.
        pred_path = None
        for cond in ["full", "c_only", "claim_only"]:
            p = run_dir / model_name / cond / "D" / "predictions.jsonl"
            if p.exists():
                pred_path = p
                break

        if pred_path is None:
            continue

        _write_error_slices(
            run_id=resolved_run_id,
            analysis_root=analysis_root,
            examples_by_claim=examples_by_claim,
            predictions_path=pred_path,
            model_name=model_name,
            prompt_mode="D",
            max_items=run_cfg.reporting.error_slice_max_items,
        )

    return resolved_run_id
