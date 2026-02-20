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
from musciclaim.pipeline.error_slices import write_error_slices
from musciclaim.pipeline.helpers import (
    condition_flags,
    ensure_dir,
    max_new_tokens,
    utc_run_id,
    write_jsonl_line,
)
from musciclaim.pipeline.matrix import build_run_matrix
from musciclaim.pipeline.metadata import collect_run_metadata, write_run_metadata
from musciclaim.pipeline.repro import compare_prediction_files, repro_key, should_repro_check
from musciclaim.prompts.templates import build_prompt
from musciclaim.schema import Condition, ImageMeta, PredictionRecord, PromptMode


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

    resolved_run_id = run_id or utc_run_id()

    runs_root = (repo_root / run_cfg.io.out_root).resolve()
    scores_root = (repo_root / run_cfg.io.scores_root).resolve()
    analysis_root = (repo_root / run_cfg.io.analysis_root).resolve()

    run_dir = runs_root / resolved_run_id
    ensure_dir(run_dir)

    examples, prov = load_musciclaims(cfg=run_cfg.dataset, limit=limit)
    examples_by_claim = {ex.claim_id: ex for ex in examples}

    run_specs = build_run_matrix(models=models_cfg, matrix=run_cfg.matrix)

    adapters = {name: build_adapter(spec=spec, name=name) for name, spec in models_cfg.items()}

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
        figure_flag, caption_flag = condition_flags(spec.condition)

        out_dir = run_dir / spec.model_name / spec.condition.value / spec.prompt_mode.value
        ensure_dir(out_dir)
        baseline_path = out_dir / "predictions.jsonl"

        max_new = max_new_tokens(run_cfg, spec.prompt_mode)

        repeats = 1
        if run_cfg.matrix.reproducibility_repeats > 1 and should_repro_check(
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
                    rec = _infer_one(
                        ex=ex,
                        adapter=adapter,
                        spec=spec,
                        run_cfg=run_cfg,
                        prov=prov,
                        figure_flag=figure_flag,
                        caption_flag=caption_flag,
                        max_new=max_new,
                        whitelist=whitelist,
                        resolved_run_id=resolved_run_id,
                    )
                    write_jsonl_line(fp, rec.to_dict())

            if repeat_idx > 0:
                rate, inconsistent = compare_prediction_files(
                    baseline=baseline_path,
                    other=out_path,
                )
                key = repro_key(
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

    aggregate_run(
        run_id=resolved_run_id,
        runs_root=runs_root,
        scores_root=scores_root,
        training_paper_ids_file=training_paper_ids_file,
        bootstrap_iters=bootstrap_iters,
        bootstrap_seed=bootstrap_seed,
    )

    for model_name in models_cfg.keys():
        pred_path = None
        for cond in ["full", "c_only", "claim_only"]:
            p = run_dir / model_name / cond / "D" / "predictions.jsonl"
            if p.exists():
                pred_path = p
                break

        if pred_path is None:
            continue

        write_error_slices(
            run_id=resolved_run_id,
            analysis_root=analysis_root,
            examples_by_claim=examples_by_claim,
            predictions_path=pred_path,
            model_name=model_name,
            prompt_mode="D",
            max_items=run_cfg.reporting.error_slice_max_items,
        )

    return resolved_run_id


def _infer_one(
    *,
    ex: Any,
    adapter: Any,
    spec: Any,
    run_cfg: RunConfig,
    prov: Any,
    figure_flag: bool,
    caption_flag: bool,
    max_new: int,
    whitelist: PanelWhitelist,
    resolved_run_id: str,
) -> PredictionRecord:
    """Run inference on a single example and return a PredictionRecord."""

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

    if figure_flag and invalid_input_image:
        return PredictionRecord(
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

    return PredictionRecord(
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
