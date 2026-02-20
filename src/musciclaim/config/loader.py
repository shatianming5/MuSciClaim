"""YAML config loading for MuSciClaim.

What it does:
    Parses models.yaml and run.yaml into typed dataclass trees.

Why it exists:
    Centralises all YAML-to-dataclass conversion so the rest of the codebase
    only deals with typed objects.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from musciclaim.config.dataclasses import (
    DatasetConfig,
    DecodingConfig,
    InferenceConfig,
    IOConfig,
    MatrixConfig,
    ModelSpec,
    PoliciesConfig,
    PreprocessingConfig,
    ReportingConfig,
    RunConfig,
)


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file into a dictionary."""

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML dict at top-level: {path}")

    return data


def load_models_config(path: str | Path) -> dict[str, ModelSpec]:
    """Load models YAML.

    Returns a dict keyed by model name (typically: 'ours', 'base').
    """

    path = Path(path)
    raw = _load_yaml(path)

    out: dict[str, ModelSpec] = {}
    for name, spec in raw.items():
        if not isinstance(spec, dict):
            raise ValueError(f"Model spec must be a dict: {name}")
        out[name] = ModelSpec(
            adapter=spec.get("adapter", "dummy"),
            repo_id=spec.get("repo_id"),
            revision=spec.get("revision"),
            modality=spec.get("modality", "text"),
            device=spec.get("device", "cpu"),
            dtype=spec.get("dtype", "float32"),
        )
    return out


def load_run_config(path: str | Path) -> RunConfig:
    """Load run YAML into a typed RunConfig."""

    path = Path(path)
    raw = _load_yaml(path)

    dataset = raw.get("dataset", {})
    io = raw.get("io", {})
    preprocessing = raw.get("preprocessing", {})
    inference = raw.get("inference", {})
    decoding = (inference.get("decoding") or {})
    matrix = raw.get("matrix", {})
    policies = raw.get("policies", {})
    reporting = raw.get("reporting", {})

    return RunConfig(
        dataset=DatasetConfig(
            repo_id=str(dataset.get("repo_id")),
            revision=dataset.get("revision"),
            split=str(dataset.get("split", "test")),
        ),
        io=IOConfig(
            out_root=str(io.get("out_root", "runs")),
            scores_root=str(io.get("scores_root", "scores")),
            analysis_root=str(io.get("analysis_root", "analysis")),
            cache_dir=str(io.get("cache_dir", ".cache/musciclaim")),
        ),
        preprocessing=PreprocessingConfig(
            resize_max_side=preprocessing.get("resize_max_side"),
            preserve_aspect=bool(preprocessing.get("preserve_aspect", True)),
        ),
        inference=InferenceConfig(
            batch_size=int(inference.get("batch_size", 1)),
            timeout_s=int(inference.get("timeout_s", 120)),
            decoding=DecodingConfig(
                temperature=float(decoding.get("temperature", 0.0)),
                top_p=float(decoding.get("top_p", 1.0)),
                max_new_tokens_decision=int(decoding.get("max_new_tokens_decision", 128)),
                max_new_tokens_reasoning=int(decoding.get("max_new_tokens_reasoning", 256)),
            ),
        ),
        matrix=MatrixConfig(
            conditions=list(matrix.get("conditions", ["full", "c_only", "f_only", "claim_only"])),
            prompt_modes=list(matrix.get("prompt_modes", ["D", "R"])),
            include_panels_run=bool(matrix.get("include_panels_run", True)),
            reproducibility_repeats=int(matrix.get("reproducibility_repeats", 2)),
        ),
        policies=PoliciesConfig(
            invalid_json_retry=bool(policies.get("invalid_json_retry", True)),
            invalid_json_max_retries=int(policies.get("invalid_json_max_retries", 1)),
            invalid_json_policy=str(policies.get("invalid_json_policy", "invalid")),
            image_failure_policy=str(policies.get("image_failure_policy", "invalid")),
        ),
        reporting=ReportingConfig(
            error_slice_seed=int(reporting.get("error_slice_seed", 1337)),
            error_slice_max_items=int(reporting.get("error_slice_max_items", 30)),
        ),
    )
