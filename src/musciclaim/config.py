"""Config loading for MuSciClaim.

The pipeline is driven by two YAML files:
- models config: model IDs, revisions, and adapter kind
- run config: dataset revision, preprocessing, run matrix, and IO roots

These configs are intentionally explicit to keep evaluation runs auditable.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

AdapterKind = Literal["hf_text", "hf_vlm", "dummy"]
ModelModality = Literal["text", "vlm"]


@dataclass(frozen=True)
class ModelSpec:
    """A single model to evaluate.

    What it does:
        Captures the minimal provenance needed to reproduce inference.

    Why it exists:
        We do not want model identity to be implicit in code or environment.
    """

    adapter: AdapterKind
    repo_id: str | None
    revision: str | None
    modality: ModelModality
    device: str
    dtype: str


@dataclass(frozen=True)
class DatasetConfig:
    """Dataset settings.

    What it does:
        Declares which dataset and split to load.

    Why it exists:
        Dataset revisions can change; pinning them is essential for reproducibility.
    """

    repo_id: str
    revision: str | None
    split: str


@dataclass(frozen=True)
class IOConfig:
    """Output and cache locations.

    What it does:
        Declares where the pipeline writes `runs/`, `scores/`, and `analysis/`, and where it caches
        downloaded assets (e.g., figures).

    Why it exists:
        Paths must be explicit and configurable so runs are reproducible across machines and CI.
    """

    out_root: str
    scores_root: str
    analysis_root: str
    cache_dir: str


@dataclass(frozen=True)
class PreprocessingConfig:
    """Image preprocessing controls.

    What it does:
        Specifies minimal, auditable preprocessing for figures (currently: optional resize).

    Why it exists:
        Figure handling can silently change results; making preprocessing explicit preserves
        auditability.
    """

    resize_max_side: int | None
    preserve_aspect: bool


@dataclass(frozen=True)
class DecodingConfig:
    """Text generation settings.

    What it does:
        Specifies decoding parameters used by adapters (temperature, top_p, max tokens).

    Why it exists:
        Decoding changes both accuracy and determinism, so it must be fixed and recorded.
    """

    temperature: float
    top_p: float
    max_new_tokens_decision: int
    max_new_tokens_reasoning: int


@dataclass(frozen=True)
class InferenceConfig:
    """Inference execution settings.

    What it does:
        Controls execution concerns like batching and timeouts, and embeds `DecodingConfig`.

    Why it exists:
        Separates model quality from engineering constraints and keeps runs comparable.
    """

    batch_size: int
    timeout_s: int
    decoding: DecodingConfig


@dataclass(frozen=True)
class MatrixConfig:
    """Run matrix settings.

    What it does:
        Defines which conditions and prompt modes to run, plus determinism repeats and optional
        localization runs.

    Why it exists:
        The evaluation is defined by its run matrix; declaring it in config prevents hidden
        experiment drift.
    """

    conditions: list[str]
    prompt_modes: list[str]
    include_panels_run: bool
    reproducibility_repeats: int


@dataclass(frozen=True)
class PoliciesConfig:
    """Error handling policies.

    What it does:
        Controls how the runner handles invalid JSON, image failures, and other recoverable issues.

    Why it exists:
        Error handling affects measured metrics; policies must be explicit to keep reporting honest.
    """

    invalid_json_retry: bool
    invalid_json_max_retries: int
    invalid_json_policy: str
    image_failure_policy: str


@dataclass(frozen=True)
class ReportingConfig:
    """Reporting controls.

    What it does:
        Configures deterministic error-slice sampling for human-readable analysis artifacts.

    Why it exists:
        Keeps qualitative analysis reproducible and prevents cherry-picking.
    """

    error_slice_seed: int
    error_slice_max_items: int


@dataclass(frozen=True)
class RunConfig:
    """Top-level run configuration.

    What it does:
        Bundles dataset, IO, preprocessing, inference, run matrix, and policy settings into one
        typed object.

    Why it exists:
        The run config is the single source of truth for an auditable evaluation.
    """

    dataset: DatasetConfig
    io: IOConfig
    preprocessing: PreprocessingConfig
    inference: InferenceConfig
    matrix: MatrixConfig
    policies: PoliciesConfig
    reporting: ReportingConfig


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
