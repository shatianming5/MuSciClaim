"""Typed configuration dataclasses for MuSciClaim.

What it does:
    Defines all frozen dataclasses that represent the two YAML configs
    (models config and run config).

Why it exists:
    Keeps evaluation runs auditable by making every setting explicit and typed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

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
