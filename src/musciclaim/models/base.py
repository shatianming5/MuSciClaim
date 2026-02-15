"""Model adapter interface.

The evaluation pipeline must be able to swap models without changing the runner.
Adapters provide a single `generate()` entrypoint that returns raw model text and basic stats.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AdapterInfo:
    """Provenance for a loaded adapter.

    What it does:
        Captures model identity (repo + revision) and adapter modality for artifact provenance.

    Why it exists:
        Evaluation outputs must be attributable to a specific model version to be auditable.
    """

    model_id: str
    model_revision: str | None
    adapter: str
    modality: str


@dataclass(frozen=True)
class GenerationResult:
    """A single generation result from an adapter.

    What it does:
        Stores raw text plus basic performance counters (latency and token counts).

    Why it exists:
        Artifact records should include enough context for debugging throughput and truncation.
    """

    text: str
    latency_ms: int
    tokens_in: int | None
    tokens_out: int | None
    truncated: bool


class ModelAdapter(ABC):
    """Abstract adapter for model inference.

    What it does:
        Defines the minimum interface the pipeline needs for model inference.

    Why it exists:
        VLMs and text models vary wildly; we want a stable runner.
    """

    @property
    @abstractmethod
    def info(self) -> AdapterInfo:
        """Adapter identity and provenance.

        What it does:
            Returns stable metadata for this adapter instance (IDs, revision, modality).

        Why it exists:
            The runner writes these fields into artifacts so results can be traced and reproduced.
        """

    @property
    @abstractmethod
    def supports_images(self) -> bool:
        """Whether this adapter can accept image inputs.

        What it does:
            Returns True for multimodal adapters and False for text-only adapters.

        Why it exists:
            The run matrix and runner must avoid sending images to models that cannot consume them.
        """

    @abstractmethod
    def generate(
        self,
        *,
        prompt: str,
        images: list[Any] | None,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        timeout_s: int,
    ) -> GenerationResult:
        """Generate raw text output from the model.

        What it does:
            Runs a single generation call using the adapter's underlying model/processor.

        Why it exists:
            Keeps the pipeline model-agnostic by exposing one stable inference entrypoint.

        Args:
            prompt: Fully-rendered prompt string.
            images: Optional list of PIL images (for VLMs).
            max_new_tokens: Generation cap for the output.
            temperature/top_p: Decoding parameters (typically deterministic).
            timeout_s: Best-effort timeout. Adapters may ignore if unsupported.
        """
