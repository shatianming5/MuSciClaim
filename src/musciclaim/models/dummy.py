"""Deterministic dummy adapter.

This adapter exists to:
- enable CI and smoke runs without GPUs
- validate pipeline wiring and artifact contracts

It does NOT attempt to be accurate; it only emits schema-correct JSON.
"""

from __future__ import annotations

import time
from typing import Any

from musciclaim.models.base import AdapterInfo, GenerationResult, ModelAdapter


class DummyAdapter(ModelAdapter):
    """A deterministic adapter that always returns valid JSON.

    What it does:
        Produces schema-correct JSON responses without any model weights or external downloads.

    Why it exists:
        Enables CI and local smoke tests to validate runner/metrics wiring without GPUs.
    """

    def __init__(self, *, name: str = "dummy") -> None:
        self._info = AdapterInfo(
            model_id=name,
            model_revision=None,
            adapter="dummy",
            modality="text",
        )

    @property
    def info(self) -> AdapterInfo:
        """Adapter identity and provenance.

        What it does:
            Returns stable metadata for this deterministic adapter instance.

        Why it exists:
            Artifacts should record a model identity even in smoke tests.
        """

        return self._info

    @property
    def supports_images(self) -> bool:
        """Whether this adapter can accept image inputs.

        What it does:
            Always returns False.

        Why it exists:
            Ensures image paths are exercised via the runner without requiring multimodal models.
        """

        return False

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
        """Generate a deterministic, schema-correct JSON response.

        What it does:
            Emits a minimal JSON object matching the requested schema inferred from the prompt.

        Why it exists:
            Lets tests validate strict parsing, artifact writing, and metrics without model
            variance.
        """

        start = time.perf_counter()

        # Heuristic: infer requested schema by looking for key names in the prompt.
        if "\"figure_panels\"" in prompt:
            text = (
                '{"figure_panels":[],"reasoning":"No evidence is provided.",'
                '"decision":"NEUTRAL"}'
            )
        elif "\"reasoning\"" in prompt:
            text = '{"reasoning":"No evidence is provided.","decision":"NEUTRAL"}'
        else:
            text = '{"decision":"NEUTRAL"}'

        latency_ms = int(round((time.perf_counter() - start) * 1000))
        return GenerationResult(
            text=text,
            latency_ms=latency_ms,
            tokens_in=0,
            tokens_out=0,
            truncated=False,
        )
