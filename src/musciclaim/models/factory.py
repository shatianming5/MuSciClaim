"""Adapter factory.

Centralizes the mapping from config to adapter implementations.
"""

from __future__ import annotations

from musciclaim.config import ModelSpec
from musciclaim.models.base import ModelAdapter
from musciclaim.models.dummy import DummyAdapter
from musciclaim.models.hf_text import HFTextAdapter
from musciclaim.models.hf_vlm import HFVLMAdapter


def build_adapter(*, spec: ModelSpec, name: str) -> ModelAdapter:
    """Instantiate a ModelAdapter from a ModelSpec."""

    if spec.adapter == "dummy":
        return DummyAdapter(name=f"{name}-dummy")

    if not spec.repo_id:
        raise ValueError(f"Model '{name}' requires repo_id when adapter={spec.adapter!r}")

    if spec.adapter == "hf_text":
        return HFTextAdapter(
            repo_id=spec.repo_id,
            revision=spec.revision,
            device=spec.device,
            dtype=spec.dtype,
        )

    if spec.adapter == "hf_vlm":
        return HFVLMAdapter(
            repo_id=spec.repo_id,
            revision=spec.revision,
            device=spec.device,
            dtype=spec.dtype,
        )

    raise ValueError(f"Unknown adapter kind: {spec.adapter}")
