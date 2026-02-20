"""Shared torch utilities for model adapters.

What it does:
    Maps human-readable dtype strings to ``torch.dtype`` values.

Why it exists:
    Both ``HFTextAdapter`` and ``HFVLMAdapter`` need this mapping;
    a single definition avoids drift between the two.
"""

from __future__ import annotations


def resolve_torch_dtype(dtype: str):
    """Map a human string (e.g. ``"bfloat16"``) to a ``torch.dtype``."""

    import torch

    d = (dtype or "").lower().strip()
    if d in {"float16", "fp16"}:
        return torch.float16
    if d in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if d in {"float32", "fp32"}:
        return torch.float32
    return None
