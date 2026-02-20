"""Hugging Face Transformers adapter for text-only models."""

from __future__ import annotations

import time
from typing import Any

from musciclaim.models.base import AdapterInfo, GenerationResult, ModelAdapter
from musciclaim.models.torch_utils import resolve_torch_dtype


class HFTextAdapter(ModelAdapter):
    """A text-only transformers adapter.

    What it does:
        Loads a model/tokenizer from the HF Hub and runs deterministic generation.

    Why it exists:
        The pipeline needs a consistent interface regardless of model architecture.
    """

    def __init__(self, *, repo_id: str, revision: str | None, device: str, dtype: str) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "Transformers + torch are required. Install with: pip install -e '.[hf]'"
            ) from e

        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(repo_id, revision=revision, use_fast=True)

        torch_dtype = resolve_torch_dtype(dtype)
        self._model = AutoModelForCausalLM.from_pretrained(
            repo_id,
            revision=revision,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
        )
        self._model.eval()

        self._device = device
        self._model.to(device)

        if self._tokenizer.pad_token is None and self._tokenizer.eos_token is not None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        self._info = AdapterInfo(
            model_id=repo_id,
            model_revision=revision,
            adapter="hf_text",
            modality="text",
        )

    @property
    def info(self) -> AdapterInfo:
        """Adapter identity and provenance.

        What it does:
            Returns the HF repo ID and revision used to load the model, plus modality metadata.

        Why it exists:
            Artifacts must record exactly which checkpoint produced each prediction.
        """

        return self._info

    @property
    def supports_images(self) -> bool:
        """Whether this adapter can accept image inputs.

        What it does:
            Always returns False (text-only).

        Why it exists:
            The runner uses this to avoid passing figures to adapters that cannot consume them.
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
        """Generate raw text output from a text-only transformers model.

        What it does:
            Tokenizes the prompt and runs `model.generate()` with deterministic settings when
            `temperature == 0`.

        Why it exists:
            Provides a minimal adapter so the evaluation runner can stay model-agnostic.
        """

        del images, timeout_s

        t0 = time.perf_counter()

        inputs = self._tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self._device)
        attn = inputs.get("attention_mask")
        if attn is not None:
            attn = attn.to(self._device)

        do_sample = float(temperature) > 0.0
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": int(max_new_tokens),
            "do_sample": do_sample,
            "pad_token_id": self._tokenizer.pad_token_id,
        }
        if do_sample:
            gen_kwargs.update({"temperature": float(temperature), "top_p": float(top_p)})

        with self._torch.no_grad():
            out = self._model.generate(input_ids=input_ids, attention_mask=attn, **gen_kwargs)

        latency_ms = int(round((time.perf_counter() - t0) * 1000))

        in_len = int(input_ids.shape[-1])
        out_ids = out[0]
        new_ids = out_ids[in_len:]
        text = self._tokenizer.decode(new_ids, skip_special_tokens=True)

        tokens_in = in_len
        tokens_out = int(new_ids.shape[-1])
        truncated = tokens_out >= int(max_new_tokens)

        return GenerationResult(
            text=text.strip(),
            latency_ms=latency_ms,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            truncated=truncated,
        )
