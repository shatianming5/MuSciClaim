"""Hugging Face Transformers adapter for multimodal (VLM) models.

This adapter is best-effort and targets the common `AutoProcessor` + `generate()` interface.
Different VLMs can require model-specific prompting; keep the adapter small and explicit.
"""

from __future__ import annotations

import time
from typing import Any

from musciclaim.models.base import AdapterInfo, GenerationResult, ModelAdapter


def _resolve_torch_dtype(dtype: str):
    import torch

    d = (dtype or "").lower().strip()
    if d in {"float16", "fp16"}:
        return torch.float16
    if d in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if d in {"float32", "fp32"}:
        return torch.float32
    return None


class HFVLMAdapter(ModelAdapter):
    """A transformers adapter for multimodal generation.

    What it does:
        Loads a HF multimodal model/processor pair and runs `generate()` with optional images.

    Why it exists:
        The evaluation runner needs a single interface for both text-only and vision-language
        models.
    """

    def __init__(self, *, repo_id: str, revision: str | None, device: str, dtype: str) -> None:
        try:
            import torch
            from transformers import (
                AutoModelForCausalLM,
                AutoModelForVision2Seq,
                AutoProcessor,
                AutoTokenizer,
            )
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "Transformers + torch are required. Install with: pip install -e '.[hf]'"
            ) from e

        self._torch = torch
        self._device = device

        self._processor = AutoProcessor.from_pretrained(repo_id, revision=revision)

        torch_dtype = _resolve_torch_dtype(dtype)

        model = None
        load_errs: list[str] = []
        for loader in (AutoModelForVision2Seq, AutoModelForCausalLM):
            try:
                model = loader.from_pretrained(
                    repo_id,
                    revision=revision,
                    torch_dtype=torch_dtype,
                    low_cpu_mem_usage=True,
                )
                break
            except Exception as e:  # pragma: no cover
                load_errs.append(f"{loader.__name__}: {e}")

        if model is None:  # pragma: no cover
            raise RuntimeError("Failed to load VLM model. Errors:\n" + "\n".join(load_errs))

        self._model = model
        self._model.eval()
        self._model.to(device)

        # Best-effort tokenizer: many processors expose `.tokenizer`.
        tok = getattr(self._processor, "tokenizer", None)
        if tok is None:
            tok = AutoTokenizer.from_pretrained(repo_id, revision=revision, use_fast=True)
        self._tokenizer = tok

        if getattr(self._tokenizer, "pad_token", None) is None and getattr(
            self._tokenizer, "eos_token", None
        ):
            self._tokenizer.pad_token = self._tokenizer.eos_token

        self._info = AdapterInfo(
            model_id=repo_id,
            model_revision=revision,
            adapter="hf_vlm",
            modality="vlm",
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
            Always returns True (multimodal).

        Why it exists:
            The run matrix and runner use this to decide which ablations are valid for a model.
        """

        return True

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
        """Generate raw text output from a multimodal transformers model.

        What it does:
            Runs the processor on (prompt, images) and calls `model.generate()` with deterministic
            settings when `temperature == 0`.

        Why it exists:
            Provides a best-effort VLM adapter so the pipeline can measure cross-modal necessity.
        """

        del timeout_s

        t0 = time.perf_counter()

        proc_kwargs: dict[str, Any] = {"text": prompt, "return_tensors": "pt"}
        if images is not None:
            proc_kwargs["images"] = images

        inputs = self._processor(**proc_kwargs)
        for k, v in list(inputs.items()):
            if hasattr(v, "to"):
                inputs[k] = v.to(self._device)

        do_sample = float(temperature) > 0.0
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": int(max_new_tokens),
            "do_sample": do_sample,
            "pad_token_id": getattr(self._tokenizer, "pad_token_id", None),
        }
        if do_sample:
            gen_kwargs.update({"temperature": float(temperature), "top_p": float(top_p)})

        with self._torch.no_grad():
            out = self._model.generate(**inputs, **gen_kwargs)

        latency_ms = int(round((time.perf_counter() - t0) * 1000))

        input_ids = inputs.get("input_ids")
        is_encdec = bool(getattr(getattr(self._model, "config", None), "is_encoder_decoder", False))

        if input_ids is not None and not is_encdec:
            in_len = int(input_ids.shape[-1])
            new_ids = out[0][in_len:]
            tokens_in = in_len
        else:
            new_ids = out[0]
            tokens_in = int(input_ids.shape[-1]) if input_ids is not None else None

        tokens_out = int(new_ids.shape[-1])
        text = self._tokenizer.decode(new_ids, skip_special_tokens=True)
        truncated = tokens_out >= int(max_new_tokens)

        return GenerationResult(
            text=text.strip(),
            latency_ms=latency_ms,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            truncated=truncated,
        )
