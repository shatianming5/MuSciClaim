"""Core dataclasses for MuSciClaim.

What it does:
    Defines the frozen dataclasses used across the evaluation pipeline —
    evaluation examples, parsed outputs, image metadata, and prediction records.

Why it exists:
    Keeps the pipeline strongly typed and easy to audit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from musciclaim.schema.enums import Decision


@dataclass(frozen=True)
class EvalExample:
    """A single MuSciClaims example in evaluation-ready form.

    What it does:
        Normalizes the dataset row into a stable set of fields used across the pipeline.

    Why it exists:
        Keeps dataset-specific quirks from leaking into prompting, inference, and metrics.
    """

    claim_id: str
    base_claim_id: str
    claim_text: str
    caption: str
    label_gold: Decision
    figure_filepath: str
    panels_gold: list[str]
    paper_id: str | None = None


@dataclass(frozen=True)
class ImageMeta:
    """Auditable image preprocessing metadata.

    What it does:
        Records original/resized dimensions and resize ratio for each processed figure.

    Why it exists:
        Multi-panel figures are sensitive to preprocessing; metadata is required for
        reproducibility.
    """

    filepath: str
    orig_w: int | None
    orig_h: int | None
    new_w: int | None
    new_h: int | None
    resize_ratio: float | None


@dataclass(frozen=True)
class GenerationResult:
    """Raw generation output from a model adapter.

    What it does:
        Holds the adapter's raw text output plus latency/token counters.

    Why it exists:
        Downstream parsing and audits should never depend on re-running the model.
    """

    text: str
    latency_ms: int
    tokens_in: int | None
    tokens_out: int | None
    truncated: bool


@dataclass(frozen=True)
class ParsedOutput:
    """A validated, schema-conforming model output.

    What it does:
        Represents a model output that passed strict JSON parsing and label validation.

    Why it exists:
        Keeps invalid outputs explicit so metrics can account for them without ambiguity.
    """

    decision: Decision
    reasoning: str | None = None
    figure_panels: list[str] | None = None


@dataclass(frozen=True)
class PredictionRecord:
    """A single JSONL record written to predictions files.

    What it does:
        Captures everything needed for reproducible, machine-checkable evaluation.

    Why it exists:
        Metrics and audits should never depend on re-running models.
    """

    claim_id: str
    base_claim_id: str
    label_gold: str
    label_pred: str | None
    panels_gold: list[str]
    panels_pred: list[str] | None
    reasoning: str | None
    invalid_output: bool
    invalid_input_image: bool
    invalid_panels: bool
    truncated: bool
    latency_ms: int | None
    tokens_in: int | None
    tokens_out: int | None
    image_meta: ImageMeta | None
    model_id: str
    model_revision: str | None
    dataset_revision: str | None
    run_id: str
    condition: str
    prompt_mode: str
    raw_text: str | None = None
    paper_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serializable dict.

        What it does:
            Produces the stable on-disk JSONL representation for this record.

        Why it exists:
            Downstream tools (metrics, audits) should consume artifacts without importing code.
        """

        d: dict[str, Any] = {
            "claim_id": self.claim_id,
            "base_claim_id": self.base_claim_id,
            "paper_id": self.paper_id,
            "label_gold": self.label_gold,
            "label_pred": self.label_pred,
            "panels_gold": self.panels_gold,
            "panels_pred": self.panels_pred,
            "reasoning": self.reasoning,
            "invalid_output": self.invalid_output,
            "invalid_input_image": self.invalid_input_image,
            "invalid_panels": self.invalid_panels,
            "truncated": self.truncated,
            "latency_ms": self.latency_ms,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "image_meta": None if self.image_meta is None else {
                "filepath": self.image_meta.filepath,
                "orig_w": self.image_meta.orig_w,
                "orig_h": self.image_meta.orig_h,
                "new_w": self.image_meta.new_w,
                "new_h": self.image_meta.new_h,
                "resize_ratio": self.image_meta.resize_ratio,
            },
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "dataset_revision": self.dataset_revision,
            "run_id": self.run_id,
            "condition": self.condition,
            "prompt_mode": self.prompt_mode,
        }
        if self.raw_text is not None:
            d["raw_text"] = self.raw_text
        return d
