"""Hugging Face dataset loading for MuSciClaims."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from musciclaim.config import DatasetConfig
from musciclaim.schema import Decision, EvalExample

REQUIRED_FIELDS = [
    "base_claim_id",
    "claim_id",
    "claim_text",
    "label_3class",
    "associated_figure_filepath",
    "associated_figure_panels",
    "caption",
]


@dataclass(frozen=True)
class DatasetProvenance:
    """Provenance information for a loaded dataset.

    What it does:
        Records the requested and resolved dataset revision, split name, and row count.

    Why it exists:
        Dataset content can change over time; provenance is required to reproduce and audit runs.
    """

    repo_id: str
    revision_requested: str | None
    revision_resolved: str | None
    split: str
    num_rows: int


def _require_fields(row: dict[str, Any]) -> None:
    missing = [k for k in REQUIRED_FIELDS if k not in row]
    if missing:
        raise KeyError(f"Dataset row missing required fields: {missing}")


def try_resolve_dataset_revision(repo_id: str, revision: str | None) -> str | None:
    """Best-effort resolution of a dataset git SHA on the Hugging Face Hub."""

    try:
        from huggingface_hub import HfApi

        api = HfApi()
        info = api.dataset_info(repo_id=repo_id, revision=revision)
        return getattr(info, "sha", None)
    except Exception:
        return None


def load_musciclaims(
    *,
    cfg: DatasetConfig,
    limit: int | None = None,
) -> tuple[list[EvalExample], DatasetProvenance]:
    """Load MuSciClaims as a list of EvalExample.

    What it does:
        Loads the HF dataset split and validates required fields and label values.

    Why it exists:
        Metrics and prompting require a stable input contract independent of HF internals.
    """

    try:
        from datasets import load_dataset  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "The 'datasets' package is required. Install with: pip install -e '.[hf]'"
        ) from e

    ds = load_dataset(cfg.repo_id, split=cfg.split, revision=cfg.revision)

    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))

    examples: list[EvalExample] = []
    for row in ds:
        if not isinstance(row, dict):
            row = dict(row)
        _require_fields(row)

        label_raw = row["label_3class"]
        try:
            label = Decision(str(label_raw))
        except ValueError as e:
            raise ValueError(f"Invalid label_3class value: {label_raw!r}") from e

        panels_raw = row.get("associated_figure_panels")
        if not isinstance(panels_raw, list):
            panels = []
        else:
            panels = [str(x) for x in panels_raw]

        examples.append(
            EvalExample(
                claim_id=str(row["claim_id"]),
                base_claim_id=str(row["base_claim_id"]),
                claim_text=str(row["claim_text"]),
                caption=str(row["caption"]),
                label_gold=label,
                figure_filepath=str(row["associated_figure_filepath"]),
                panels_gold=panels,
                paper_id=str(row["paper_id"]) if row.get("paper_id") is not None else None,
            )
        )

    resolved = try_resolve_dataset_revision(cfg.repo_id, cfg.revision)

    prov = DatasetProvenance(
        repo_id=cfg.repo_id,
        revision_requested=cfg.revision,
        revision_resolved=resolved,
        split=cfg.split,
        num_rows=len(examples),
    )
    return examples, prov


def iter_examples(examples: Iterable[EvalExample]) -> Iterable[EvalExample]:
    """A tiny adapter to keep call sites explicit."""

    yield from examples
