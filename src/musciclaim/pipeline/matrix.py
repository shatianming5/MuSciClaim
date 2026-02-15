"""Run matrix expansion.

A run is defined by:
- model name (ours/base/...)
- input condition (full/c_only/f_only/claim_only)
- prompt mode (D/R/PANELS)

The matrix builder also enforces modality constraints.
"""

from __future__ import annotations

from dataclasses import dataclass

from musciclaim.config import MatrixConfig, ModelSpec
from musciclaim.schema import Condition, PromptMode


@dataclass(frozen=True)
class RunSpec:
    """A single runnable evaluation spec.

    What it does:
        Identifies one run cell by model name, input condition, and prompt mode.

    Why it exists:
        Makes the run matrix explicit so evaluation settings are reproducible and auditable.
    """

    model_name: str
    condition: Condition
    prompt_mode: PromptMode


def _parse_condition(s: str) -> Condition:
    try:
        return Condition(s)
    except ValueError as e:
        raise ValueError(f"Unknown condition: {s!r}") from e


def _parse_prompt_mode(s: str) -> PromptMode:
    try:
        return PromptMode(s)
    except ValueError as e:
        raise ValueError(f"Unknown prompt mode: {s!r}") from e


def build_run_matrix(*, models: dict[str, ModelSpec], matrix: MatrixConfig) -> list[RunSpec]:
    """Build the explicit run matrix."""

    conditions = [_parse_condition(c) for c in matrix.conditions]
    prompt_modes = [_parse_prompt_mode(m) for m in matrix.prompt_modes]

    runs: list[RunSpec] = []

    for model_name, spec in models.items():
        allowed_conditions = {
            Condition.C_ONLY,
            Condition.CLAIM_ONLY,
        }
        if spec.modality == "vlm":
            allowed_conditions = {
                Condition.FULL,
                Condition.C_ONLY,
                Condition.F_ONLY,
                Condition.CLAIM_ONLY,
            }

        for cond in conditions:
            if cond not in allowed_conditions:
                continue

            for pm in prompt_modes:
                if pm == PromptMode.PANELS:
                    continue
                runs.append(RunSpec(model_name=model_name, condition=cond, prompt_mode=pm))

        if matrix.include_panels_run and spec.modality == "vlm":
            # Localization is only meaningful in the full condition.
            runs.append(
                RunSpec(
                    model_name=model_name,
                    condition=Condition.FULL,
                    prompt_mode=PromptMode.PANELS,
                )
            )

    # Stable ordering for reproducibility.
    runs.sort(key=lambda r: (r.model_name, r.condition.value, r.prompt_mode.value))
    return runs
