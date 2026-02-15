"""Strict JSON parsing for model outputs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from musciclaim.schema import Decision, ParsedOutput, PromptMode


@dataclass(frozen=True)
class ParseError(Exception):
    """Raised when a model output cannot be parsed or validated.

    What it does:
        Carries a human-readable error message describing the strict schema violation.

    Why it exists:
        The runner needs a structured, machine-checkable way to record invalid model outputs.
    """

    message: str

    def __str__(self) -> str:  # pragma: no cover
        return self.message


def _strict_load_json_object(text: str) -> dict[str, Any]:
    s = (text or "").strip()
    if not s:
        raise ParseError("empty output")

    if not (s.startswith("{") and s.endswith("}")):
        raise ParseError("output must be a single JSON object with no extra text")

    try:
        obj = json.loads(s)
    except json.JSONDecodeError as e:
        raise ParseError(f"invalid JSON: {e.msg}") from e

    if not isinstance(obj, dict):
        raise ParseError("JSON must be an object")

    return obj


def parse_model_output(*, text: str, mode: PromptMode) -> ParsedOutput:
    """Parse and validate a model output for a given prompt mode."""

    obj = _strict_load_json_object(text)

    if mode == PromptMode.D:
        expected = {"decision"}
    elif mode == PromptMode.R:
        expected = {"reasoning", "decision"}
    elif mode == PromptMode.PANELS:
        expected = {"figure_panels", "reasoning", "decision"}
    else:
        raise ValueError(f"Unknown prompt mode: {mode}")

    if set(obj.keys()) != expected:
        raise ParseError(f"unexpected keys: got {sorted(obj.keys())}, expected {sorted(expected)}")

    decision_raw = obj.get("decision")
    if not isinstance(decision_raw, str):
        raise ParseError("decision must be a string")

    try:
        decision = Decision(decision_raw)
    except ValueError as e:
        raise ParseError(f"invalid decision: {decision_raw!r}") from e

    reasoning: str | None = None
    if "reasoning" in obj:
        reasoning_raw = obj.get("reasoning")
        if not isinstance(reasoning_raw, str):
            raise ParseError("reasoning must be a string")
        reasoning = reasoning_raw

    panels: list[str] | None = None
    if "figure_panels" in obj:
        raw = obj.get("figure_panels")
        if not isinstance(raw, list) or not all(isinstance(x, str) for x in raw):
            raise ParseError("figure_panels must be a list of strings")
        panels = raw

    return ParsedOutput(decision=decision, reasoning=reasoning, figure_panels=panels)


def try_parse_model_output(
    *,
    text: str,
    mode: PromptMode,
) -> tuple[ParsedOutput | None, str | None]:
    """Non-throwing parse helper."""

    try:
        return parse_model_output(text=text, mode=mode), None
    except ParseError as e:
        return None, str(e)
