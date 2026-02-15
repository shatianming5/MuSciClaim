"""Prompt templates for MuSciClaim.

The prompts are intentionally strict: outputs must be valid JSON with fixed keys.
"""

from __future__ import annotations

from musciclaim.schema import PromptMode


def _format_fix_preamble() -> str:
    return (
        "Your previous output was not valid JSON.\n"
        "Output exactly ONE JSON object and NOTHING ELSE.\n"
        "The output must start with { and end with }.\n"
        "Use only the allowed labels: \"SUPPORT\", \"CONTRADICT\", \"NEUTRAL\".\n"
    )


def build_prompt(
    *,
    mode: PromptMode,
    claim_text: str,
    caption_text: str | None,
    figure_provided: bool,
    caption_provided: bool,
    retry: bool,
) -> str:
    """Build a strict JSON prompt for the given prompt mode.

    What it does:
        Produces a self-contained prompt describing the task and schema.

    Why it exists:
        Keeping prompts centralized prevents subtle drift across conditions.
    """

    caption = caption_text if caption_provided else "(not provided)"
    figure = "(provided)" if figure_provided else "(not provided)"

    header = (
        "You are an AI model tasked with verifying claims related to visual evidence "
        "using zero-shot learning.\n"
        "Your job is to analyze the given figure(s) and caption(s) to decide whether they "
        "SUPPORT or CONTRADICT or are NEUTRAL with respect to the claim.\n"
    )

    body = (
        f"CLAIM: {claim_text}\n"
        f"FIGURE: {figure}\n"
        f"IMAGE CAPTION(S): {caption}\n"
    )

    if mode == PromptMode.D:
        schema = (
            "After completing your analysis, output exactly one JSON object with exactly one key: "
            "\"decision\".\n"
            "For \"decision\", output exactly one word: "
            "\"SUPPORT\" or \"CONTRADICT\" or \"NEUTRAL\" (uppercase).\n"
            "The output must start with { and end with }.\n"
        )
    elif mode == PromptMode.R:
        schema = (
            "After completing your analysis, output exactly one JSON object with exactly two keys: "
            "\"reasoning\" and \"decision\".\n"
            "- \"reasoning\": 1-2 sentences grounded in the figure/caption (mention axes/legend/"
            "trends/panels when relevant).\n"
            "- \"decision\": \"SUPPORT\" or \"CONTRADICT\" or \"NEUTRAL\" (uppercase).\n"
            "No extra text.\n"
        )
    elif mode == PromptMode.PANELS:
        schema = (
            "Output exactly one JSON object with exactly three keys: "
            "\"figure_panels\", \"reasoning\", \"decision\".\n"
            "- \"figure_panels\": a list of panel labels you used (e.g., "
            "[\"Panel A\",\"Panel C\"]) or [].\n"
            "- \"reasoning\": 1-2 sentences grounded in those panels.\n"
            "- \"decision\": \"SUPPORT\" or \"CONTRADICT\" or \"NEUTRAL\" (uppercase).\n"
            "No extra text.\n"
        )
    else:
        raise ValueError(f"Unknown prompt mode: {mode}")

    pre = _format_fix_preamble() if retry else ""
    return pre + header + body + schema
