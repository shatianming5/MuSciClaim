"""MuSciClaim schema package.

Re-exports all public symbols so existing ``from musciclaim.schema import X``
statements continue to work after the module-to-package conversion.
"""

from musciclaim.schema.dataclasses import (
    EvalExample,
    GenerationResult,
    ImageMeta,
    ParsedOutput,
    PredictionRecord,
)
from musciclaim.schema.enums import Condition, Decision, PromptMode

__all__ = [
    "Condition",
    "Decision",
    "EvalExample",
    "GenerationResult",
    "ImageMeta",
    "ParsedOutput",
    "PredictionRecord",
    "PromptMode",
]
