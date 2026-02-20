"""Core enums for MuSciClaim.

What it does:
    Defines the three enum types that govern labels, prompt schemas,
    and input ablation conditions.

Why it exists:
    Prevents stringly-typed drift and keeps parsing/metrics contracts stable.
"""

from __future__ import annotations

from enum import Enum


class Decision(str, Enum):
    """Allowed 3-way decisions.

    What it does:
        Enumerates the only valid labels for MuSciClaims claim verification.

    Why it exists:
        Prevents stringly-typed label drift and keeps parsing/metrics contracts stable.
    """

    SUPPORT = "SUPPORT"
    CONTRADICT = "CONTRADICT"
    NEUTRAL = "NEUTRAL"


class PromptMode(str, Enum):
    """Prompt/output schema variants.

    What it does:
        Declares which strict JSON schema the model must follow (decision-only, reasoning, panels).

    Why it exists:
        Output shape affects parsing, metrics, and localization; it must be explicit and validated.
    """

    D = "D"  # decision-only
    R = "R"  # short reasoning + decision
    PANELS = "PANELS"  # panels + reasoning + decision


class Condition(str, Enum):
    """Input ablation conditions.

    What it does:
        Defines the input ablations used to diagnose cross-modal necessity.

    Why it exists:
        High scores are meaningless if the model can ignore evidence; ablations make this testable.
    """

    FULL = "full"
    C_ONLY = "c_only"
    F_ONLY = "f_only"
    CLAIM_ONLY = "claim_only"
