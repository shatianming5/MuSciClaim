"""Panel label normalization and validation.

MuSciClaims localization uses panel labels (e.g. "Panel A"). Model outputs are often messy
("A", "panel a", etc.), so we normalize them into a canonical form.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_CANON_RE = re.compile(r"^(?:PANEL\s*)?([A-Z])$")


def normalize_panel_label(raw: str) -> str | None:
    """Normalize a single panel label to the canonical form "Panel X".

    What it does:
        Converts common variants ("A", "panel a", "PanelA") into "Panel A".

    Why it exists:
        Localization evaluation should not be dominated by formatting noise.
    """

    s = (raw or "").strip()
    if not s:
        return None

    s = s.replace("_", " ").strip().upper()
    s = s.replace("PANEL", "PANEL ").strip()
    s = re.sub(r"\s+", " ", s)

    m = _CANON_RE.match(s)
    if not m:
        return None

    return f"Panel {m.group(1)}"


def normalize_panel_list(raw_panels: list[str] | None) -> tuple[list[str], bool]:
    """Normalize a list of raw panel labels.

    Returns:
        (normalized_panels, invalid)

    invalid=True means at least one raw label could not be normalized.
    """

    if not raw_panels:
        return [], False

    out: list[str] = []
    invalid = False

    for p in raw_panels:
        norm = normalize_panel_label(p)
        if norm is None:
            invalid = True
            continue
        out.append(norm)

    # Stable unique ordering.
    out = sorted(set(out), key=lambda x: x)
    return out, invalid


@dataclass(frozen=True)
class PanelWhitelist:
    """A whitelist for panel labels.

    What it does:
        Provides a consistent policy for rejecting out-of-range panel labels.

    Why it exists:
        Without a whitelist, models can emit arbitrary strings and still appear "valid".
    """

    allowed: set[str]

    @classmethod
    def az(cls) -> "PanelWhitelist":
        """Build an A-Z whitelist.

        What it does:
            Returns a whitelist containing `Panel A` through `Panel Z`.

        Why it exists:
            Keeps localization outputs bounded to a plausible, dataset-aligned set of labels.
        """

        return cls(allowed={f"Panel {chr(c)}" for c in range(ord("A"), ord("Z") + 1)})

    def validate(self, panels: list[str]) -> bool:
        """Return True if all panels are in the whitelist.

        What it does:
            Checks every normalized panel label against the allowed set.

        Why it exists:
            Prevents arbitrary strings from being treated as valid localization evidence.
        """

        return all(p in self.allowed for p in panels)
