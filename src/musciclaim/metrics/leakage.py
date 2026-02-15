"""Leakage / contamination audit helpers.

The MuSciClaims plan recommends checking whether evaluation papers overlap with training data.
This module supports an optional audit based on a provided paper_id list.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


def read_id_set(path: Path) -> set[str]:
    """Read a newline-delimited ID file into a set.

    Lines starting with '#' and empty lines are ignored.
    """

    ids: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        ids.add(s)
    return ids


@dataclass(frozen=True)
class OverlapStats:
    """Overlap summary for a set of records.

    What it does:
        Stores counts used for a simple leakage audit based on `paper_id` overlap.

    Why it exists:
        Overlap between training and evaluation papers can inflate reported metrics.
    """

    n_total: int
    n_known: int
    n_overlap: int

    @property
    def overlap_rate_known(self) -> float:
        """Compute the overlap rate among records with known IDs.

        What it does:
            Returns `n_overlap / n_known`, using 0.0 when `n_known == 0`.

        Why it exists:
            Missing `paper_id` values should not dilute the overlap rate.
        """

        return (self.n_overlap / self.n_known) if self.n_known else 0.0


def compute_overlap_stats(*, records: list[dict], training_ids: set[str]) -> OverlapStats:
    """Compute overlap counts between record paper_id values and a training ID set.

    What it does:
        Counts total records, records with a known paper_id, and known records that overlap.

    Why it exists:
        If evaluation papers overlap with training data, reported scores may be inflated.
    """

    n_total = len(records)
    known = [r for r in records if r.get("paper_id")]
    n_known = len(known)
    n_overlap = sum(1 for r in known if str(r.get("paper_id")) in training_ids)
    return OverlapStats(n_total=n_total, n_known=n_known, n_overlap=n_overlap)


def split_by_overlap(
    *,
    records: list[dict],
    training_ids: set[str],
) -> tuple[list[dict], list[dict]]:
    """Split records into (overlap, non_overlap) based on paper_id.

    Records with missing paper_id are excluded from both outputs.
    """

    overlap: list[dict] = []
    non: list[dict] = []

    for r in records:
        pid = r.get("paper_id")
        if not pid:
            continue
        if str(pid) in training_ids:
            overlap.append(r)
        else:
            non.append(r)

    return overlap, non
