"""Paired bootstrap utilities.

These helpers support statistical comparisons between two models evaluated on the same examples.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class BootstrapCI:
    """A bootstrap confidence interval for a scalar metric difference.

    What it does:
        Stores a point estimate and a 95% paired bootstrap interval for `(metric(a) - metric(b))`.

    Why it exists:
        Keeps statistical comparisons explicit and machine-checkable in score artifacts.
    """

    diff: float
    ci95_low: float
    ci95_high: float
    iters: int
    seed: int
    n: int


def _quantile_sorted(sorted_vals: list[float], q: float) -> float:
    """Compute a quantile for a sorted list using linear interpolation."""

    if not sorted_vals:
        return 0.0

    if q <= 0.0:
        return float(sorted_vals[0])
    if q >= 1.0:
        return float(sorted_vals[-1])

    n = len(sorted_vals)
    idx = q * (n - 1)
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return float(sorted_vals[lo])

    frac = idx - lo
    return float(sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac)


def paired_bootstrap_ci_diff(
    *,
    gold: list[str],
    pred_a: list[str | None],
    pred_b: list[str | None],
    metric_fn: Callable[[list[str], list[str | None]], float],
    iters: int,
    seed: int,
) -> BootstrapCI:
    """Compute paired bootstrap CI for (metric(a) - metric(b)).

    What it does:
        Resamples example indices with replacement, computing the metric difference each time.

    Why it exists:
        Avoids "looks better" claims by reporting a paired uncertainty interval.
        Computed on the same test set using a paired resampling scheme.
    """

    if not (len(gold) == len(pred_a) == len(pred_b)):
        raise ValueError("gold/pred length mismatch")

    n = len(gold)
    if n == 0:
        return BootstrapCI(diff=0.0, ci95_low=0.0, ci95_high=0.0, iters=iters, seed=seed, n=0)

    rng = random.Random(seed)

    diffs: list[float] = []
    for _ in range(int(iters)):
        idxs = [rng.randrange(n) for _ in range(n)]

        g = [gold[i] for i in idxs]
        a = [pred_a[i] for i in idxs]
        b = [pred_b[i] for i in idxs]

        diffs.append(float(metric_fn(g, a) - metric_fn(g, b)))

    diffs.sort()

    diff_point = float(metric_fn(gold, pred_a) - metric_fn(gold, pred_b))
    lo = _quantile_sorted(diffs, 0.025)
    hi = _quantile_sorted(diffs, 0.975)

    return BootstrapCI(
        diff=diff_point,
        ci95_low=lo,
        ci95_high=hi,
        iters=int(iters),
        seed=int(seed),
        n=n,
    )
