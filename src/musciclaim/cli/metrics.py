"""CLI: recompute metrics from existing predictions."""

from __future__ import annotations

import argparse
from pathlib import Path

from musciclaim.metrics.aggregate import aggregate_run


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate predictions into summary.csv.")
    parser.add_argument("--run-id", required=True, help="Run id under runs/")
    parser.add_argument(
        "--repo-root",
        default=".",
        help="Repository root (default: current directory).",
    )
    parser.add_argument("--runs-root", default="runs", help="Runs root directory (default: runs)")
    parser.add_argument(
        "--scores-root",
        default="scores",
        help="Scores root directory (default: scores)",
    )
    parser.add_argument(
        "--training-paper-ids-file",
        default=None,
        help="Optional newline-delimited paper_id list for leakage auditing.",
    )
    parser.add_argument(
        "--bootstrap-iters",
        type=int,
        default=2000,
        help="Paired bootstrap iterations for Ours-vs-Base significance (default: 2000).",
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=1337,
        help="Paired bootstrap seed for Ours-vs-Base significance (default: 1337).",
    )

    args = parser.parse_args()

    run_id = args.run_id
    repo_root = Path(args.repo_root)

    summary = aggregate_run(
        run_id=run_id,
        runs_root=(repo_root / args.runs_root),
        scores_root=(repo_root / args.scores_root),
        training_paper_ids_file=Path(args.training_paper_ids_file)
        if args.training_paper_ids_file
        else None,
        bootstrap_iters=args.bootstrap_iters,
        bootstrap_seed=args.bootstrap_seed,
    )

    print(f"wrote: {summary}")
