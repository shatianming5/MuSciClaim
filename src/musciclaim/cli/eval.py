"""CLI: run evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path

from musciclaim.config import load_models_config, load_run_config
from musciclaim.pipeline.runner import run_evaluation


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MuSciClaims evaluation and write artifacts.")
    parser.add_argument("--run-config", required=True, help="Path to configs/run.yaml")
    parser.add_argument("--models-config", required=True, help="Path to configs/models.yaml")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of examples (for smoke runs).",
    )
    parser.add_argument("--run-id", default=None, help="Override run_id (default: UTC timestamp).")
    parser.add_argument(
        "--repo-root",
        default=".",
        help="Repository root (default: current directory).",
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

    run_cfg = load_run_config(args.run_config)
    models_cfg = load_models_config(args.models_config)

    resolved = run_evaluation(
        run_cfg=run_cfg,
        models_cfg=models_cfg,
        limit=args.limit,
        run_id=args.run_id,
        training_paper_ids_file=Path(args.training_paper_ids_file)
        if args.training_paper_ids_file
        else None,
        bootstrap_iters=args.bootstrap_iters,
        bootstrap_seed=args.bootstrap_seed,
        repo_root=Path(args.repo_root),
    )

    print(f"run_id: {resolved}")
    print(f"runs: {Path(args.repo_root) / run_cfg.io.out_root / resolved}")
    print(f"scores: {Path(args.repo_root) / run_cfg.io.scores_root / resolved / 'summary.csv'}")
    print(f"analysis: {Path(args.repo_root) / run_cfg.io.analysis_root / resolved}")
