#!/usr/bin/env python3
"""Fail fast if required directories are missing a README.md.

This repo relies on folder-level READMEs to keep the codebase understandable.
The check is intentionally simple and opinionated.
"""

from __future__ import annotations

import argparse
from pathlib import Path


DEFAULT_REQUIRED_DIRS = [
    Path("."),
    Path(".github"),
    Path(".github/workflows"),
    Path("docs"),
    Path("configs"),
    Path("scripts"),
    Path("src"),
    Path("src/musciclaim"),
    Path("tests"),
    Path("runs"),
    Path("scores"),
    Path("analysis"),
]


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def check_readmes(repo_root: Path, required_dirs: list[Path]) -> list[str]:
    """Return a list of human-readable problems."""

    problems: list[str] = []

    for rel_dir in required_dirs:
        abs_dir = (repo_root / rel_dir).resolve()
        if not abs_dir.exists() or not abs_dir.is_dir():
            problems.append(f"Missing directory: {rel_dir}")
            continue

        readme = abs_dir / "README.md"
        if not readme.exists():
            problems.append(f"Missing README.md: {rel_dir}/README.md")
            continue

        text = _read_text(readme)
        if "## Contents" not in text:
            problems.append(f"README missing '## Contents' section: {rel_dir}/README.md")

    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description="Check folder README.md presence and structure.")
    parser.add_argument(
        "--repo-root",
        default=".",
        help="Path to the repository root (default: current working directory).",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    problems = check_readmes(repo_root=repo_root, required_dirs=DEFAULT_REQUIRED_DIRS)

    if problems:
        for p in problems:
            print(f"ERROR: {p}")
        print(f"\nFound {len(problems)} problem(s).")
        return 1

    print("OK: README checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
