"""CLI: README enforcement.

This is the installed entrypoint version of `scripts/check_readmes.py`.
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


def check_readmes(repo_root: Path) -> list[str]:
    problems: list[str] = []

    for rel_dir in DEFAULT_REQUIRED_DIRS:
        abs_dir = (repo_root / rel_dir).resolve()
        if not abs_dir.exists() or not abs_dir.is_dir():
            problems.append(f"Missing directory: {rel_dir}")
            continue

        readme = abs_dir / "README.md"
        if not readme.exists():
            problems.append(f"Missing README.md: {rel_dir}/README.md")
            continue

        if "## Contents" not in _read_text(readme):
            problems.append(f"README missing '## Contents' section: {rel_dir}/README.md")

    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description="Check folder README.md presence and structure.")
    parser.add_argument(
        "--repo-root",
        default=".",
        help="Path to the repository root (default: current working directory).",
    )
    args = parser.parse_args()

    problems = check_readmes(repo_root=Path(args.repo_root))
    if problems:
        for p in problems:
            print(f"ERROR: {p}")
        raise SystemExit(1)

    print("OK: README checks passed.")
