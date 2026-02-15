#!/usr/bin/env python3
"""Fail fast if public classes/methods are missing required docstring markers.

This repo relies on clear English docstrings to keep the codebase understandable and auditable.
Public classes and their public methods must include both:
- "What it does:"
- "Why it exists:"
"""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
from pathlib import Path

REQUIRED_MARKERS = ("What it does:", "Why it exists:")


@dataclass(frozen=True)
class Problem:
    """A single docstring check finding.

    What it does:
        Captures the location and reason for a docstring/marker violation.

    Why it exists:
        Produces stable, actionable error output for CI and local checks.
    """

    path: Path
    lineno: int
    symbol: str
    message: str


def _has_markers(doc: str) -> bool:
    return all(m in doc for m in REQUIRED_MARKERS)


def _is_public(name: str) -> bool:
    return not name.startswith("_")


def _is_dunder(name: str) -> bool:
    return name.startswith("__") and name.endswith("__")


def check_docstrings(*, package_root: Path) -> list[Problem]:
    """Check public classes and public methods for required docstring markers."""

    problems: list[Problem] = []

    for py in sorted(package_root.rglob("*.py")):
        if py.name == "__init__.py":
            continue
        if "__pycache__" in py.parts:
            continue

        try:
            mod = ast.parse(py.read_text(encoding="utf-8"), filename=str(py))
        except SyntaxError as e:
            lineno = int(getattr(e, "lineno", 1) or 1)
            problems.append(
                Problem(
                    path=py,
                    lineno=lineno,
                    symbol=py.as_posix(),
                    message=f"SyntaxError while parsing: {e.msg}",
                )
            )
            continue

        for node in mod.body:
            if not isinstance(node, ast.ClassDef):
                continue
            if not _is_public(node.name):
                continue

            cdoc = ast.get_docstring(node) or ""
            if not cdoc:
                problems.append(
                    Problem(path=py, lineno=node.lineno, symbol=node.name, message="Missing class docstring")
                )
            elif not _has_markers(cdoc):
                problems.append(
                    Problem(
                        path=py,
                        lineno=node.lineno,
                        symbol=node.name,
                        message="Class docstring missing required markers",
                    )
                )

            for item in node.body:
                if not isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                if _is_dunder(item.name) or not _is_public(item.name):
                    continue

                fdoc = ast.get_docstring(item) or ""
                fq = f"{node.name}.{item.name}"
                if not fdoc:
                    problems.append(
                        Problem(path=py, lineno=item.lineno, symbol=fq, message="Missing method docstring")
                    )
                elif not _has_markers(fdoc):
                    problems.append(
                        Problem(
                            path=py,
                            lineno=item.lineno,
                            symbol=fq,
                            message="Method docstring missing required markers",
                        )
                    )

    return problems


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check required docstring markers for public classes and public methods."
    )
    parser.add_argument(
        "--repo-root",
        default=".",
        help="Path to the repository root (default: current working directory).",
    )
    parser.add_argument(
        "--package-root",
        default="src/musciclaim",
        help="Package root to scan (default: src/musciclaim).",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    package_root = (repo_root / args.package_root).resolve()
    if not package_root.exists():
        print(f"ERROR: Package root does not exist: {package_root}")
        return 1

    problems = check_docstrings(package_root=package_root)
    if problems:
        for p in problems:
            rel = p.path
            if p.path.is_relative_to(repo_root):
                rel = p.path.relative_to(repo_root)
            print(f"ERROR: {rel}:{p.lineno} {p.symbol}: {p.message}")
        print(f"\nFound {len(problems)} problem(s).")
        return 1

    print("OK: Docstring checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
