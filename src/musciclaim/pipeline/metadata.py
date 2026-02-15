"""Run metadata capture.

A run should be auditable without re-running:
- environment versions
- config values
- repo state (git commit)
"""

from __future__ import annotations

import dataclasses
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _pkg_version(name: str) -> str | None:
    try:
        import importlib.metadata as im

        return im.version(name)
    except Exception:
        return None


def _git_head(repo_root: Path) -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            stderr=subprocess.DEVNULL,
        )
        return out.decode("utf-8").strip()
    except Exception:
        return None


def _git_dirty(repo_root: Path) -> bool | None:
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=repo_root,
            stderr=subprocess.DEVNULL,
        )
        return bool(out.decode("utf-8").strip())
    except Exception:
        return None


def collect_run_metadata(
    *,
    repo_root: Path,
    run_id: str,
    run_cfg: Any,
    models_cfg: Any,
) -> dict[str, Any]:
    """Collect run metadata for auditability."""

    meta: dict[str, Any] = {
        "run_id": run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": {
            "version": sys.version,
            "executable": sys.executable,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "packages": {
            "numpy": _pkg_version("numpy"),
            "PyYAML": _pkg_version("PyYAML"),
            "tqdm": _pkg_version("tqdm"),
            "datasets": _pkg_version("datasets"),
            "huggingface_hub": _pkg_version("huggingface_hub"),
            "transformers": _pkg_version("transformers"),
            "torch": _pkg_version("torch"),
            "Pillow": _pkg_version("Pillow"),
        },
        "git": {
            "head": _git_head(repo_root),
            "dirty": _git_dirty(repo_root),
        },
        "config": {
            "run": dataclasses.asdict(run_cfg) if dataclasses.is_dataclass(run_cfg) else run_cfg,
            "models": {
                k: dataclasses.asdict(v) if dataclasses.is_dataclass(v) else v
                for k, v in (models_cfg or {}).items()
            },
        },
    }

    # Torch/GPU info (best effort).
    try:
        import torch

        gpu: dict[str, Any] = {
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": getattr(torch.version, "cuda", None),
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        }
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            gpu["devices"] = [
                {
                    "index": i,
                    "name": torch.cuda.get_device_name(i),
                }
                for i in range(torch.cuda.device_count())
            ]
        meta["gpu"] = gpu
    except Exception:
        pass

    return meta


def write_run_metadata(*, path: Path, metadata: dict[str, Any]) -> None:
    """Write metadata JSON to disk."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
