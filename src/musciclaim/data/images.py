"""Image retrieval and minimal preprocessing.

Key invariants:
- No cropping (multi-panel figures must remain intact for localization).
- Resize (if any) must preserve aspect ratio and be logged.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from musciclaim.config import PreprocessingConfig
from musciclaim.schema import ImageMeta


@dataclass(frozen=True)
class ImageLoadResult:
    """Result of attempting to load a figure image.

    What it does:
        Carries either a loaded image plus preprocessing metadata, or an error string.

    Why it exists:
        Image failures must be surfaced as structured signals so the runner can record them and
        metrics can account for them.
    """

    image: Any | None  # PIL.Image.Image, but kept as Any to avoid hard dependency in type checking
    meta: ImageMeta
    error: str | None


def _require_pillow():
    try:
        from PIL import Image  # noqa: F401
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Pillow is required. Install with: pip install -e '.[hf]'") from e


def download_hf_dataset_file(
    *,
    repo_id: str,
    filename: str,
    revision: str | None,
    cache_dir: str,
) -> Path:
    """Download a dataset file from the HF Hub and return its local path."""

    try:
        from huggingface_hub import hf_hub_download
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "huggingface_hub is required. Install with: pip install -e '.[hf]'"
        ) from e

    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    local_path = hf_hub_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        filename=filename,
        cache_dir=cache_dir,
    )
    return Path(local_path)


def open_and_preprocess_image(*, path: Path, preprocessing: PreprocessingConfig) -> ImageLoadResult:
    """Open an image file and apply minimal preprocessing."""

    _require_pillow()
    from PIL import Image

    orig_w = orig_h = new_w = new_h = None
    resize_ratio: float | None = None

    try:
        img = Image.open(path)
        orig_w, orig_h = img.size

        # Ensure RGB for model processors.
        if img.mode != "RGB":
            img = img.convert("RGB")

        if preprocessing.resize_max_side:
            max_side = int(preprocessing.resize_max_side)
            w, h = img.size
            if max(w, h) > max_side:
                ratio = max_side / float(max(w, h))
                new_w = int(round(w * ratio))
                new_h = int(round(h * ratio))
                img = img.resize((new_w, new_h), resample=Image.Resampling.BICUBIC)
                resize_ratio = ratio
            else:
                new_w, new_h = w, h
                resize_ratio = 1.0
        else:
            new_w, new_h = img.size
            resize_ratio = 1.0

        meta = ImageMeta(
            filepath=str(path),
            orig_w=orig_w,
            orig_h=orig_h,
            new_w=new_w,
            new_h=new_h,
            resize_ratio=resize_ratio,
        )
        return ImageLoadResult(image=img, meta=meta, error=None)
    except Exception as e:
        meta = ImageMeta(
            filepath=str(path),
            orig_w=orig_w,
            orig_h=orig_h,
            new_w=new_w,
            new_h=new_h,
            resize_ratio=resize_ratio,
        )
        return ImageLoadResult(image=None, meta=meta, error=str(e))
