"""
preprocess.py — Robust image preprocessing pipeline for solar panel fault detection.

Handles any input image regardless of:
  - Size / aspect ratio  (resizes to 224×224)
  - Mode                 (RGB, RGBA, L/grayscale, P/palette, CMYK, I/32-bit, F/float, YCbCr, LAB, HSV)
  - Bit depth            (8-bit, 16-bit TIFF, 32-bit float)
  - File format          (JPEG, PNG, BMP, TIFF, WebP, GIF-first-frame)

Public API
----------
    load_and_preprocess(image_path)  -> PIL.Image (RGB, 224×224)
    to_tensor(pil_image)             -> torch.Tensor (1, 3, 224, 224), normalised
    preprocess_for_model(image_path) -> torch.Tensor  (combines both steps)
    get_image_info(image_path)       -> dict with diagnostics
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Union

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

# ── Constants ────────────────────────────────────────────────────────────────

TARGET_SIZE = (224, 224)          # (width, height) — ResNet-18 standard input
NORM_MEAN   = [0.5, 0.5, 0.5]    # must match train.py / predict.py
NORM_STD    = [0.5, 0.5, 0.5]

SUPPORTED_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp",
    ".tif", ".tiff", ".webp", ".gif",
}

# Modes PIL can safely convert to RGB directly via .convert("RGB")
_DIRECT_TO_RGB = {"RGB", "RGBA", "L", "P", "1", "YCbCr", "LAB", "HSV"}

# ── Validation ────────────────────────────────────────────────────────────────

def validate_image_path(image_path: Union[str, Path]) -> Path:
    """
    Check that the path exists, is a file, and has a recognised image extension.
    Raises FileNotFoundError or ValueError with a clear message on failure.
    """
    path = Path(image_path)

    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")

    if not path.is_file():
        raise ValueError(f"Path is not a file: {path}")

    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file extension '{path.suffix}'. "
            f"Supported: {sorted(SUPPORTED_EXTENSIONS)}"
        )

    return path


# ── Core conversion ───────────────────────────────────────────────────────────

def _normalise_bit_depth(img: Image.Image) -> Image.Image:
    """
    Normalise 16-bit (I) and 32-bit float (F) images to 8-bit (L)
    by stretching the value range to [0, 255].

    16-bit thermal TIFFs from infrared cameras are the primary motivation.
    """
    if img.mode not in ("I", "F"):
        return img

    arr = np.array(img, dtype=np.float32)
    lo, hi = arr.min(), arr.max()

    if hi > lo:
        arr = (arr - lo) / (hi - lo) * 255.0
    else:
        arr = np.zeros_like(arr)

    return Image.fromarray(arr.astype(np.uint8), mode="L")


def _to_rgb(img: Image.Image) -> Image.Image:
    """
    Convert any PIL image mode to 8-bit RGB.

    Conversion map
    ──────────────
    RGB       → keep as-is
    RGBA      → flatten alpha onto white background, then RGB
    L         → replicate channel 3× → RGB  (grayscale thermal → RGB)
    P         → expand palette → RGB
    1         → boolean → L → RGB
    CMYK      → PIL built-in → RGB
    I / F     → normalise to L first (see above) → RGB
    YCbCr,
    LAB, HSV  → PIL built-in → RGB
    other     → best-effort via PIL
    """
    if img.mode == "RGB":
        return img

    # High bit-depth modes → 8-bit grayscale first
    if img.mode in ("I", "F"):
        img = _normalise_bit_depth(img)         # → "L"

    # RGBA → white-background composite
    if img.mode == "RGBA":
        background = Image.new("RGB", img.size, (255, 255, 255))
        # paste using alpha channel as mask
        background.paste(img, mask=img.split()[3])
        return background

    # Everything else (L, P, 1, CMYK, YCbCr, LAB, HSV, …)
    return img.convert("RGB")


def _resize(img: Image.Image, size: tuple[int, int] = TARGET_SIZE) -> Image.Image:
    """
    Resize to exactly (width, height) using high-quality Lanczos resampling.
    Handles any input size — upscaling and downscaling both work correctly.
    """
    if img.size == size:
        return img
    return img.resize(size, resample=Image.LANCZOS)


# ── Public API ────────────────────────────────────────────────────────────────

def load_and_preprocess(image_path: Union[str, Path]) -> Image.Image:
    """
    Load an image from disk and return a PIL Image that is:
      - Mode  : RGB (3-channel, 8-bit per channel)
      - Size  : 224 × 224 pixels

    Handles all common image modes and bit-depths automatically.

    Parameters
    ----------
    image_path : str or Path
        Path to the input image (any supported format).

    Returns
    -------
    PIL.Image.Image
        Ready-to-use RGB image at 224 × 224.

    Raises
    ------
    FileNotFoundError  – if the file does not exist.
    ValueError         – if the extension is unsupported or image is corrupt.
    """
    path = validate_image_path(image_path)

    try:
        with Image.open(path) as raw:
            # GIF / animated formats → use first frame only
            if hasattr(raw, "n_frames") and raw.n_frames > 1:
                raw.seek(0)

            # Copy out of the context manager so callers can use it freely
            img = raw.copy()
    except Exception as exc:
        raise ValueError(f"Cannot open image '{path}': {exc}") from exc

    img = _to_rgb(img)
    img = _resize(img)
    return img                 # PIL Image, RGB, 224×224


_tensor_transform = transforms.Compose(
    [
        transforms.ToTensor(),                             # [0,255] uint8 → [0,1] float32
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),  # → roughly [-1, 1]
    ]
)


def to_tensor(pil_image: Image.Image) -> torch.Tensor:
    """
    Convert a preprocessed PIL Image (RGB, 224×224) to a model-ready tensor.

    Parameters
    ----------
    pil_image : PIL.Image.Image
        Must be RGB mode (output of load_and_preprocess works directly).

    Returns
    -------
    torch.Tensor
        Shape (1, 3, 224, 224), dtype float32, values ≈ [−1, 1].
    """
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")

    return _tensor_transform(pil_image).unsqueeze(0)   # add batch dim


def preprocess_for_model(
    image_path: Union[str, Path],
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Full pipeline: file path → model-ready tensor in one call.

    Parameters
    ----------
    image_path : str or Path
        Path to any supported image file.
    device : torch.device, optional
        Target device. Defaults to CPU.

    Returns
    -------
    torch.Tensor
        Shape (1, 3, 224, 224) on the requested device.
    """
    if device is None:
        device = torch.device("cpu")

    pil_img = load_and_preprocess(image_path)
    tensor  = to_tensor(pil_img)
    return tensor.to(device)


def get_image_info(image_path: Union[str, Path]) -> Dict:
    """
    Return a diagnostic dictionary describing the raw image before any conversion.
    Useful for debugging and logging.

    Returns
    -------
    dict with keys:
        path, format, original_size, original_mode,
        needs_resize, needs_mode_conversion, file_size_kb
    """
    path = validate_image_path(image_path)

    try:
        with Image.open(path) as img:
            info = {
                "path":                 str(path.resolve()),
                "format":               img.format or path.suffix.upper().lstrip("."),
                "original_size":        img.size,         # (width, height)
                "original_mode":        img.mode,
                "needs_resize":         img.size != TARGET_SIZE,
                "needs_mode_conversion": img.mode != "RGB",
                "file_size_kb":         round(path.stat().st_size / 1024, 2),
            }
    except Exception as exc:
        raise ValueError(f"Cannot inspect image '{path}': {exc}") from exc

    return info
