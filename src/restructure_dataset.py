"""
restructure_dataset.py

BUG FIX vs original:
  save_resized() previously converted images to grayscale ("L" mode) before
  saving.  train.py loads images with ImageFolder and normalises with
  mean=[0.5,0.5,0.5] / std=[0.5,0.5,0.5] — a 3-channel (RGB) normalisation.
  Grayscale images have only 1 channel; torchvision's ImageFolder auto-converts
  them to RGB at load time, but this round-trip through JPEG grayscale can
  discard information and causes subtle inconsistencies between what was stored
  and what the model actually sees.

  Fix: save_resized() now uses the shared preprocess pipeline (load_and_preprocess)
  which always outputs 224×224 RGB, matching exactly what train.py and predict.py
  expect.
"""

import argparse
import shutil
from pathlib import Path
from typing import Dict, List, Tuple
import random

# Re-use the shared preprocessing pipeline so train / restructure / predict
# all apply identical resize + mode conversion logic.
from preprocess import load_and_preprocess

# ── Constants ─────────────────────────────────────────────────────────────────

TARGET_CLASSES = ["normal", "hotspot", "diode", "crack", "shadowing", "soiling"]

CLASS_MAPPING = {
    "No-Anomaly":     "normal",
    "Hot-Spot":       "hotspot",
    "Hot-Spot-Multi": "hotspot",
    "Diode":          "diode",
    "Diode-Multi":    "diode",
    "Cracking":       "crack",
    "Shadowing":      "shadowing",
    "Soiling":        "soiling",
    "Vegetation":     "soiling",   # optional mapping choice
    "Offline-Module": None,        # ignore
    "Cell":           None,        # ignore
    "Cell-Multi":     None,        # ignore
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def safe_list_images(folder: Path) -> List[Path]:
    if not folder.exists() or not folder.is_dir():
        return []
    valid_ext = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    return [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in valid_ext]


def split_items(
    items: List[Path], train_ratio: float, val_ratio: float, seed: int
) -> Tuple[List[Path], List[Path], List[Path]]:
    rng = random.Random(seed)
    copied = items[:]
    rng.shuffle(copied)

    n = len(copied)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)

    if n >= 3:
        if n_train == 0: n_train = 1
        if n_val   == 0: n_val   = 1
        n_test = n - n_train - n_val
        if n_test == 0:
            n_test = 1
            if n_train > n_val and n_train > 1:
                n_train -= 1
            elif n_val > 1:
                n_val -= 1
    else:
        n_test = n - n_train - n_val

    train_items = copied[:n_train]
    val_items   = copied[n_train : n_train + n_val]
    test_items  = copied[n_train + n_val :]
    return train_items, val_items, test_items


def save_preprocessed(src_path: Path, dst_path: Path) -> bool:
    """
    Load image via the shared pipeline (any size/mode → RGB 224×224)
    and save as JPEG to the destination path.

    FIX: removed the `.convert("L")` call that was converting to grayscale.
    """
    try:
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        pil_img = load_and_preprocess(src_path)   # → RGB, 224×224
        # Save as PNG to avoid repeated JPEG compression artefacts.
        pil_img.save(dst_path.with_suffix(".png"))
        return True
    except Exception:
        return False


def collect_grouped_images(split_root: Path) -> Dict[str, List[Path]]:
    grouped: Dict[str, List[Path]] = {c: [] for c in TARGET_CLASSES}

    if not split_root.exists() or not split_root.is_dir():
        print(f"Warning: folder not found, skipping: {split_root}")
        return grouped

    for source_class, target_class in CLASS_MAPPING.items():
        if target_class is None:
            continue
        source_folder = split_root / source_class
        grouped[target_class].extend(safe_list_images(source_folder))

    return grouped


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Restructure folder-based solar thermal dataset."
    )
    parser.add_argument("--data-root",       type=str, default="data",
                        help="Root folder containing train/ and test/")
    parser.add_argument("--processed-root",  type=str, default="data/processed",
                        help="Output processed dataset folder")
    parser.add_argument("--seed",            type=int, default=42,
                        help="Random seed for reproducible split")
    args = parser.parse_args()

    data_root      = Path(args.data_root)
    processed_root = Path(args.processed_root)

    if processed_root.exists():
        shutil.rmtree(processed_root)

    for split_name in ["train", "val", "test"]:
        for cls in TARGET_CLASSES:
            (processed_root / split_name / cls).mkdir(parents=True, exist_ok=True)

    train_grouped = collect_grouped_images(data_root / "train")
    test_grouped  = collect_grouped_images(data_root / "test")

    saved_counts = {
        split: {c: 0 for c in TARGET_CLASSES}
        for split in ["train", "val", "test"]
    }
    skipped_bad = 0

    for cls in TARGET_CLASSES:
        tr_items, va_items, te_from_train = split_items(
            train_grouped[cls], train_ratio=0.7, val_ratio=0.15, seed=args.seed
        )
        final_test_items = te_from_train + test_grouped[cls]

        for i, src in enumerate(tr_items):
            dst = processed_root / "train" / cls / f"{src.stem}_tr_{i}"
            if save_preprocessed(src, dst): saved_counts["train"][cls] += 1
            else: skipped_bad += 1

        for i, src in enumerate(va_items):
            dst = processed_root / "val" / cls / f"{src.stem}_va_{i}"
            if save_preprocessed(src, dst): saved_counts["val"][cls] += 1
            else: skipped_bad += 1

        for i, src in enumerate(final_test_items):
            dst = processed_root / "test" / cls / f"{src.stem}_te_{i}"
            if save_preprocessed(src, dst): saved_counts["test"][cls] += 1
            else: skipped_bad += 1

    print("\nProcessed dataset created successfully.")
    print(f"Output: {processed_root.resolve()}")
    print("\nImage counts per class (all saved as RGB 224×224 PNG):")
    total_used = 0
    for cls in TARGET_CLASSES:
        c_tr = saved_counts["train"][cls]
        c_va = saved_counts["val"][cls]
        c_te = saved_counts["test"][cls]
        total_used += c_tr + c_va + c_te
        print(f"  {cls:10s} -> train: {c_tr:4d}, val: {c_va:4d}, test: {c_te:4d}, "
              f"total: {c_tr+c_va+c_te:4d}")

    print(f"\nTotal images used   : {total_used}")
    print(f"Skipped (unreadable): {skipped_bad}")
    print("\nIgnored source classes : Offline-Module, Cell, Cell-Multi")
    print("Vegetation is mapped to soiling.")
    print("NOTE: Images are now saved as RGB (not grayscale) to match train.py expectations.")


if __name__ == "__main__":
    main()
