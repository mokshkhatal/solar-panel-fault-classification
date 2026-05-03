import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image


TARGET_CLASSES = ["normal", "hotspot", "diode", "crack", "shadowing", "soiling"]

def map_label_to_target(label: str) -> Optional[str]:
    """
    Map InfraredSolarModules anomaly labels to final target classes.
    Return None for labels that should be ignored.
    """
    norm = label.strip().lower()
    if norm == "no-anomaly":
        return "normal"
    if norm in {"hot-spot", "hot-spot-multi"}:
        return "hotspot"
    if norm in {"diode", "diode-multi"}:
        return "diode"
    if norm == "cracking":
        return "crack"
    if norm in {"cell", "cell-multi"}:
        return "crack"
    if norm == "shadowing":
        return "shadowing"
    if norm == "vegetation":
        return "shadowing"
    if norm == "soiling":
        return "soiling"
    if norm == "offline-module":
        return None
    return None


def split_global(
    items: List[Tuple[Path, str]], train_ratio: float, val_ratio: float
) -> Tuple[List[Tuple[Path, str]], List[Tuple[Path, str]], List[Tuple[Path, str]]]:
    """
    Split already-shuffled items into train/val/test.
    """
    n = len(items)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val
    train_items = items[:n_train]
    val_items = items[n_train : n_train + n_val]
    test_items = items[n_train + n_val : n_train + n_val + n_test]
    return train_items, val_items, test_items


def save_resized_image(src_path: Path, dst_path: Path, size: Tuple[int, int]) -> bool:
    try:
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(src_path) as img:
            img = img.resize(size)
            img.save(dst_path)
        return True
    except Exception:
        return False


def count_by_class(items: List[Tuple[Path, str]]) -> Dict[str, int]:
    out = {c: 0 for c in TARGET_CLASSES}
    for _, cls in items:
        out[cls] += 1
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare balanced dataset splits from InfraredSolarModules JSON.")
    parser.add_argument(
        "--raw-root",
        type=str,
        default="data/raw/InfraredSolarModules",
        help="Raw dataset root containing images/ and module_metadata.json",
    )
    parser.add_argument(
        "--processed-root",
        type=str,
        default="data/processed",
        help="Output folder for processed train/val/test dataset",
    )
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=500,
        help="Maximum number of samples to keep per class for balancing",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    if args.max_per_class <= 0:
        print("Error: --max-per-class must be greater than 0.")
        return

    raw_root = Path(args.raw_root)
    processed_root = Path(args.processed_root)
    images_root = raw_root / "images"
    metadata_path = raw_root / "module_metadata.json"

    if not metadata_path.exists():
        print(f"Error: metadata file not found: {metadata_path}")
        return
    if not images_root.exists() or not images_root.is_dir():
        print(f"Error: images folder not found: {images_root}")
        return

    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    # 1) Load JSON and group images by mapped class.
    grouped_all: Dict[str, List[Path]] = {c: [] for c in TARGET_CLASSES}
    skipped_unmapped = 0
    skipped_missing = 0
    for _, item in metadata.items():
        image_rel = item.get("image_filepath", "")
        raw_label = item.get("anomaly_class", "")
        mapped = map_label_to_target(raw_label)
        if mapped is None:
            skipped_unmapped += 1
            continue
        img_path = raw_root / image_rel
        if not img_path.exists():
            skipped_missing += 1
            continue
        grouped_all[mapped].append(img_path)

    total_before = sum(len(v) for v in grouped_all.values())
    if total_before == 0:
        print("No usable images found after mapping.")
        return

    rng = random.Random(args.seed)

    # 2) Apply per-class balancing limit.
    # This prevents classes like 'normal' from dominating training.
    balanced_items: List[Tuple[Path, str]] = []
    balanced_counts: Dict[str, int] = {c: 0 for c in TARGET_CLASSES}
    for cls in TARGET_CLASSES:
        cls_items = grouped_all[cls][:]
        rng.shuffle(cls_items)
        if len(cls_items) > args.max_per_class:
            cls_items = cls_items[: args.max_per_class]
        for p in cls_items:
            balanced_items.append((p, cls))
        balanced_counts[cls] = len(cls_items)

    # 3) Merge all classes and 4) shuffle globally before split.
    rng.shuffle(balanced_items)

    # 5) Global 70/15/15 split.
    train_items, val_items, test_items = split_global(
        balanced_items, train_ratio=0.7, val_ratio=0.15
    )

    if processed_root.exists():
        shutil.rmtree(processed_root)

    saved = {"train": 0, "val": 0, "test": 0}
    skipped_bad_images = 0
    seen_name_counts: Dict[str, int] = {}

    for split_name, split_items in [
        ("train", train_items),
        ("val", val_items),
        ("test", test_items),
    ]:
        for src_path, cls_name in split_items:
            # Unique filename to avoid collisions when same names exist across folders.
            key = f"{cls_name}_{src_path.stem}"
            count = seen_name_counts.get(key, 0)
            seen_name_counts[key] = count + 1
            new_name = f"{src_path.stem}_{count}{src_path.suffix.lower()}"

            dst_path = processed_root / split_name / cls_name / new_name
            ok = save_resized_image(src_path, dst_path, size=(224, 224))
            if ok:
                saved[split_name] += 1
            else:
                skipped_bad_images += 1

    # Stats: selected balanced distribution by class.
    print("\nFinal dataset distribution (after balancing limit):")
    total_balanced = 0
    for cls in TARGET_CLASSES:
        print(f"{cls}: {balanced_counts[cls]}")
        total_balanced += balanced_counts[cls]
    print(f"Total images selected: {total_balanced}")

    # Stats: split-level class distribution from selected items.
    train_counts = count_by_class(train_items)
    val_counts = count_by_class(val_items)
    test_counts = count_by_class(test_items)

    print("\nSplit distribution:")
    for cls in TARGET_CLASSES:
        print(
            f"{cls}: train={train_counts[cls]}, val={val_counts[cls]}, test={test_counts[cls]}"
        )

    print("\nProcessed dataset created:")
    print(f"train: {saved['train']}")
    print(f"val: {saved['val']}")
    print(f"test: {saved['test']}")
    print(f"Skipped unmapped/ignored labels: {skipped_unmapped}")
    print(f"Skipped missing image files: {skipped_missing}")
    print(f"Skipped unreadable images: {skipped_bad_images}")
    print(f"Output folder: {processed_root.resolve()}")


if __name__ == "__main__":
    main()
