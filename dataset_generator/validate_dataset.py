"""
validate_dataset.py

Dataset validation script for the DriftSense-X project.
Inspects all three dataset splits (train, validation, test) and verifies:
1. Reference image exists and is readable.
2. Search image exists and is readable.
3. Reference image resolution is 1000x1000.
4. Search image resolution is 1000x1000.
5. labels.csv exists in each split folder.
6. Every image has a corresponding label.
7. Ground-truth x coordinate is numeric and within search image bounds (0 <= x <= 1000).
8. Ground-truth y coordinate is numeric and within search image bounds (0 <= y <= 1000).
9. Coordinates are numeric.
10. No duplicate or missing filenames.
11. Exactly 1000 train pairs, 200 validation pairs, and 200 test pairs (total 1400 expected).
12. Reports architecture distribution (DRAM / FinFET).
"""

import os
import csv
import sys
from collections import Counter
from PIL import Image


def find_dataset_dir() -> str:
    """Finds the root 'dataset' directory regardless of execution context."""
    cwd = os.getcwd()
    if os.path.exists(os.path.join(cwd, "dataset")):
        return os.path.abspath(os.path.join(cwd, "dataset"))
    elif os.path.exists(os.path.join(cwd, "..", "dataset")):
        return os.path.abspath(os.path.join(cwd, "..", "dataset"))
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(script_dir)
        return os.path.abspath(os.path.join(parent_dir, "dataset"))


def validate_split(split_name: str, expected_count: int, dataset_dir: str):
    """
    Validates a single dataset split.

    Returns:
        dict containing split statistics and pass/fail boolean.
    """
    split_dir = os.path.join(dataset_dir, split_name)
    ref_dir = os.path.join(split_dir, "reference")
    search_dir = os.path.join(split_dir, "search")
    csv_path = os.path.join(split_dir, "labels.csv")

    stats = {
        "split": split_name.upper(),
        "expected_pairs": expected_count,
        "pairs_count": 0,
        "images_valid": 0,
        "labels_count": 0,
        "invalid_count": 0,
        "arch_counts": Counter(),
        "errors": []
    }

    # 1. Verify labels.csv exists
    if not os.path.exists(csv_path):
        stats["errors"].append(f"labels.csv does not exist in {split_dir}")
        stats["invalid_count"] = expected_count
        return stats

    # Read labels.csv
    labels_records = []
    seen_filenames = set()
    duplicate_filenames = set()

    with open(csv_path, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row_idx, row in enumerate(reader, start=2):
            img_name = row.get('image', '').strip()
            x_str = row.get('x', '').strip()
            y_str = row.get('y', '').strip()
            style = row.get('style', 'Unknown').strip()

            if not img_name:
                stats["errors"].append(f"Row {row_idx}: Empty image filename in labels.csv")
                stats["invalid_count"] += 1
                continue

            if img_name in seen_filenames:
                duplicate_filenames.add(img_name)
            seen_filenames.add(img_name)

            labels_records.append((img_name, x_str, y_str, style, row_idx))

    stats["labels_count"] = len(labels_records)

    if duplicate_filenames:
        stats["errors"].append(f"Duplicate filenames in labels.csv: {sorted(list(duplicate_filenames))}")

    # Check for missing expected files (00001.png to N.png)
    for i in range(1, expected_count + 1):
        fname = f"{i:05d}.png"
        if fname not in seen_filenames:
            stats["errors"].append(f"Missing expected filename in labels.csv: {fname}")

    # Inspect images present on disk
    ref_files = set(os.listdir(ref_dir)) if os.path.exists(ref_dir) else set()
    search_files = set(os.listdir(search_dir)) if os.path.exists(search_dir) else set()

    valid_pairs = 0

    for img_name, x_str, y_str, style, row_idx in labels_records:
        pair_valid = True

        # Check numeric coordinates
        try:
            x_val = float(x_str)
            y_val = float(y_str)
        except ValueError:
            stats["errors"].append(f"Row {row_idx} ({img_name}): Non-numeric coordinates (x={x_str}, y={y_str})")
            pair_valid = False
            x_val, y_val = None, None

        # Check coordinate bounds (0 <= x, y <= 1000)
        if x_val is not None and y_val is not None:
            if not (0.0 <= x_val <= 1000.0) or not (0.0 <= y_val <= 1000.0):
                stats["errors"].append(f"Row {row_idx} ({img_name}): GT out of bounds ({x_val:.2f}, {y_val:.2f})")
                pair_valid = False

        # Verify reference image
        ref_path = os.path.join(ref_dir, img_name)
        if not os.path.exists(ref_path):
            stats["errors"].append(f"Missing reference image: {ref_path}")
            pair_valid = False
        else:
            try:
                with Image.open(ref_path) as img:
                    img.verify()
                with Image.open(ref_path) as img:
                    if img.size != (1000, 1000):
                        stats["errors"].append(f"Reference image {img_name} dimensions {img.size} != (1000, 1000)")
                        pair_valid = False
            except Exception as e:
                stats["errors"].append(f"Corrupt or unreadable reference image {img_name}: {e}")
                pair_valid = False

        # Verify search image
        search_path = os.path.join(search_dir, img_name)
        if not os.path.exists(search_path):
            stats["errors"].append(f"Missing search image: {search_path}")
            pair_valid = False
        else:
            try:
                with Image.open(search_path) as img:
                    img.verify()
                with Image.open(search_path) as img:
                    if img.size != (1000, 1000):
                        stats["errors"].append(f"Search image {img_name} dimensions {img.size} != (1000, 1000)")
                        pair_valid = False
            except Exception as e:
                stats["errors"].append(f"Corrupt or unreadable search image {img_name}: {e}")
                pair_valid = False

        if pair_valid:
            valid_pairs += 1
            stats["arch_counts"][style] += 1
        else:
            stats["invalid_count"] += 1

    stats["pairs_count"] = len(ref_files.intersection(search_files).intersection(seen_filenames))
    stats["images_valid"] = valid_pairs

    # Check for orphan files (images without labels)
    orphan_ref = ref_files - seen_filenames
    orphan_search = search_files - seen_filenames
    if orphan_ref:
        stats["errors"].append(f"Found {len(orphan_ref)} reference images with no label record.")
    if orphan_search:
        stats["errors"].append(f"Found {len(orphan_search)} search images with no label record.")

    # Validate exact expected total
    if stats["images_valid"] != expected_count or stats["labels_count"] != expected_count:
        stats["errors"].append(
            f"Split count mismatch: Valid pairs ({stats['images_valid']}) or Labels ({stats['labels_count']}) != Expected ({expected_count})"
        )

    return stats


def main():
    dataset_dir = find_dataset_dir()
    print(f"Inspecting dataset at: {dataset_dir}\n")

    splits_config = [
        ("train", 1000),
        ("validation", 200),
        ("test", 200)
    ]

    all_stats = {}
    total_valid = 0
    total_expected = 1400
    overall_pass = True

    for split_name, expected_count in splits_config:
        stats = validate_split(split_name, expected_count, dataset_dir)
        all_stats[split_name] = stats
        total_valid += stats["images_valid"]
        if stats["errors"] or stats["images_valid"] != expected_count:
            overall_pass = False

    # Print requested formatted report
    for split_name, _ in splits_config:
        st = all_stats[split_name]
        print(f"{st['split']}")
        print("-" * len(st['split']))
        print(f"Pairs: {st['pairs_count']}")
        print(f"Images: {st['images_valid']}")
        print(f"Labels: {st['labels_count']}")
        print(f"Invalid: {st['invalid_count']}")
        if st["arch_counts"]:
            arch_str = ", ".join([f"{k}: {v}" for k, v in sorted(st["arch_counts"].items())])
            print(f"Architecture: {arch_str}")
        print()

    print("TOTAL")
    print("-----")
    print(f"{total_expected} expected")
    print(f"{total_valid} valid\n")

    print("Overall Dataset Status:")
    if overall_pass and total_valid == total_expected:
        print("PASS")
    else:
        print("FAIL")

    if not overall_pass:
        print("\nErrors Found:")
        for split_name, _ in splits_config:
            st = all_stats[split_name]
            if st["errors"]:
                print(f"\n--- {st['split']} Errors ---")
                for err in st["errors"][:10]:  # Limit print to first 10 per split
                    print(f" - {err}")
                if len(st["errors"]) > 10:
                    print(f" ... and {len(st['errors']) - 10} more errors.")

    sys.exit(0 if (overall_pass and total_valid == total_expected) else 1)


if __name__ == "__main__":
    main()
