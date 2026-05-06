"""
Compare actual vs golden snapshots.

Usage:
    python tests/compare_snapshots.py [--dir tests/snapshots]

Prints a per-snapshot summary with shape, dtype, max absolute diff, and
fraction of differing pixels. Exits non-zero if any mismatch is found.
"""

import argparse
import sys
from pathlib import Path

import numpy as np


def _id_to_rgb(semantic_id: int) -> tuple[int, int, int]:
    h = int(semantic_id) & 0xFFFFFFFF
    h = ((h >> 16) ^ h) * 0x45D9F3B
    h = ((h >> 16) ^ h) * 0x45D9F3B
    h = (h >> 16) ^ h
    return ((h & 0xFF0000) >> 16, (h & 0x00FF00) >> 8, h & 0x0000FF)


def _array_to_rgb(array: np.ndarray) -> np.ndarray:
    if np.issubdtype(array.dtype, np.integer):
        h, w = array.shape
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        for uid in np.unique(array):
            rgb[array == uid] = _id_to_rgb(int(uid))
    else:
        finite = array[np.isfinite(array)]
        lo, hi = (float(finite.min()), float(finite.max())) if finite.size else (0.0, 1.0)
        normalized = np.clip((array - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
        gray = (normalized * 255).astype(np.uint8)
        rgb = np.stack([gray, gray, gray], axis=-1)
    return rgb


def _save_diff_png(actual: np.ndarray, golden: np.ndarray, path: Path) -> None:
    try:
        from PIL import Image
    except ImportError:
        return

    diff = (actual != golden).astype(np.uint8) * 255
    rgb = np.stack([diff, np.zeros_like(diff), np.zeros_like(diff)], axis=-1)
    Image.fromarray(rgb).save(path)


def compare(snapshots_dir: Path) -> bool:
    golden_dir = snapshots_dir / "golden"
    actual_dir = snapshots_dir / "actual"

    actual_npys = sorted(actual_dir.glob("*.npy"))
    if not actual_npys:
        print(f"No .npy files found in {actual_dir}")
        return True

    all_match = True
    col = "{:<50} {:<12} {:<20} {:<12} {}"
    print(col.format("Name", "Shape", "Dtype", "Max Δ", "Diff pixels"))
    print("-" * 110)

    for actual_path in actual_npys:
        name = actual_path.stem
        golden_path = golden_dir / actual_path.name

        if not golden_path.exists():
            print(f"  MISSING GOLDEN  {name}")
            all_match = False
            continue

        actual = np.load(actual_path)
        golden = np.load(golden_path)

        if actual.shape != golden.shape:
            print(
                col.format(name, str(actual.shape), str(actual.dtype), "N/A", f"SHAPE MISMATCH (golden={golden.shape})")
            )
            all_match = False
            continue

        if np.array_equal(actual, golden):
            print(col.format(name, str(actual.shape), str(actual.dtype), "0", "0 / 0 (MATCH)"))
            continue

        all_match = False
        if np.issubdtype(actual.dtype, np.integer):
            max_diff = int(np.max(np.abs(actual.astype(np.int64) - golden.astype(np.int64))))
        else:
            max_diff_val = np.max(np.abs(actual.astype(np.float64) - golden.astype(np.float64)))
            max_diff = f"{max_diff_val:.4g}"

        diff_mask = actual != golden
        n_diff = int(diff_mask.sum())
        n_total = actual.size
        pct = 100.0 * n_diff / n_total
        print(
            col.format(name, str(actual.shape), str(actual.dtype), str(max_diff), f"{n_diff} / {n_total} ({pct:.2f}%)")
        )

        diff_png = actual_dir / f"{name}_diff.png"
        _save_diff_png(actual, golden, diff_png)

    print()
    if all_match:
        print("All snapshots match.")
    else:
        print("MISMATCH detected. Diff PNGs written to actual/ where applicable.")

    return all_match


def main():
    parser = argparse.ArgumentParser(description="Compare actual vs golden snapshots.")
    parser.add_argument(
        "--dir",
        default=Path(__file__).parent / "snapshots",
        type=Path,
        help="Path to snapshots directory (default: tests/snapshots/)",
    )
    args = parser.parse_args()

    ok = compare(args.dir)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
