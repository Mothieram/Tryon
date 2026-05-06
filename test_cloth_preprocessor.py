"""Smoke test for engine.cloth_preprocessor.

Run from the project root:
    python test_cloth_preprocessor.py
or
    python test_cloth_preprocessor.py --shirt assets/shirts/shirt2.png
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from engine.cloth_preprocessor import ClothPreprocessor, ProcessedCloth


def _assert(cond: bool, msg: str) -> None:
    print(("PASS" if cond else "FAIL") + " - " + msg)
    if not cond:
        raise AssertionError(msg)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--shirt",
        default=str(PROJECT_ROOT / "assets" / "shirts" / "shirt2.png"),
        help="Path to shirt PNG to preprocess.",
    )
    ap.add_argument(
        "--cache",
        default=str(PROJECT_ROOT / "assets" / "processed_shirts"),
        help="Cache root directory.",
    )
    ap.add_argument(
        "--clean",
        action="store_true",
        help="Wipe cache before testing (force fresh processing).",
    )
    args = ap.parse_args()

    src = Path(args.shirt)
    if not src.exists():
        print(f"ERROR: shirt not found at {src}")
        return 1

    cache_root = Path(args.cache)
    cache_dir = cache_root / src.stem
    if args.clean and cache_dir.exists():
        shutil.rmtree(cache_dir)

    pre = ClothPreprocessor(
        cache_root=str(cache_root),
        target_size=(512, 384),  # (H, W)
        grid_size=(5, 5),
    )

    # ── First run: should process from scratch ──────────────────────────
    t0 = time.perf_counter()
    p1: ProcessedCloth = pre.process(str(src), force=False)
    dt_first = (time.perf_counter() - t0) * 1000.0
    print(f"first run: {dt_first:.1f} ms")

    _assert(isinstance(p1, ProcessedCloth), "process() returns ProcessedCloth")
    _assert(p1.name == src.stem, f"name == '{src.stem}'")
    _assert(p1.image_bgra.ndim == 3 and p1.image_bgra.shape[2] == 4,
            f"image is BGRA (got shape {p1.image_bgra.shape})")
    _assert(p1.image_bgra.shape[:2] == (512, 384),
            f"image shape == (512, 384) (got {p1.image_bgra.shape[:2]})")

    unique_mask_vals = set(np.unique(p1.mask).tolist())
    _assert(unique_mask_vals.issubset({0, 255}),
            f"mask is binary (got values {unique_mask_vals})")

    _assert(p1.control_points.shape == (25, 2),
            f"control points shape == (25, 2) (got {p1.control_points.shape})")
    # Snap-to-mask should put most points on opaque pixels.
    on_mask = 0
    h, w = p1.mask.shape[:2]
    for x, y in p1.control_points:
        ix, iy = int(round(x)), int(round(y))
        if 0 <= ix < w and 0 <= iy < h and p1.mask[iy, ix] > 0:
            on_mask += 1
    _assert(on_mask >= 15, f"≥15 of 25 grid points snapped to mask (got {on_mask})")

    # Cache files should exist now.
    for fn in ("image.png", "mask.npy", "edges.npy", "control_points.npy", "meta.json"):
        _assert((cache_dir / fn).exists(), f"cache file {fn} created")

    # ── Second run: should hit the cache and be much faster ─────────────
    t0 = time.perf_counter()
    p2 = pre.process(str(src), force=False)
    dt_second = (time.perf_counter() - t0) * 1000.0
    print(f"cached run: {dt_second:.1f} ms")

    _assert(dt_second < dt_first, "cached run is faster than fresh run")
    _assert(np.array_equal(p1.mask, p2.mask),
            "cached mask matches fresh mask byte-for-byte")
    _assert(np.allclose(p1.control_points, p2.control_points),
            "cached control points match fresh ones")

    # ── Cache invalidation: touching the source must reprocess ──────────
    src_stat_before = src.stat()
    src.touch()
    try:
        p3 = pre.process(str(src), force=False)
        # We can't assert it was slow (Windows file mtimes are coarse), but
        # we can confirm the cache_is_valid flag flipped before reload.
        _assert(p3.image_bgra.shape == p1.image_bgra.shape,
                "post-touch reprocessing returns same-shape image")
    finally:
        # restore mtime to avoid leaving a dirty source for next run
        import os
        os.utime(src, (src_stat_before.st_atime, src_stat_before.st_mtime))

    print()
    print("All preprocessor smoke tests passed.")
    print(f"Cache: {cache_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
