"""Verify VITON-HD layout, build a deterministic train/val split, and report stats."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List

import numpy as np
import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.dataset import deterministic_train_val_split


REQUIRED_SUBDIRS = [
    "image", "cloth", "cloth-mask", "agnostic-v3.2",
    "image-densepose", "image-parse-v3", "image-parse-agnostic-v3.2",
    "openpose_json",
]


def _check_split(split_dir: Path) -> Dict[str, int]:
    print(f"\n[check] {split_dir}")
    counts: Dict[str, int] = {}
    if not split_dir.exists():
        print(f"  [skip] not found")
        return counts
    for sub in REQUIRED_SUBDIRS:
        d = split_dir / sub
        if not d.exists():
            print(f"  MISSING: {sub}/")
            counts[sub] = 0
            continue
        files = [p for p in d.iterdir() if p.is_file()]
        counts[sub] = len(files)
        print(f"  ok    {sub:35s}  files={len(files)}")
    return counts


def _read_pairs(path: Path) -> List[str]:
    if not path.exists():
        return []
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def _parse_label_distribution(parse_dir: Path, n_samples: int = 64) -> Dict[int, int]:
    """Sample a few parse maps and count label-pixel frequencies (sanity check)."""
    files = sorted([p for p in parse_dir.iterdir() if p.suffix == ".png"])[:n_samples]
    counter: Counter = Counter()
    for fp in files:
        arr = np.array(Image.open(fp))
        if arr.ndim == 3:
            arr = arr[..., 0]
        for label in np.unique(arr):
            counter[int(label)] += int((arr == label).sum())
    return dict(sorted(counter.items()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    repo_root = Path(__file__).resolve().parents[2]
    data_root = Path(cfg["data"]["root"])
    if not data_root.is_absolute():
        data_root = (repo_root / data_root).resolve()
    print(f"[info] data root = {data_root}")

    # ---- Directory check ----
    train_counts = _check_split(data_root / "train")
    test_counts = _check_split(data_root / "test")

    # ---- Pairs files ----
    train_pairs_file = data_root / cfg["data"]["train_pairs"]
    if not train_pairs_file.exists():
        alt = repo_root / cfg["data"]["train_pairs"]
        if alt.exists():
            train_pairs_file = alt
    test_pairs_file = data_root / cfg["data"]["test_pairs"]
    if not test_pairs_file.exists():
        alt = repo_root / cfg["data"]["test_pairs"]
        if alt.exists():
            test_pairs_file = alt

    print(f"\n[info] train_pairs file = {train_pairs_file} (exists={train_pairs_file.exists()})")
    print(f"[info] test_pairs  file = {test_pairs_file} (exists={test_pairs_file.exists()})")
    print(f"[info] train pairs count = {len(_read_pairs(train_pairs_file))}")
    print(f"[info] test  pairs count = {len(_read_pairs(test_pairs_file))}")

    # ---- Deterministic train/val split ----
    val_pairs_file = data_root / cfg["data"]["val_pairs"]
    train_pairs, val_pairs = deterministic_train_val_split(
        train_pairs_file, val_pairs_file,
        val_split=cfg["data"]["val_split"], seed=cfg["training"]["seed"],
    )
    print(f"\n[split] train = {len(train_pairs)}  val = {len(val_pairs)}")
    print(f"  saved to {val_pairs_file}")
    print(f"  and to  {train_pairs_file.with_name('train_pairs_split.txt')}")

    # ---- Sample image dimensions ----
    sample_image = next((data_root / "train" / "image").iterdir(), None)
    if sample_image is not None:
        with Image.open(sample_image) as im:
            print(f"\n[info] sample image dimensions = {im.size}  (W, H)")

    # ---- Parse-label distribution sample ----
    parse_dir = data_root / "train" / "image-parse-v3"
    if parse_dir.exists():
        dist = _parse_label_distribution(parse_dir, n_samples=32)
        print(f"\n[info] parse label distribution (32 samples) — label : pixel count")
        for label, count in dist.items():
            print(f"  {label:3d}  {count:>12,}")

    # ---- Summary file ----
    summary = {
        "data_root": str(data_root),
        "train_pairs": len(_read_pairs(train_pairs_file)),
        "test_pairs": len(_read_pairs(test_pairs_file)),
        "split_train": len(train_pairs),
        "split_val": len(val_pairs),
        "train_subdir_counts": train_counts,
        "test_subdir_counts": test_counts,
    }
    summary_path = data_root / "dataset_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[info] summary written to {summary_path}")


if __name__ == "__main__":
    main()
