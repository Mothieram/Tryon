"""VITON-HD paired dataset for TPS try-on training."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset

from .augmentation import PairedAugmentation
from .utils import (
    NUM_PARSE_CLASSES,
    build_person_rep,
    extract_body_part_masks,
    get_agnostic_flow_mask,
    load_openpose_keypoints,
    make_cloth_semantic_mask,
    parse_to_onehot,
)


def _read_pairs(path: str | Path) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                pairs.append((parts[0], parts[1]))
    return pairs


def _write_pairs(path: str | Path, pairs: List[Tuple[str, str]]) -> None:
    with open(path, "w") as f:
        for p, c in pairs:
            f.write(f"{p} {c}\n")


def deterministic_train_val_split(
    train_pairs_file: str | Path,
    val_pairs_file: str | Path,
    val_split: float,
    seed: int = 42,
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """Split train_pairs.txt into train + val. Result is cached at val_pairs_file
    so subsequent runs are deterministic. The split itself is fully determined by
    `seed`, so the cache is a convenience; if the target dir is read-only (e.g.
    /kaggle/input on Kaggle) we silently skip caching.
    """
    train_pairs_file = Path(train_pairs_file)
    val_pairs_file = Path(val_pairs_file)
    train_only_file = train_pairs_file.with_name("train_pairs_split.txt")

    if val_pairs_file.exists() and train_only_file.exists():
        return _read_pairs(train_only_file), _read_pairs(val_pairs_file)

    all_pairs = _read_pairs(train_pairs_file)
    rng = random.Random(seed)
    rng.shuffle(all_pairs)
    n_val = max(1, int(len(all_pairs) * val_split))
    val_pairs = all_pairs[:n_val]
    train_pairs = all_pairs[n_val:]
    try:
        _write_pairs(train_only_file, train_pairs)
        _write_pairs(val_pairs_file, val_pairs)
    except OSError as e:
        print(
            f"[warn] could not cache split files under {val_pairs_file.parent} "
            f"({e.strerror or e}); proceeding with in-memory split (deterministic by seed={seed})."
        )
    return train_pairs, val_pairs


class VitonHDDataset(Dataset):
    """VITON-HD paired dataset.

    Returns a dict of tensors with the inputs needed by the GMM and refinement
    networks. All image tensors are float32 in [0, 1] except DensePose (kept in
    [0, 1] here; build_person_rep normalizes it to [-1, 1]).
    """

    def __init__(
        self,
        cfg: Dict[str, Any],
        split: str = "train",
        augment: bool = True,
        pairs_override: Optional[List[Tuple[str, str]]] = None,
    ):
        self.cfg = cfg
        self.split = split
        data_cfg = cfg["data"]
        self.root = Path(data_cfg["root"])

        # VITON-HD uses 'train/' for both train and val (val is just held-out pairs)
        if split in ("train", "val"):
            self.split_dir = self.root / "train"
        elif split == "test":
            self.split_dir = self.root / "test"
        else:
            raise ValueError(f"Unknown split: {split}")

        self.subdirs = data_cfg["subdirs"]
        self.resolution = tuple(data_cfg["resolution"])           # (H, W)
        self.num_parse_classes = cfg["model"].get("num_parse_classes", NUM_PARSE_CLASSES)

        # Pair list selection
        if pairs_override is not None:
            self.pairs = pairs_override
        else:
            if split == "train":
                self.pairs = _read_pairs(self.root / "train_pairs_split.txt") \
                    if (self.root / "train_pairs_split.txt").exists() \
                    else _read_pairs(self.root / data_cfg["train_pairs"])
            elif split == "val":
                self.pairs = _read_pairs(self.root / data_cfg["val_pairs"])
            else:
                self.pairs = _read_pairs(self.root / data_cfg["test_pairs"])

        self.augmenter = PairedAugmentation(cfg["augmentation"]) if (augment and split == "train") else None

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.pairs)

    # ------------------------------------------------------------------
    def _path(self, subdir_key: str, filename: str) -> Path:
        return self.split_dir / self.subdirs[subdir_key] / filename

    def _load_rgb(self, path: Path) -> torch.Tensor:
        img = Image.open(path).convert("RGB")
        img = img.resize((self.resolution[1], self.resolution[0]), Image.BILINEAR)
        return TF.to_tensor(img).float()           # (3, H, W) in [0, 1]

    def _load_gray(self, path: Path) -> torch.Tensor:
        img = Image.open(path).convert("L")
        img = img.resize((self.resolution[1], self.resolution[0]), Image.NEAREST)
        arr = np.array(img, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).unsqueeze(0)  # (1, H, W) in [0, 1]

    def _load_parse(self, path: Path) -> torch.Tensor:
        img = Image.open(path)
        img = img.resize((self.resolution[1], self.resolution[0]), Image.NEAREST)
        arr = np.array(img, dtype=np.int64)
        if arr.ndim == 3:
            arr = arr[..., 0]
        return torch.from_numpy(arr)               # (H, W) integer

    # ------------------------------------------------------------------
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        person_name, cloth_name = self.pairs[idx]
        cloth_stem = Path(cloth_name).stem
        person_stem = Path(person_name).stem

        # ---- Load all paired inputs ----
        image     = self._load_rgb(self._path("image", person_name))
        agnostic  = self._load_rgb(self._path("agnostic", person_name))
        densepose = self._load_rgb(self._path("densepose", person_name))
        cloth     = self._load_rgb(self._path("cloth", cloth_name))
        cloth_mask = self._load_gray(self._path("cloth_mask", cloth_name))

        parse_map = self._load_parse(self._path("parse", person_stem + ".png"))
        try:
            parse_agn = self._load_parse(self._path("parse_agnostic", person_stem + ".png"))
        except FileNotFoundError:
            parse_agn = parse_map.clone()

        # OpenPose keypoints (optional — used by augmentation flip)
        kp_path = self._path("openpose_json", person_stem + "_keypoints.json")
        try:
            keypoints = torch.from_numpy(load_openpose_keypoints(kp_path))
        except FileNotFoundError:
            keypoints = torch.zeros(25, 3, dtype=torch.float32)

        # Cloth semantic mask (3ch) from the binary cloth mask
        cloth_sem_mask = make_cloth_semantic_mask(cloth_mask)

        sample: Dict[str, Any] = {
            "image": image,
            "agnostic": agnostic,
            "densepose": densepose,
            "cloth": cloth,
            "cloth_mask": cloth_mask,
            "cloth_sem_mask": cloth_sem_mask,
            "parse_map": parse_map,
            "parse_agnostic_map": parse_agn,
            "keypoints": keypoints,
            "person_name": person_name,
            "cloth_name": cloth_name,
        }

        # Pre-compute the agnostic-flow mask before augmentation so it flips with the rest.
        sample["agnostic_flow_mask"] = get_agnostic_flow_mask(parse_map)

        # ---- Paired augmentation ----
        if self.augmenter is not None:
            sample = self.augmenter(sample)

        # ---- Derived tensors (post-augmentation) ----
        parse_onehot = parse_to_onehot(sample["parse_map"], self.num_parse_classes)
        body_part_masks = extract_body_part_masks(sample["parse_map"])
        person_rep = build_person_rep(
            agnostic=sample["agnostic"],
            densepose=sample["densepose"],
            parse_onehot=parse_onehot,
            body_part_masks=body_part_masks,
        )

        # ---- Targets ----
        # GMM target cloth = the cloth as it appears on the person (cloth region of `image`).
        # We approximate it by masking the original image with the agnostic-flow mask;
        # for VITON-HD's same-cloth pairs this is a strong supervision signal.
        target_cloth = sample["image"] * sample["agnostic_flow_mask"]
        target_mask = sample["agnostic_flow_mask"]

        return {
            "person_rep": person_rep.float(),                       # (~30, H, W)
            "cloth": sample["cloth"].float(),                       # (3, H, W)
            "cloth_mask": sample["cloth_mask"].float(),             # (1, H, W)
            "cloth_sem_mask": sample["cloth_sem_mask"].float(),     # (3, H, W)
            "image": sample["image"].float(),                       # (3, H, W)
            "agnostic": sample["agnostic"].float(),                 # (3, H, W)
            "densepose": sample["densepose"].float(),               # (3, H, W)
            "parse_map": sample["parse_map"].long(),                # (H, W)
            "parse_onehot": parse_onehot.float(),                   # (C, H, W)
            "body_part_masks": body_part_masks.float(),             # (4, H, W)
            "agnostic_flow_mask": sample["agnostic_flow_mask"].float(),  # (1, H, W)
            "target_cloth": target_cloth.float(),                   # (3, H, W)
            "target_mask": target_mask.float(),                     # (1, H, W)
            "person_name": sample["person_name"],
            "cloth_name": sample["cloth_name"],
        }
