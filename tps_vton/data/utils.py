"""Helpers for parsing VITON-HD label maps and building person representation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import torch
import torch.nn.functional as F


# VITON-HD / LIP parse-v3 label IDs we care about.
# (These follow the conventional VITON-HD parse-v3 colormap.)
PARSE_LABELS: Dict[str, Sequence[int]] = {
    "torso": [5, 6, 7],          # upper-clothes / dress / coat regions
    "left_sleeve": [14],
    "right_sleeve": [15],
    "collar": [1, 2],            # hat / hair as proxy near collar — refined below
}

# Cloth-region labels for agnostic-flow mask (where to *apply* cloth on the person)
CLOTH_REGION_LABELS = [5, 6, 7, 14, 15]

# Left/right swap pairs for horizontal flip (parse-v3 / LIP convention)
LR_SWAP_PAIRS = [
    (14, 15),   # left arm / right arm
    (12, 13),   # left leg / right leg
    (9, 10),    # left shoe / right shoe
]

NUM_PARSE_CLASSES = 20


def parse_to_onehot(parse_map: torch.Tensor, num_classes: int = NUM_PARSE_CLASSES) -> torch.Tensor:
    """Convert (H, W) integer parse map to (num_classes, H, W) one-hot float tensor."""
    if parse_map.ndim == 3 and parse_map.shape[0] == 1:
        parse_map = parse_map.squeeze(0)
    assert parse_map.ndim == 2, f"Expected (H, W), got {parse_map.shape}"
    parse_long = parse_map.long().clamp(0, num_classes - 1)
    onehot = F.one_hot(parse_long, num_classes=num_classes)        # (H, W, C)
    return onehot.permute(2, 0, 1).float()


def extract_body_part_masks(parse_map: torch.Tensor) -> torch.Tensor:
    """Return (4, H, W) float masks: torso, left_sleeve, right_sleeve, collar."""
    if parse_map.ndim == 3 and parse_map.shape[0] == 1:
        parse_map = parse_map.squeeze(0)
    masks = []
    for part in ("torso", "left_sleeve", "right_sleeve", "collar"):
        ids = PARSE_LABELS[part]
        m = torch.zeros_like(parse_map, dtype=torch.float32)
        for i in ids:
            m = m + (parse_map == i).float()
        masks.append(m.clamp(0, 1))
    return torch.stack(masks, dim=0)


def get_agnostic_flow_mask(parse_map: torch.Tensor) -> torch.Tensor:
    """Binary (1, H, W) mask of cloth-region pixels on the person."""
    if parse_map.ndim == 3 and parse_map.shape[0] == 1:
        parse_map = parse_map.squeeze(0)
    m = torch.zeros_like(parse_map, dtype=torch.float32)
    for i in CLOTH_REGION_LABELS:
        m = m + (parse_map == i).float()
    return m.clamp(0, 1).unsqueeze(0)


def swap_left_right_labels(parse_map: torch.Tensor) -> torch.Tensor:
    """After horizontal flip, swap L/R semantic labels in the parse map."""
    out = parse_map.clone()
    for a, b in LR_SWAP_PAIRS:
        a_mask = parse_map == a
        b_mask = parse_map == b
        out[a_mask] = b
        out[b_mask] = a
    return out


def mirror_keypoints_horizontal(keypoints: np.ndarray, image_width: int) -> np.ndarray:
    """Mirror OpenPose keypoints horizontally and swap L/R joint indices.

    keypoints: (N, 3) array of (x, y, conf).
    """
    kp = keypoints.copy()
    kp[:, 0] = (image_width - 1) - kp[:, 0]

    # OpenPose BODY_25 left/right swap pairs (joints only — face/hands ignored here)
    swap_pairs = [
        (2, 5), (3, 6), (4, 7),       # shoulders / elbows / wrists
        (9, 12), (10, 13), (11, 14),  # hips / knees / ankles
        (15, 16), (17, 18),           # eyes / ears
        (22, 19), (23, 20), (24, 21), # foot keypoints
    ]
    for a, b in swap_pairs:
        if a < kp.shape[0] and b < kp.shape[0]:
            kp[[a, b]] = kp[[b, a]]
    return kp


def load_openpose_keypoints(json_path: str | Path) -> np.ndarray:
    """Load BODY_25 keypoints from an OpenPose JSON file as (25, 3)."""
    with open(json_path, "r") as f:
        data = json.load(f)
    if not data.get("people"):
        return np.zeros((25, 3), dtype=np.float32)
    pose = data["people"][0].get("pose_keypoints_2d", [])
    arr = np.array(pose, dtype=np.float32).reshape(-1, 3)
    if arr.shape[0] < 25:
        pad = np.zeros((25 - arr.shape[0], 3), dtype=np.float32)
        arr = np.concatenate([arr, pad], axis=0)
    return arr[:25]


def make_cloth_semantic_mask(cloth_mask: torch.Tensor) -> torch.Tensor:
    """Approximate 3-channel semantic mask (collar / sleeves / body) from a binary cloth mask.

    Heuristic decomposition based on bounding-box position:
      - top 15% of cloth bbox  -> collar channel
      - left/right 20%         -> sleeve channels (combined)
      - center                 -> body channel

    cloth_mask: (1, H, W) float in [0, 1].
    Returns:    (3, H, W) float in [0, 1] — channels: [collar, sleeves, body].
    """
    assert cloth_mask.ndim == 3 and cloth_mask.shape[0] == 1, cloth_mask.shape
    m = (cloth_mask[0] > 0.5).float()
    H, W = m.shape
    out = torch.zeros(3, H, W, dtype=torch.float32)

    if m.sum() < 1:
        return out

    ys, xs = torch.where(m > 0)
    y0, y1 = int(ys.min().item()), int(ys.max().item())
    x0, x1 = int(xs.min().item()), int(xs.max().item())
    bbox_h = max(1, y1 - y0 + 1)
    bbox_w = max(1, x1 - x0 + 1)

    collar_y_end = y0 + max(1, int(0.15 * bbox_h))
    sleeve_w = max(1, int(0.20 * bbox_w))
    left_x_end = x0 + sleeve_w
    right_x_start = x1 - sleeve_w + 1

    out[0, y0:collar_y_end, x0:x1 + 1] = m[y0:collar_y_end, x0:x1 + 1]
    out[1, :, x0:left_x_end] = m[:, x0:left_x_end]
    out[1, :, right_x_start:x1 + 1] = m[:, right_x_start:x1 + 1]

    body = m.clone()
    body = body - out[0] - out[1]
    out[2] = body.clamp(0, 1)
    return out


def build_person_rep(
    agnostic: torch.Tensor,         # (3, H, W) in [0, 1]
    densepose: torch.Tensor,        # (3, H, W) in [0, 1]; will be normalized to [-1, 1]
    parse_onehot: torch.Tensor,     # (num_classes, H, W)
    body_part_masks: torch.Tensor,  # (4, H, W)
) -> torch.Tensor:
    """Concatenate inputs into the ~30-channel person representation."""
    densepose_norm = densepose * 2.0 - 1.0
    return torch.cat([agnostic, densepose_norm, parse_onehot, body_part_masks], dim=0)
