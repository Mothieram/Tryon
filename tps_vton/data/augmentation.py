"""Paired augmentations for VITON-HD try-on training.

Geometric transforms must be applied identically to all paired tensors
(person/cloth/masks/parse/densepose). Color jitter is applied only to the
cloth image so person appearance stays fixed.
"""

from __future__ import annotations

import random
from typing import Dict

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from .utils import mirror_keypoints_horizontal, swap_left_right_labels


class PairedAugmentation:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.do_flip = bool(cfg.get("horizontal_flip", True))

        cj = cfg.get("color_jitter", {}) or {}
        self.cloth_color_jitter = T.ColorJitter(
            brightness=cj.get("brightness", 0.0),
            contrast=cj.get("contrast", 0.0),
            saturation=cj.get("saturation", 0.0),
            hue=cj.get("hue", 0.0),
        )

        aff = cfg.get("random_affine", {}) or {}
        self.affine_enabled = bool(aff.get("enabled", False))
        self.affine_angle = float(aff.get("angle", 0.0))
        self.affine_translate = tuple(aff.get("translate", [0.0, 0.0]))
        self.affine_scale = tuple(aff.get("scale", [1.0, 1.0]))

    # --- Helpers ---------------------------------------------------------
    @staticmethod
    def _hflip(t: torch.Tensor) -> torch.Tensor:
        return TF.hflip(t)

    @staticmethod
    def _affine(t: torch.Tensor, angle, translate_px, scale, interp) -> torch.Tensor:
        return TF.affine(
            t, angle=angle, translate=list(translate_px), scale=scale,
            shear=[0.0, 0.0], interpolation=interp,
        )

    # --- Main entry ------------------------------------------------------
    def __call__(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # ---- Horizontal flip (paired) ----
        if self.do_flip and random.random() > 0.5:
            for k in (
                "cloth", "cloth_mask", "cloth_sem_mask",
                "image", "agnostic", "densepose", "parse_map",
                "parse_agnostic_map", "agnostic_flow_mask",
            ):
                if k in sample:
                    sample[k] = self._hflip(sample[k])

            if "parse_map" in sample:
                sample["parse_map"] = swap_left_right_labels(sample["parse_map"])
            if "parse_agnostic_map" in sample:
                sample["parse_agnostic_map"] = swap_left_right_labels(sample["parse_agnostic_map"])

            if "keypoints" in sample and sample["keypoints"] is not None:
                _, _, W = sample["image"].shape
                sample["keypoints"] = torch.from_numpy(
                    mirror_keypoints_horizontal(sample["keypoints"].numpy(), image_width=W)
                )

        # ---- Color jitter (cloth only) ----
        if "cloth" in sample:
            sample["cloth"] = self.cloth_color_jitter(sample["cloth"])

        # ---- Random affine on cloth + cloth_mask + cloth_sem_mask ----
        if self.affine_enabled and random.random() > 0.5:
            angle = random.uniform(-self.affine_angle, self.affine_angle)
            tx = random.uniform(-self.affine_translate[0], self.affine_translate[0])
            ty = random.uniform(-self.affine_translate[1], self.affine_translate[1])
            scale = random.uniform(self.affine_scale[0], self.affine_scale[1])

            _, H, W = sample["cloth"].shape
            translate_px = (tx * W, ty * H)

            sample["cloth"] = self._affine(
                sample["cloth"], angle, translate_px, scale,
                interp=TF.InterpolationMode.BILINEAR,
            )
            if "cloth_mask" in sample:
                sample["cloth_mask"] = self._affine(
                    sample["cloth_mask"], angle, translate_px, scale,
                    interp=TF.InterpolationMode.NEAREST,
                )
            if "cloth_sem_mask" in sample:
                sample["cloth_sem_mask"] = self._affine(
                    sample["cloth_sem_mask"], angle, translate_px, scale,
                    interp=TF.InterpolationMode.NEAREST,
                )

        return sample
