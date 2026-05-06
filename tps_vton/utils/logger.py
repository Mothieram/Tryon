"""Thin TensorBoard logger wrapper (with W&B as a fallback hook)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch.utils.tensorboard import SummaryWriter


class Logger:
    def __init__(self, log_dir: str | Path, run_name: Optional[str] = None):
        log_dir = Path(log_dir)
        if run_name:
            log_dir = log_dir / run_name
        log_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = log_dir
        self.writer = SummaryWriter(log_dir=str(log_dir))

    # ----- scalars -----
    def scalar(self, tag: str, value: float, step: int) -> None:
        self.writer.add_scalar(tag, float(value), step)

    def scalars(self, prefix: str, values: Dict[str, Any], step: int) -> None:
        for k, v in values.items():
            if isinstance(v, torch.Tensor):
                v = v.detach().float().mean().item()
            try:
                self.writer.add_scalar(f"{prefix}/{k}", float(v), step)
            except (TypeError, ValueError):
                pass  # skip non-scalar entries

    # ----- images -----
    def image(self, tag: str, img: torch.Tensor, step: int) -> None:
        # img may be (C, H, W) in [0, 1] or (H, W, C) — SummaryWriter handles both via dataformats
        if img.ndim == 3 and img.shape[0] in (1, 3):
            self.writer.add_image(tag, img.clamp(0, 1), step, dataformats="CHW")
        elif img.ndim == 3:
            self.writer.add_image(tag, img.clamp(0, 1), step, dataformats="HWC")
        else:
            raise ValueError(f"unsupported image shape {tuple(img.shape)} for tag {tag}")

    def images(self, prefix: str, imgs: Dict[str, torch.Tensor], step: int) -> None:
        for k, v in imgs.items():
            self.image(f"{prefix}/{k}", v, step)

    def lr(self, optimizer: torch.optim.Optimizer, step: int, tag: str = "train/lr") -> None:
        for i, group in enumerate(optimizer.param_groups):
            self.writer.add_scalar(f"{tag}_group{i}", group["lr"], step)

    def close(self) -> None:
        self.writer.flush()
        self.writer.close()
