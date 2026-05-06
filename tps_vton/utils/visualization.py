"""Visualization helpers: TPS-grid drawing, side-by-side warp comparisons, comp-mask heatmaps.

All functions return torch.Tensor images shaped (3, H, W) in [0, 1] so the
TensorBoard logger can drop them straight in.
"""

from __future__ import annotations

import io
from typing import Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.utils as vutils


def _fig_to_tensor(fig) -> torch.Tensor:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=80)
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf).convert("RGB")
    arr = np.array(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)        # (3, H, W) in [0, 1]


def visualize_tps_grid(theta: torch.Tensor, grid_size: int) -> torch.Tensor:
    """Draw the regular grid + deformed grid overlaid for the first sample in the batch.

    theta: (B, N, 2) where N == grid_size^2.
    """
    if theta.ndim != 3:
        raise ValueError(f"theta must be (B, N, 2), got {theta.shape}")
    t = theta[0].detach().cpu().numpy().reshape(grid_size, grid_size, 2)

    axis = np.linspace(-1.0, 1.0, grid_size)
    ax_x, ax_y = np.meshgrid(axis, axis, indexing="xy")
    src_x = ax_x + t[..., 0]
    src_y = ax_y + t[..., 1]

    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    # Regular grid
    for i in range(grid_size):
        ax.plot(ax_x[i, :], ax_y[i, :], color="gray", lw=0.5, alpha=0.5)
        ax.plot(ax_x[:, i], ax_y[:, i], color="gray", lw=0.5, alpha=0.5)
    # Deformed grid
    for i in range(grid_size):
        ax.plot(src_x[i, :], src_y[i, :], color="red", lw=1.0)
        ax.plot(src_x[:, i], src_y[:, i], color="red", lw=1.0)
    ax.scatter(src_x.flatten(), src_y.flatten(), c="red", s=8)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(1.5, -1.5)        # invert y for image-space convention
    ax.set_aspect("equal")
    ax.set_title(f"TPS grid {grid_size}x{grid_size}")
    ax.grid(False)
    return _fig_to_tensor(fig)


def visualize_warp_result(
    cloth: torch.Tensor,
    warped_cloth: torch.Tensor,
    target: torch.Tensor,
    person: torch.Tensor,
    n_samples: int = 4,
) -> torch.Tensor:
    """Side-by-side grid: cloth | warped | target | person."""
    n = min(n_samples, cloth.size(0))
    rows = []
    for i in range(n):
        row = torch.cat([
            cloth[i].detach().cpu().clamp(0, 1),
            warped_cloth[i].detach().cpu().clamp(0, 1),
            target[i].detach().cpu().clamp(0, 1),
            person[i].detach().cpu().clamp(0, 1),
        ], dim=2)                                       # concat along width
        rows.append(row)
    grid = torch.cat(rows, dim=1)                       # concat along height
    return grid


def visualize_comp_mask(comp_mask: torch.Tensor) -> torch.Tensor:
    """Render a (B, 1, H, W) composition mask as an RGB heatmap (first sample)."""
    m = comp_mask[0, 0].detach().cpu().numpy()
    fig, ax = plt.subplots(figsize=(3, 4))
    im = ax.imshow(m, cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_title("composition mask")
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return _fig_to_tensor(fig)


def make_image_grid(images: torch.Tensor, nrow: int = 4) -> torch.Tensor:
    """Wrapper around torchvision.utils.make_grid returning (3, H, W) in [0, 1]."""
    return vutils.make_grid(images.detach().cpu().clamp(0, 1), nrow=nrow, padding=2)
