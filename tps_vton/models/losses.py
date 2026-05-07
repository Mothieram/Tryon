"""All loss functions for the TPS try-on pipeline.

Provides individual loss modules plus two aggregator classes:
  - GMMLossComputer        (Stage 1)
  - RefinementLossComputer (Stage 2)
Each aggregator returns (total_loss, dict_of_individual_losses) so the training
loop can log every component separately.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm


# ----------------------------------------------------------------------
# VGG perceptual loss
# ----------------------------------------------------------------------

# Layer indices in torchvision VGG16's `features` Sequential
_VGG16_LAYER_IDX = {
    "relu1_2": 3,
    "relu2_2": 8,
    "relu3_3": 15,
    "relu4_3": 22,
}

# ImageNet normalization stats
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class VGGPerceptualLoss(nn.Module):
    """L1 distance on intermediate VGG16 feature maps. VGG is frozen."""

    def __init__(
        self,
        layers: Sequence[str] = ("relu1_2", "relu2_2", "relu3_3", "relu4_3"),
        normalize_input: bool = True,
    ):
        super().__init__()
        try:
            vgg = tvm.vgg16(weights=tvm.VGG16_Weights.IMAGENET1K_V1).features
        except Exception:
            vgg = tvm.vgg16(pretrained=True).features
        for p in vgg.parameters():
            p.requires_grad = False
        vgg.eval()

        # Slice VGG into the segments needed to extract each requested layer.
        self.layer_names = list(layers)
        self.layer_indices = [_VGG16_LAYER_IDX[name] for name in self.layer_names]
        max_idx = max(self.layer_indices)

        slices: list[nn.Module] = []
        prev = 0
        for idx in self.layer_indices:
            slices.append(nn.Sequential(*[vgg[i] for i in range(prev, idx + 1)]))
            prev = idx + 1
        self.slices = nn.ModuleList(slices)
        self.max_idx = max_idx

        self.normalize_input = normalize_input
        self.register_buffer("mean", _IMAGENET_MEAN)
        self.register_buffer("std", _IMAGENET_STD)

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        if not self.normalize_input:
            return x
        return (x - self.mean) / self.std

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # pred / target: (B, 3, H, W) in [0, 1]
        x = self._normalize(pred)
        y = self._normalize(target)
        loss = pred.new_zeros(())
        for slc in self.slices:
            x = slc(x)
            with torch.no_grad():
                y = slc(y)
            loss = loss + F.l1_loss(x, y)
        return loss / max(1, len(self.slices))


# ----------------------------------------------------------------------
# Edge / Sobel loss
# ----------------------------------------------------------------------

class SobelEdge(nn.Module):
    """Apply Sobel filter (horizontal + vertical) to a (B, C, H, W) tensor."""

    def __init__(self):
        super().__init__()
        kx = torch.tensor([[-1.0, 0.0, 1.0],
                           [-2.0, 0.0, 2.0],
                           [-1.0, 0.0, 1.0]])
        ky = kx.t().contiguous()
        kernel = torch.stack([kx, ky], dim=0).unsqueeze(1)        # (2, 1, 3, 3)
        self.register_buffer("kernel", kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        k = self.kernel.repeat(C, 1, 1, 1)                        # (2C, 1, 3, 3)
        # depthwise conv: 2 outputs per channel (gx, gy)
        weight = k.view(2 * C, 1, 3, 3)
        out = F.conv2d(x, weight, padding=1, groups=C)            # (B, 2C, H, W)
        return out


class EdgeLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.sobel = SobelEdge()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.l1_loss(self.sobel(pred), self.sobel(target))


# ----------------------------------------------------------------------
# Grid smoothness (second-order) and TPS regularization
# ----------------------------------------------------------------------

def grid_smoothness_loss(theta: torch.Tensor, grid_size: int) -> torch.Tensor:
    """Second-order smoothness on TPS control-point offsets.

    theta: (B, N, 2) where N == grid_size * grid_size.
    Penalizes inconsistent displacement between neighboring control points to
    prevent local grid folding.
    """
    B, N, _ = theta.shape
    assert N == grid_size * grid_size, f"theta has {N} points but grid_size={grid_size}"
    grid = theta.view(B, grid_size, grid_size, 2)

    dx = grid[:, :, 1:, :] - grid[:, :, :-1, :]      # horizontal first diff
    dy = grid[:, 1:, :, :] - grid[:, :-1, :, :]      # vertical first diff

    ddx = dx[:, :, 1:, :] - dx[:, :, :-1, :]         # horizontal second diff
    ddy = dy[:, 1:, :, :] - dy[:, :-1, :, :]         # vertical second diff

    return 0.5 * (ddx.pow(2).mean() + ddy.pow(2).mean())


def tps_regularization(theta: torch.Tensor) -> torch.Tensor:
    """L2 penalty pulling theta toward the identity (zero offsets)."""
    return F.mse_loss(theta, torch.zeros_like(theta))


# ----------------------------------------------------------------------
# GAN losses (LSGAN / MSE)
# ----------------------------------------------------------------------

def lsgan_g_loss(pred_fake: torch.Tensor) -> torch.Tensor:
    """Generator wants D to call composed output real (label = 1)."""
    return F.mse_loss(pred_fake, torch.ones_like(pred_fake))


def lsgan_d_loss(pred_real: torch.Tensor, pred_fake: torch.Tensor) -> torch.Tensor:
    """Discriminator: real -> 1, fake -> 0, averaged."""
    loss_real = F.mse_loss(pred_real, torch.ones_like(pred_real))
    loss_fake = F.mse_loss(pred_fake, torch.zeros_like(pred_fake))
    return 0.5 * (loss_real + loss_fake)


# ----------------------------------------------------------------------
# Stage 1: GMM aggregator
# ----------------------------------------------------------------------

class GMMLossComputer(nn.Module):
    """Aggregate all GMM losses with config-controlled weights.

    Returns (total_loss, individual_losses_dict) where the dict contains
    every individual (unweighted) loss so they can be logged separately.
    """

    def __init__(self, cfg: Dict, coarse_grid: int, fine_grid: int):
        super().__init__()
        self.weights = cfg["losses"]["gmm"]
        self.perceptual = VGGPerceptualLoss(
            layers=cfg["losses"]["perceptual"]["vgg_layers"],
            normalize_input=cfg["losses"]["perceptual"]["normalize_input"],
        )
        self.edge = EdgeLoss()
        self.coarse_grid = coarse_grid
        self.fine_grid = fine_grid

    def forward(
        self,
        warped_cloth: torch.Tensor,
        warped_mask: torch.Tensor,
        coarse_theta: torch.Tensor,
        fine_theta: torch.Tensor,
        coarse_warped: torch.Tensor,
        target_cloth: torch.Tensor,
        target_mask: torch.Tensor,
        reg_weight: float,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # target_cloth is already (image * cloth_region_mask). Apply the same mask
        # to predictions so we score only the cloth region — the input cloth's
        # white background must not get scored against the target's zero background.
        m = target_mask
        coarse_pred = coarse_warped * m
        fine_pred = warped_cloth * m

        # Coarse stage
        coarse_l1 = F.l1_loss(coarse_pred, target_cloth)
        coarse_perc = self.perceptual(coarse_pred.clamp(0, 1), target_cloth.clamp(0, 1))

        # Fine stage (primary)
        fine_l1 = F.l1_loss(fine_pred, target_cloth)
        fine_perc = self.perceptual(fine_pred.clamp(0, 1), target_cloth.clamp(0, 1))

        mask_loss = F.mse_loss(warped_mask, target_mask)
        edge_loss = self.edge(fine_pred, target_cloth)

        # TPS regularization (warmup-controlled)
        coarse_reg = tps_regularization(coarse_theta)
        fine_reg = tps_regularization(fine_theta)

        # Second-order smoothness
        coarse_smooth = grid_smoothness_loss(coarse_theta, self.coarse_grid)
        fine_smooth = grid_smoothness_loss(fine_theta, self.fine_grid)

        w = self.weights
        total = (
            w["coarse_l1"] * coarse_l1
            + w["coarse_perceptual"] * coarse_perc
            + w["fine_l1"] * fine_l1
            + w["fine_perceptual"] * fine_perc
            + w["mask"] * mask_loss
            + w["edge"] * edge_loss
            + reg_weight * (coarse_reg + fine_reg)
            + w["grid_smoothness"] * (coarse_smooth + fine_smooth)
        )

        parts: Dict[str, torch.Tensor] = {
            "coarse_l1": coarse_l1.detach(),
            "coarse_perceptual": coarse_perc.detach(),
            "fine_l1": fine_l1.detach(),
            "fine_perceptual": fine_perc.detach(),
            "mask": mask_loss.detach(),
            "edge": edge_loss.detach(),
            "coarse_reg": coarse_reg.detach(),
            "fine_reg": fine_reg.detach(),
            "coarse_smooth": coarse_smooth.detach(),
            "fine_smooth": fine_smooth.detach(),
            "reg_weight": torch.tensor(reg_weight, device=warped_cloth.device),
            "total": total.detach(),
        }
        return total, parts


# ----------------------------------------------------------------------
# Stage 2: Refinement aggregator (generator side)
# ----------------------------------------------------------------------

class RefinementLossComputer(nn.Module):
    """Generator-side losses for the refinement network.

    The discriminator is updated separately in the training loop using
    `lsgan_d_loss` directly.
    """

    def __init__(self, cfg: Dict):
        super().__init__()
        self.weights = cfg["losses"]["refinement"]
        self.perceptual = VGGPerceptualLoss(
            layers=cfg["losses"]["perceptual"]["vgg_layers"],
            normalize_input=cfg["losses"]["perceptual"]["normalize_input"],
        )
        self.edge = EdgeLoss()

    def forward(
        self,
        composed: torch.Tensor,           # (B, 3, H, W) — comp_mask*refined + (1-comp_mask)*person
        comp_mask: torch.Tensor,          # (B, 1, H, W) — predicted composition mask
        target_image: torch.Tensor,       # (B, 3, H, W) — ground-truth person+cloth
        gt_comp_mask: torch.Tensor,       # (B, 1, H, W) — agnostic-flow / cloth-region mask
        pred_fake: Optional[torch.Tensor] = None,   # discriminator output on composed
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # Pixel L1 weighted by the GT cloth region — the rest of the image is the
        # original person and isn't being refined here.
        l1 = F.l1_loss(composed * gt_comp_mask, target_image * gt_comp_mask)

        perc = self.perceptual(
            (composed * gt_comp_mask).clamp(0, 1),
            (target_image * gt_comp_mask).clamp(0, 1),
        )

        # F.binary_cross_entropy is unsafe under autocast; do this term in fp32.
        with torch.cuda.amp.autocast(enabled=False):
            comp_bce = F.binary_cross_entropy(
                comp_mask.float().clamp(1e-6, 1 - 1e-6), gt_comp_mask.float()
            )
        edge_loss = self.edge(composed, target_image)

        gan = composed.new_zeros(())
        if pred_fake is not None:
            gan = lsgan_g_loss(pred_fake)

        w = self.weights
        total = (
            w["l1"] * l1
            + w["perceptual"] * perc
            + w["comp_mask_bce"] * comp_bce
            + w["edge"] * edge_loss
            + w["gan"] * gan
        )

        parts: Dict[str, torch.Tensor] = {
            "G_l1": l1.detach(),
            "G_perceptual": perc.detach(),
            "G_comp_mask_bce": comp_bce.detach(),
            "G_edge": edge_loss.detach(),
            "G_gan": gan.detach(),
            "G_total": total.detach(),
        }
        return total, parts
