"""MultiScaleGMM: coarse 5x5 + fine 10x10 TPS warping network."""

from __future__ import annotations

from typing import Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .correlation import CorrelationLayer
from .feature_extractor import DualStreamExtractor
from .regression import RegressionNet
from .tps import TPSGridGen


class MultiScaleGMM(nn.Module):
    """Two-stage TPS GMM.

    Coarse stage (5x5 = 25 control points) handles global alignment;
    fine stage (10x10 = 100 control points) refines local detail by predicting
    *residual* offsets on top of the coarse warp.
    """

    def __init__(
        self,
        H: int,
        W: int,
        cloth_in_ch: int = 7,
        person_in_ch: int = 30,
        encoder_features: Sequence[int] = (64, 128, 256, 512),
        coarse_grid: int = 5,
        fine_grid: int = 10,
        regression_dropout: float = 0.1,
    ):
        super().__init__()
        self.H, self.W = H, W
        self.coarse_grid_size = coarse_grid
        self.fine_grid_size = fine_grid

        # Encoder downsamples by 2^len(features) (4 levels of 2x maxpool -> /16).
        self.feat_downsample = 2 ** len(encoder_features)
        assert H % self.feat_downsample == 0 and W % self.feat_downsample == 0, \
            f"H={H}, W={W} must be divisible by {self.feat_downsample}"
        self.feat_h = H // self.feat_downsample
        self.feat_w = W // self.feat_downsample
        corr_channels = self.feat_h * self.feat_w

        # Two encoders are reused for both stages (parameters shared across stages).
        self.feature_extractor = DualStreamExtractor(
            cloth_in_ch=cloth_in_ch,
            person_in_ch=person_in_ch,
            features=encoder_features,
        )

        # Coarse stage
        self.coarse_corr = CorrelationLayer()
        self.coarse_regression = RegressionNet(
            num_ctrl_pts=coarse_grid * coarse_grid,
            corr_in_channels=corr_channels,
            dropout=regression_dropout,
        )
        self.coarse_tps = TPSGridGen(out_h=H, out_w=W, grid_size=coarse_grid)

        # Fine stage
        self.fine_corr = CorrelationLayer()
        self.fine_regression = RegressionNet(
            num_ctrl_pts=fine_grid * fine_grid,
            corr_in_channels=corr_channels,
            dropout=regression_dropout,
        )
        self.fine_tps = TPSGridGen(out_h=H, out_w=W, grid_size=fine_grid)

    # ------------------------------------------------------------------
    def forward(
        self,
        cloth: torch.Tensor,            # (B, 3, H, W)
        cloth_mask: torch.Tensor,       # (B, 1, H, W)
        cloth_sem_mask: torch.Tensor,   # (B, 3, H, W)
        person_rep: torch.Tensor,       # (B, person_in_ch, H, W)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B, _, H, W = cloth.shape
        assert (H, W) == (self.H, self.W), \
            f"input HxW={(H, W)} doesn't match model HxW={(self.H, self.W)}"

        # ---- Encode person once (used by both stages) ----
        f_cloth_list, f_person_list = self.feature_extractor(
            cloth, cloth_mask, cloth_sem_mask, person_rep
        )
        f_cloth = f_cloth_list[-1]                              # (B, C, h, w)
        f_person = f_person_list[-1]

        # ===== Coarse stage =====
        coarse_corr_vol = self.coarse_corr(f_cloth, f_person)   # (B, h*w, h, w)
        coarse_theta = self.coarse_regression(coarse_corr_vol)  # (B, 25, 2)
        coarse_grid = self.coarse_tps(coarse_theta)             # (B, H, W, 2)
        coarse_warped = F.grid_sample(
            cloth, coarse_grid, mode="bilinear",
            padding_mode="border", align_corners=True,
        )                                                       # (B, 3, H, W)
        coarse_mask = F.grid_sample(
            cloth_mask, coarse_grid, mode="bilinear",
            padding_mode="border", align_corners=True,
        )                                                       # (B, 1, H, W)

        # ===== Fine stage (residual) =====
        # Re-extract cloth features from the coarse-warped cloth.
        coarse_cloth_in = torch.cat([coarse_warped, coarse_mask, cloth_sem_mask], dim=1)
        f_coarse_warped = self.feature_extractor.cloth_enc(coarse_cloth_in)[-1]

        fine_corr_vol = self.fine_corr(f_coarse_warped, f_person)
        fine_theta = self.fine_regression(fine_corr_vol)        # (B, 100, 2) — residual
        fine_grid = self.fine_tps(fine_theta)                   # (B, H, W, 2)

        warped_cloth = F.grid_sample(
            coarse_warped, fine_grid, mode="bilinear",
            padding_mode="border", align_corners=True,
        )
        warped_mask = F.grid_sample(
            coarse_mask, fine_grid, mode="bilinear",
            padding_mode="border", align_corners=True,
        )

        # Return everything the loss & visualization layers need.
        return warped_cloth, warped_mask, coarse_theta, fine_theta, coarse_warped

    # Convenience accessor used by the refinement network for skip connections
    def encode_person(self, person_rep: torch.Tensor):
        return self.feature_extractor.person_enc(person_rep)
