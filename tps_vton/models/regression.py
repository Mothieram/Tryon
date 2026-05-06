"""Regression head that predicts TPS control-point offsets from a correlation volume."""

from __future__ import annotations

import torch
import torch.nn as nn


class RegressionNet(nn.Module):
    """Conv head -> adaptive pool -> FC -> (num_ctrl_pts, 2) offsets.

    The final layer is identity-initialized (zero weights and bias) so that on
    the first forward pass the predicted offsets are zero and the TPS warp is
    the identity. Without this the regression head produces chaotic offsets
    that fold the grid before training stabilizes.
    """

    POOL_SIZE = (4, 3)  # adaptive pool output (H_out, W_out)

    def __init__(
        self,
        num_ctrl_pts: int,
        corr_in_channels: int,           # = H_feat * W_feat of the encoder bottleneck
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_ctrl_pts = num_ctrl_pts
        self.corr_in_channels = corr_in_channels

        # Conv stack reduces correlation-volume channels and spatial size.
        self.conv = nn.Sequential(
            nn.Conv2d(corr_in_channels, 512, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(self.POOL_SIZE)
        flat_dim = 128 * self.POOL_SIZE[0] * self.POOL_SIZE[1]

        self.fc = nn.Sequential(
            nn.Linear(flat_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_ctrl_pts * 2),
        )

        # Identity initialization: first forward pass produces zero offsets.
        nn.init.zeros_(self.fc[-1].weight)
        nn.init.zeros_(self.fc[-1].bias)

    def forward(self, corr: torch.Tensor) -> torch.Tensor:
        # corr: (B, H*W, H, W) — channels equal flattened spatial size of the
        # other branch's feature map (HxW from the encoder bottleneck).
        assert corr.shape[1] == self.corr_in_channels, \
            f"RegressionNet expects {self.corr_in_channels} input channels, got {corr.shape[1]}"
        x = self.conv(corr)
        x = self.pool(x)
        x = x.flatten(1)
        theta = self.fc(x)
        return theta.view(-1, self.num_ctrl_pts, 2)
