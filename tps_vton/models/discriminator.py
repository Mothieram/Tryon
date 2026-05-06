"""70x70 PatchGAN discriminator (conditional on the person image)."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


class PatchDiscriminator(nn.Module):
    """Conditional PatchGAN as in pix2pix.

    Input: composed_output (3) + person_image (3) = 6 channels.
    Output: (B, 1, h, w) patch-level real/fake logits, where each spatial
    location corresponds to roughly a 70x70 patch in the input.
    """

    def __init__(
        self,
        in_ch: int = 6,
        features: Sequence[int] = (64, 128, 256, 512),
        leaky_slope: float = 0.2,
    ):
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_ch, features[0], 4, stride=2, padding=1),
            nn.LeakyReLU(leaky_slope, inplace=True),
        ]
        prev = features[0]
        for i, f in enumerate(features[1:], start=1):
            stride = 2 if i < len(features) - 1 else 1     # last conv keeps spatial size
            layers += [
                nn.Conv2d(prev, f, 4, stride=stride, padding=1, bias=False),
                nn.InstanceNorm2d(f),
                nn.LeakyReLU(leaky_slope, inplace=True),
            ]
            prev = f
        layers.append(nn.Conv2d(prev, 1, 4, stride=1, padding=1))
        self.model = nn.Sequential(*layers)

    def forward(self, composed: torch.Tensor, person: torch.Tensor) -> torch.Tensor:
        # (B, 3, H, W) + (B, 3, H, W) -> (B, 6, H, W)
        x = torch.cat([composed, person], dim=1)
        return self.model(x)
