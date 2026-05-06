"""Dual-stream CNN feature extractor for the GMM."""

from __future__ import annotations

from typing import List, Sequence, Tuple

import torch
import torch.nn as nn


class CNN_Encoder(nn.Module):
    """4-level conv encoder: Conv-BN-ReLU-MaxPool at each level.

    Returns the feature map after each level so the UNet refinement network can
    consume them via skip connections.
    """

    def __init__(self, in_ch: int, features: Sequence[int] = (64, 128, 256, 512)):
        super().__init__()
        self.in_ch = in_ch
        self.features = list(features)

        levels: List[nn.Module] = []
        prev = in_ch
        for f in self.features:
            levels.append(
                nn.Sequential(
                    nn.Conv2d(prev, f, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(f),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(f, f, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(f),
                    nn.ReLU(inplace=True),
                )
            )
            prev = f
        self.levels = nn.ModuleList(levels)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        # (B, in_ch, H, W) -> list of feature maps, one per level (post-pool)
        assert x.shape[1] == self.in_ch, \
            f"CNN_Encoder expects {self.in_ch} input channels, got {x.shape[1]}"
        feats: List[torch.Tensor] = []
        for block in self.levels:
            x = block(x)
            x = self.pool(x)
            feats.append(x)
        return feats


class DualStreamExtractor(nn.Module):
    """Two parallel encoders: one for cloth (7ch), one for person rep (~30ch)."""

    def __init__(
        self,
        cloth_in_ch: int = 7,
        person_in_ch: int = 30,
        features: Sequence[int] = (64, 128, 256, 512),
    ):
        super().__init__()
        self.cloth_enc = CNN_Encoder(in_ch=cloth_in_ch, features=features)
        self.person_enc = CNN_Encoder(in_ch=person_in_ch, features=features)

    def forward(
        self,
        cloth: torch.Tensor,
        cloth_mask: torch.Tensor,
        cloth_sem_mask: torch.Tensor,
        person_rep: torch.Tensor,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        # (B, 3, H, W) + (B, 1, H, W) + (B, 3, H, W) -> (B, 7, H, W)
        cloth_in = torch.cat([cloth, cloth_mask, cloth_sem_mask], dim=1)
        f_cloth = self.cloth_enc(cloth_in)
        f_person = self.person_enc(person_rep)
        return f_cloth, f_person
