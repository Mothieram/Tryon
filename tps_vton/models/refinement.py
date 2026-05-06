"""UNet refinement network with optional GMM-feature fusion."""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class _ConvBlock(nn.Module):
    """Conv-BN-ReLU x2."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class RefinementUNet(nn.Module):
    """4-down/4-up UNet that produces a refined cloth image and a composition mask.

    Inputs (concatenated channel-wise):
      warped_cloth (3) + person_image (3) + agnostic_flow_mask (1) = 7

    Outputs:
      refined_cloth (B, 3, H, W) in [0, 1]
      comp_mask     (B, 1, H, W) in [0, 1]

    GMM encoder features (one per encoder level, channels = `gmm_features`)
    can be injected via the `gmm_feats` argument to the forward pass and are
    concatenated to the matching skip connection on the up-path.
    """

    def __init__(
        self,
        in_ch: int = 7,
        features: Sequence[int] = (64, 128, 256, 512),
        gmm_features: Optional[Sequence[int]] = (64, 128, 256, 512),
    ):
        super().__init__()
        self.in_ch = in_ch
        self.features = list(features)
        self.gmm_features = list(gmm_features) if gmm_features is not None else None

        # ---- Down path ----
        downs: List[nn.Module] = []
        prev = in_ch
        for f in self.features:
            downs.append(_ConvBlock(prev, f))
            prev = f
        self.downs = nn.ModuleList(downs)
        self.pool = nn.MaxPool2d(2, 2)

        # ---- Bottleneck ----
        self.bottleneck = _ConvBlock(self.features[-1], self.features[-1])

        # ---- Up path ----
        # We upsample the bottleneck (channels = features[-1]) and at each up step
        # concatenate the matching down-path skip and (optionally) the matching
        # GMM feature map.
        ups_up: List[nn.Module] = []          # 1x1 reductions after upsampling
        ups_block: List[nn.Module] = []       # conv blocks after the concat
        skip_features = list(reversed(self.features))   # [512, 256, 128, 64]
        gmm_feats_rev = list(reversed(self.gmm_features)) if self.gmm_features else [0] * len(skip_features)

        prev_up = self.features[-1]
        for i, skip_ch in enumerate(skip_features):
            ups_up.append(nn.Conv2d(prev_up, skip_ch, kernel_size=1, bias=False))
            in_concat = skip_ch + skip_ch + (gmm_feats_rev[i] if self.gmm_features else 0)
            ups_block.append(_ConvBlock(in_concat, skip_ch))
            prev_up = skip_ch
        self.ups_up = nn.ModuleList(ups_up)
        self.ups_block = nn.ModuleList(ups_block)

        # ---- Output head ----
        self.head_refined = nn.Conv2d(self.features[0], 3, kernel_size=1)
        self.head_compmask = nn.Conv2d(self.features[0], 1, kernel_size=1)

    # ------------------------------------------------------------------
    def forward(
        self,
        warped_cloth: torch.Tensor,         # (B, 3, H, W)
        person_image: torch.Tensor,         # (B, 3, H, W)
        agnostic_flow_mask: torch.Tensor,   # (B, 1, H, W)
        gmm_feats: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([warped_cloth, person_image, agnostic_flow_mask], dim=1)
        assert x.shape[1] == self.in_ch, \
            f"RefinementUNet expects {self.in_ch} input channels, got {x.shape[1]}"

        skips: List[torch.Tensor] = []
        for block in self.downs:
            x = block(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)

        # Up path with skip + (optional) GMM-feature concat
        skips_rev = list(reversed(skips))
        gmm_rev = list(reversed(gmm_feats)) if gmm_feats is not None else None
        for i, (up_conv, block) in enumerate(zip(self.ups_up, self.ups_block)):
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
            x = up_conv(x)
            parts = [x, skips_rev[i]]
            if gmm_rev is not None and i < len(gmm_rev):
                gf = gmm_rev[i]
                if gf.shape[2:] != x.shape[2:]:
                    gf = F.interpolate(gf, size=x.shape[2:], mode="bilinear", align_corners=False)
                parts.append(gf)
            x = torch.cat(parts, dim=1)
            x = block(x)

        refined = torch.sigmoid(self.head_refined(x))
        comp_mask = torch.sigmoid(self.head_compmask(x))
        return refined, comp_mask


def compose_output(
    refined_cloth: torch.Tensor, person_image: torch.Tensor, comp_mask: torch.Tensor
) -> torch.Tensor:
    """Final blend: comp_mask * refined + (1 - comp_mask) * person."""
    return comp_mask * refined_cloth + (1.0 - comp_mask) * person_image
