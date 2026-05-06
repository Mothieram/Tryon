"""L2-normalized correlation (cosine similarity) layer for the GMM."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CorrelationLayer(nn.Module):
    """Compute L2-normalized cosine similarity between two feature maps.

    feat_cloth, feat_person: (B, C, H, W)
    Output: (B, H*W, H, W) cost volume — entry [b, j, y, x] is the cosine
    similarity between cloth feature at flat position j and person feature at
    spatial location (y, x).
    """

    def forward(self, feat_cloth: torch.Tensor, feat_person: torch.Tensor) -> torch.Tensor:
        assert feat_cloth.shape == feat_person.shape, \
            f"feature shape mismatch: {feat_cloth.shape} vs {feat_person.shape}"
        B, C, H, W = feat_cloth.shape

        # L2-normalize along channel dim (cosine similarity, scale-invariant)
        fc = F.normalize(feat_cloth.view(B, C, -1), dim=1)        # (B, C, H*W)
        fp = F.normalize(feat_person.view(B, C, -1), dim=1)        # (B, C, H*W)

        # cost[b, j, i] = <fc[:, j], fp[:, i]>  ->  (B, H*W, H*W)
        corr = torch.bmm(fc.permute(0, 2, 1), fp)
        return corr.view(B, H * W, H, W)
