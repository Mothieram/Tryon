"""
Lightweight `modules` package for engine-local SCHP fallback.

`engine/networks/*` imports symbols as:
    from modules import InPlaceABN, InPlaceABNSync

The original SCHP `modules` package builds custom C++/CUDA ops at import time.
For portability, this fallback provides API-compatible PyTorch versions.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as functional

ACT_RELU = "relu"
ACT_LEAKY_RELU = "leaky_relu"
ACT_ELU = "elu"
ACT_NONE = "none"


class ABN(nn.Module):
    """Activated BatchNorm compatible with SCHP interfaces."""

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        activation: str = ACT_LEAKY_RELU,
        slope: float = 0.01,
    ):
        super().__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps
        self.momentum = momentum
        self.activation = activation
        self.slope = slope

        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

    def forward(self, x):
        x = functional.batch_norm(
            x,
            self.running_mean,
            self.running_var,
            self.weight,
            self.bias,
            self.training,
            self.momentum,
            self.eps,
        )

        if self.activation == ACT_RELU:
            return functional.relu(x, inplace=True)
        if self.activation == ACT_LEAKY_RELU:
            return functional.leaky_relu(x, negative_slope=self.slope, inplace=True)
        if self.activation == ACT_ELU:
            return functional.elu(x, inplace=True)
        return x


class InPlaceABN(ABN):
    """API-compatible replacement for SCHP InPlaceABN."""


class InPlaceABNSync(ABN):
    """API-compatible replacement for SCHP InPlaceABNSync."""


__all__ = [
    "ABN",
    "InPlaceABN",
    "InPlaceABNSync",
    "ACT_RELU",
    "ACT_LEAKY_RELU",
    "ACT_ELU",
    "ACT_NONE",
]

