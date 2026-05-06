"""TPS grid generator for warping cloth onto a person via grid_sample."""

from __future__ import annotations

import torch
import torch.nn as nn


def _U(r_sq: torch.Tensor) -> torch.Tensor:
    """Radial basis: U(r) = r^2 * log(r^2). Safe at r=0 because r^2 -> 0 dominates."""
    return r_sq * torch.log(r_sq + 1e-6)


class TPSGridGen(nn.Module):
    """Thin-plate-spline grid generator.

    Given per-control-point offsets `theta` (B, N, 2) in normalized [-1, 1]
    coordinates, produces a (B, out_h, out_w, 2) sampling grid suitable for
    F.grid_sample with align_corners=True.

    The N target control points sit on a regular grid_size x grid_size grid in
    [-1, 1]; the predicted offsets define where in the source image each
    target control point should sample from. The TPS interpolant smoothly
    extrapolates this displacement field to all pixels of the output grid.
    """

    def __init__(self, out_h: int, out_w: int, grid_size: int):
        super().__init__()
        self.out_h = out_h
        self.out_w = out_w
        self.grid_size = grid_size
        self.N = grid_size * grid_size

        # ---- Target control points P (N, 2) on a regular grid in [-1, 1] ----
        axis = torch.linspace(-1.0, 1.0, grid_size, dtype=torch.float32)
        gy, gx = torch.meshgrid(axis, axis, indexing="ij")
        P = torch.stack([gx.flatten(), gy.flatten()], dim=1)        # (N, 2)
        self.register_buffer("P", P)

        # ---- Inverse of the TPS kernel matrix L (N+3, N+3) ----
        L = self._build_L(P)
        self.register_buffer("L_inv", torch.inverse(L))

        # ---- Target sampling grid pixels & precomputed kernel evaluation M ----
        ty = torch.linspace(-1.0, 1.0, out_h, dtype=torch.float32)
        tx = torch.linspace(-1.0, 1.0, out_w, dtype=torch.float32)
        gy, gx = torch.meshgrid(ty, tx, indexing="ij")
        target_grid = torch.stack([gx, gy], dim=-1)                 # (H, W, 2)
        flat = target_grid.view(-1, 2)                              # (HW, 2)

        diff = flat.unsqueeze(1) - P.unsqueeze(0)                   # (HW, N, 2)
        dist_sq = (diff ** 2).sum(-1)                               # (HW, N)
        K_part = _U(dist_sq)                                        # (HW, N)
        ones = torch.ones(flat.shape[0], 1, dtype=torch.float32)
        M = torch.cat([K_part, ones, flat], dim=1)                  # (HW, N+3)
        self.register_buffer("M", M)

    @staticmethod
    def _build_L(P: torch.Tensor) -> torch.Tensor:
        N = P.shape[0]
        diff = P.unsqueeze(0) - P.unsqueeze(1)                      # (N, N, 2)
        dist_sq = (diff ** 2).sum(-1)
        K = _U(dist_sq)                                             # (N, N)
        ones = torch.ones(N, 1, dtype=P.dtype)
        P_hat = torch.cat([ones, P], dim=1)                         # (N, 3)

        top = torch.cat([K, P_hat], dim=1)                          # (N, N+3)
        bottom = torch.cat(
            [P_hat.t(), torch.zeros(3, 3, dtype=P.dtype)], dim=1
        )                                                           # (3, N+3)
        return torch.cat([top, bottom], dim=0)                      # (N+3, N+3)

    def forward(self, theta: torch.Tensor) -> torch.Tensor:
        # theta: (B, N, 2) — offsets to target control points
        assert theta.ndim == 3 and theta.shape[1] == self.N and theta.shape[2] == 2, \
            f"TPSGridGen expects theta of shape (B, {self.N}, 2), got {theta.shape}"
        B = theta.shape[0]

        # Source control points: target + offset
        Y = self.P.unsqueeze(0) + theta                             # (B, N, 2)

        # Pad with 3 zero rows for the affine constraint block
        pad = torch.zeros(B, 3, 2, dtype=theta.dtype, device=theta.device)
        Y_padded = torch.cat([Y, pad], dim=1)                       # (B, N+3, 2)

        # Solve for the TPS coefficients [w; A]
        W = torch.matmul(self.L_inv.unsqueeze(0), Y_padded)         # (B, N+3, 2)

        # Evaluate at every output pixel
        grid_flat = torch.matmul(self.M.unsqueeze(0), W)            # (B, HW, 2)
        return grid_flat.view(B, self.out_h, self.out_w, 2)
