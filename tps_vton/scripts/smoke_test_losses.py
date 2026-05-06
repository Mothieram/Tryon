"""Step 3 smoke test: exercise every loss function with dummy tensors."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.losses import (
    EdgeLoss,
    GMMLossComputer,
    RefinementLossComputer,
    SobelEdge,
    VGGPerceptualLoss,
    grid_smoothness_loss,
    lsgan_d_loss,
    lsgan_g_loss,
    tps_regularization,
)


def _check_finite(name: str, t: torch.Tensor) -> None:
    assert torch.isfinite(t).all(), f"{name} contains NaN/Inf: {t}"
    print(f"  {name:<26} {float(t):.6f}")


def main() -> None:
    cfg_path = Path(__file__).resolve().parents[1] / "configs" / "config.yaml"
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] device = {device}")

    B, C, H, W = 2, 3, 64, 48
    pred = torch.rand(B, C, H, W, device=device, requires_grad=True)
    target = torch.rand(B, C, H, W, device=device)
    mask = torch.rand(B, 1, H, W, device=device)
    target_mask = torch.rand(B, 1, H, W, device=device)

    print("\n--- Individual losses ---")

    # VGG perceptual
    vgg = VGGPerceptualLoss().to(device).eval()
    p = vgg(pred, target)
    _check_finite("VGG perceptual", p)

    # Edge loss
    edge = EdgeLoss().to(device)
    e = edge(pred, target)
    _check_finite("Edge (Sobel)", e)

    # Sobel direct on a non-3 channel input
    s = SobelEdge().to(device)(mask)
    assert s.shape == (B, 2, H, W), s.shape
    print(f"  Sobel(1ch) shape           {tuple(s.shape)}")

    # Grid smoothness for 5x5 and 10x10
    theta5 = torch.zeros(B, 25, 2, device=device, requires_grad=True)
    theta10 = torch.randn(B, 100, 2, device=device) * 0.05
    s5 = grid_smoothness_loss(theta5, grid_size=5)
    s10 = grid_smoothness_loss(theta10, grid_size=10)
    _check_finite("grid_smoothness (5x5)", s5)
    _check_finite("grid_smoothness (10x10)", s10)
    assert s5.item() == 0.0, "smoothness on zero theta must be 0"

    # TPS regularization
    r = tps_regularization(theta10)
    _check_finite("tps_regularization", r)

    # LSGAN
    pred_fake = torch.randn(B, 1, 16, 12, device=device)
    pred_real = torch.randn(B, 1, 16, 12, device=device)
    g = lsgan_g_loss(pred_fake)
    d = lsgan_d_loss(pred_real, pred_fake)
    _check_finite("LSGAN G", g)
    _check_finite("LSGAN D", d)

    # ---- Aggregators ----
    print("\n--- GMMLossComputer ---")
    gmm_loss_fn = GMMLossComputer(cfg, coarse_grid=5, fine_grid=10).to(device)
    warped_cloth = torch.rand(B, 3, H, W, device=device, requires_grad=True)
    coarse_warped = torch.rand(B, 3, H, W, device=device, requires_grad=True)
    warped_mask = torch.rand(B, 1, H, W, device=device, requires_grad=True)
    target_cloth = torch.rand(B, 3, H, W, device=device)
    coarse_theta = torch.zeros(B, 25, 2, device=device, requires_grad=True)
    fine_theta = torch.zeros(B, 100, 2, device=device, requires_grad=True)
    target_cloth_mask = torch.rand(B, 1, H, W, device=device)

    total, parts = gmm_loss_fn(
        warped_cloth, warped_mask, coarse_theta, fine_theta, coarse_warped,
        target_cloth, target_cloth_mask, reg_weight=0.05,
    )
    _check_finite("GMM total", total)
    for k, v in parts.items():
        _check_finite(f"  {k}", v)

    # Backward should run without error
    total.backward()
    assert warped_cloth.grad is not None and torch.isfinite(warped_cloth.grad).all()
    assert coarse_theta.grad is not None and torch.isfinite(coarse_theta.grad).all()
    print("  [GMM backward OK]")

    print("\n--- RefinementLossComputer ---")
    refine_loss_fn = RefinementLossComputer(cfg).to(device)
    composed = torch.rand(B, 3, H, W, device=device, requires_grad=True)
    comp_mask = torch.rand(B, 1, H, W, device=device, requires_grad=True)
    target_image = torch.rand(B, 3, H, W, device=device)
    gt_comp_mask = (torch.rand(B, 1, H, W, device=device) > 0.5).float()
    fake = torch.randn(B, 1, 8, 6, device=device, requires_grad=True)

    total_g, parts_g = refine_loss_fn(composed, comp_mask, target_image, gt_comp_mask, pred_fake=fake)
    _check_finite("Refinement total", total_g)
    for k, v in parts_g.items():
        _check_finite(f"  {k}", v)
    total_g.backward()
    assert composed.grad is not None and torch.isfinite(composed.grad).all()
    print("  [Refinement backward OK]")

    print("\n[OK] Step 3 smoke test passed.")


if __name__ == "__main__":
    main()
