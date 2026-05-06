"""Step 2 smoke test: forward a batch through MultiScaleGMM, check shapes."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.gmm import MultiScaleGMM


def main() -> None:
    cfg_path = Path(__file__).resolve().parents[1] / "configs" / "config.yaml"
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    H, W = cfg["data"]["resolution"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] device = {device}, resolution = {H}x{W}")

    model = MultiScaleGMM(
        H=H, W=W,
        cloth_in_ch=cfg["model"]["cloth_input_channels"],
        person_in_ch=cfg["model"]["person_rep_channels"],
        encoder_features=cfg["model"]["encoder_features"],
        coarse_grid=cfg["model"]["coarse_grid"],
        fine_grid=cfg["model"]["fine_grid"],
        regression_dropout=cfg["model"]["regression_dropout"],
    ).to(device)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[info] MultiScaleGMM params = {n_params:,}")

    B = 2
    cloth = torch.randn(B, 3, H, W, device=device)
    cloth_mask = torch.rand(B, 1, H, W, device=device)
    cloth_sem = torch.rand(B, 3, H, W, device=device)
    person_rep = torch.randn(B, cfg["model"]["person_rep_channels"], H, W, device=device)

    with torch.no_grad():
        warped_cloth, warped_mask, coarse_theta, fine_theta, coarse_warped = model(
            cloth, cloth_mask, cloth_sem, person_rep
        )

    print("\n=== GMM output shapes ===")
    print(f"  warped_cloth   = {tuple(warped_cloth.shape)}")
    print(f"  warped_mask    = {tuple(warped_mask.shape)}")
    print(f"  coarse_warped  = {tuple(coarse_warped.shape)}")
    print(f"  coarse_theta   = {tuple(coarse_theta.shape)}")
    print(f"  fine_theta     = {tuple(fine_theta.shape)}")

    # ---- Assertions ----
    assert warped_cloth.shape == (B, 3, H, W), warped_cloth.shape
    assert warped_mask.shape == (B, 1, H, W), warped_mask.shape
    assert coarse_warped.shape == (B, 3, H, W), coarse_warped.shape
    assert coarse_theta.shape == (B, 25, 2), coarse_theta.shape
    assert fine_theta.shape == (B, 100, 2), fine_theta.shape

    # Identity init should produce ~zero theta on the first forward pass
    print(f"\n  ||coarse_theta||_inf = {coarse_theta.abs().max().item():.3e}")
    print(f"  ||fine_theta||_inf   = {fine_theta.abs().max().item():.3e}")
    assert coarse_theta.abs().max().item() < 1e-4, "coarse_theta not identity-init"
    assert fine_theta.abs().max().item() < 1e-4, "fine_theta not identity-init"

    # With theta = 0, the TPS sampling grid should equal the target identity grid.
    # (Pixel-space diffs against random-noise cloth are dominated by aliasing on
    # any sub-pixel grid jitter, so we check the grid directly.)
    coarse_grid = model.coarse_tps(coarse_theta)              # (B, H, W, 2)
    target_x = torch.linspace(-1, 1, W, device=device)
    target_y = torch.linspace(-1, 1, H, device=device)
    gy, gx = torch.meshgrid(target_y, target_x, indexing="ij")
    identity_grid = torch.stack([gx, gy], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)
    grid_diff = (coarse_grid - identity_grid).abs().max().item()
    print(f"  ||coarse_grid - identity_grid||_inf = {grid_diff:.3e}")
    assert grid_diff < 1e-4, f"identity grid diff too large: {grid_diff}"

    # And on a smooth (constant-channel) input, grid_sample at identity should be exact-ish.
    smooth = torch.ones_like(cloth) * 0.5
    smooth_warp = torch.nn.functional.grid_sample(
        smooth, coarse_grid, mode="bilinear", padding_mode="border", align_corners=True
    )
    smooth_diff = (smooth_warp - smooth).abs().max().item()
    print(f"  ||identity warp on smooth input||_inf = {smooth_diff:.3e}")
    assert smooth_diff < 1e-5, f"identity warp diff on smooth input: {smooth_diff}"

    print("\n[OK] Step 2 smoke test passed.")


if __name__ == "__main__":
    main()
