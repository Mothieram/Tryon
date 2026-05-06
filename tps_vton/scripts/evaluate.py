"""Full evaluation: SSIM, LPIPS, FID, L1 on the VITON-HD test set."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.utils as vutils
import yaml
from skimage.metrics import structural_similarity as sk_ssim
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.dataset import VitonHDDataset
from models.gmm import MultiScaleGMM
from models.refinement import RefinementUNet, compose_output
from training.validator import load_checkpoint
from utils.helpers import get_device, set_seed

try:
    import lpips as _lpips_pkg
except ImportError:
    _lpips_pkg = None

try:
    from torchmetrics.image.fid import FrechetInceptionDistance
except ImportError:
    FrechetInceptionDistance = None


def _move(batch, device):
    return {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()}


def _ssim_batch(pred: torch.Tensor, target: torch.Tensor) -> List[float]:
    p = pred.detach().cpu().numpy().transpose(0, 2, 3, 1).astype(np.float32)
    t = target.detach().cpu().numpy().transpose(0, 2, 3, 1).astype(np.float32)
    return [float(sk_ssim(p[i], t[i], channel_axis=-1, data_range=1.0)) for i in range(p.shape[0])]


@torch.no_grad()
def evaluate(
    cfg_path: str,
    gmm_checkpoint: str,
    refine_checkpoint: str,
    out_dir: str = "./eval_outputs",
    save_samples: int = 16,
    max_batches: int = -1,
) -> None:
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["training"]["seed"])
    device = get_device(cfg["training"].get("device"))
    print(f"[info] device = {device}")

    repo_root = Path(__file__).resolve().parents[2]
    data_root = Path(cfg["data"]["root"])
    if not data_root.is_absolute():
        data_root = (repo_root / data_root).resolve()
    cfg["data"]["root"] = str(data_root)

    # Find test_pairs.txt — may live at the dataset root or repo root.
    test_pairs_file = data_root / cfg["data"]["test_pairs"]
    if not test_pairs_file.exists():
        alt = repo_root / cfg["data"]["test_pairs"]
        if alt.exists():
            test_pairs_file = alt
    print(f"[info] test_pairs file = {test_pairs_file}")

    from data.dataset import _read_pairs
    test_pairs = _read_pairs(test_pairs_file)
    test_set = VitonHDDataset(cfg, split="test", augment=False, pairs_override=test_pairs)
    test_loader = DataLoader(
        test_set, batch_size=max(1, cfg["training"]["batch_size"] // 2),
        shuffle=False, num_workers=cfg["data"]["num_workers"], pin_memory=True,
    )
    print(f"[info] test pairs = {len(test_set)}")

    H, W = cfg["data"]["resolution"]

    # ---- Models ----
    gmm = MultiScaleGMM(
        H=H, W=W,
        cloth_in_ch=cfg["model"]["cloth_input_channels"],
        person_in_ch=cfg["model"]["person_rep_channels"],
        encoder_features=cfg["model"]["encoder_features"],
        coarse_grid=cfg["model"]["coarse_grid"],
        fine_grid=cfg["model"]["fine_grid"],
        regression_dropout=cfg["model"]["regression_dropout"],
    ).to(device)
    load_checkpoint(gmm_checkpoint, gmm, map_location=device, strict=False)
    gmm.eval()

    refine = RefinementUNet(
        in_ch=7,
        features=cfg["model"]["encoder_features"],
        gmm_features=cfg["model"]["encoder_features"],
    ).to(device)
    load_checkpoint(refine_checkpoint, refine, map_location=device, strict=False)
    refine.eval()
    print(f"[info] models loaded")

    # ---- Metric scaffolding ----
    lpips_fn = None
    if _lpips_pkg is not None:
        try:
            lpips_fn = _lpips_pkg.LPIPS(net="alex").to(device).eval()
            for p in lpips_fn.parameters():
                p.requires_grad = False
        except Exception as exc:
            print(f"[warn] LPIPS init failed: {exc}")

    fid = None
    if FrechetInceptionDistance is not None:
        try:
            fid = FrechetInceptionDistance(feature=2048, normalize=False).to(device)
        except Exception as exc:
            print(f"[warn] FID init failed: {exc}; FID will be skipped.")
            fid = None
    else:
        print("[warn] torchmetrics FID not installed; FID will be skipped.")

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    all_lpips: List[float] = []
    all_ssim: List[float] = []
    all_l1: List[float] = []
    saved = 0

    for bidx, batch in enumerate(test_loader):
        if max_batches > 0 and bidx >= max_batches:
            break
        batch = _move(batch, device)

        wc, wm, _, _, _ = gmm(batch["cloth"], batch["cloth_mask"], batch["cloth_sem_mask"], batch["person_rep"])
        gp = gmm.encode_person(batch["person_rep"])
        r, cm = refine(wc, batch["image"], batch["agnostic_flow_mask"], gmm_feats=gp)
        composed = compose_output(r, batch["image"], cm).clamp(0, 1)
        target = batch["image"].clamp(0, 1)

        if lpips_fn is not None:
            lp = lpips_fn(composed * 2 - 1, target * 2 - 1)
            all_lpips.extend([float(x) for x in lp.flatten()])

        all_ssim.extend(_ssim_batch(composed, target))
        all_l1.append(F.l1_loss(composed, target).item())

        if fid is not None:
            fid.update((target * 255).clamp(0, 255).to(torch.uint8), real=True)
            fid.update((composed * 255).clamp(0, 255).to(torch.uint8), real=False)

        # Save a few samples to disk
        if saved < save_samples:
            n = min(composed.size(0), save_samples - saved)
            tile = torch.cat([
                batch["cloth"][:n].cpu(),
                wc[:n].cpu(),
                composed[:n].cpu(),
                target[:n].cpu(),
            ], dim=0)
            grid = vutils.make_grid(tile.clamp(0, 1), nrow=n, padding=2)
            vutils.save_image(grid, out_path / f"samples_batch{bidx:03d}.png")
            saved += n

    # ---- Aggregate ----
    results = {
        "LPIPS": float(np.mean(all_lpips)) if all_lpips else float("nan"),
        "SSIM": float(np.mean(all_ssim)),
        "L1": float(np.mean(all_l1)),
        "FID": float(fid.compute().item()) if fid is not None else float("nan"),
    }

    # Pretty-print
    print("\n========== VITON-HD Test Results ==========")
    print(f"  LPIPS  : {results['LPIPS']:.4f}   (lower is better)")
    print(f"  SSIM   : {results['SSIM']:.4f}   (higher is better)")
    print(f"  FID    : {results['FID']:.4f}   (lower is better)")
    print(f"  L1     : {results['L1']:.4f}")
    print("===========================================")
    print(f"  sample tiles saved to {out_path.resolve()}")

    # Reference benchmarks (from architecture doc)
    print("\nTarget benchmarks (VITON-HD):")
    print("  SSIM   : >0.80 acceptable | >0.85 good | >0.88 excellent")
    print("  LPIPS  : <0.15 acceptable | <0.10 good | <0.07 excellent")
    print("  FID    : <15  acceptable | <10  good | <8   excellent")


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--gmm_checkpoint", type=str, required=True)
    p.add_argument("--refine_checkpoint", type=str, required=True)
    p.add_argument("--out_dir", type=str, default="./eval_outputs")
    p.add_argument("--save_samples", type=int, default=16)
    p.add_argument("--max_batches", type=int, default=-1)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    evaluate(
        args.config,
        gmm_checkpoint=args.gmm_checkpoint,
        refine_checkpoint=args.refine_checkpoint,
        out_dir=args.out_dir,
        save_samples=args.save_samples,
        max_batches=args.max_batches,
    )
