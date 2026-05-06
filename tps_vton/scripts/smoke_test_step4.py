"""Step 4 smoke test: scheduler values, viz outputs, checkpoint roundtrip."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.scheduler import RegWeightSchedule, build_lr_scheduler
from training.validator import ValidationTracker, load_checkpoint
from utils.helpers import make_amp_scaler, set_seed
from utils.logger import Logger
from utils.visualization import (
    visualize_comp_mask,
    visualize_tps_grid,
    visualize_warp_result,
)


def main() -> None:
    cfg_path = Path(__file__).resolve().parents[1] / "configs" / "config.yaml"
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["training"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------- LR scheduler ----------
    print("--- LR scheduler ---")
    base_lr = cfg["training"]["optimizer"]["lr"]
    model = torch.nn.Linear(4, 4)
    opt = torch.optim.Adam(model.parameters(), lr=base_lr,
                           betas=cfg["training"]["optimizer"]["betas"])

    warmup_epochs = cfg["training"]["scheduler"]["warmup_epochs"]
    total_epochs = 5
    steps_per_epoch = 10
    sched = build_lr_scheduler(
        opt,
        warmup_epochs=warmup_epochs,
        total_epochs=total_epochs,
        steps_per_epoch=steps_per_epoch,
        warmup_start_factor=cfg["training"]["scheduler"]["warmup_start_factor"],
        eta_min=cfg["training"]["scheduler"]["eta_min"],
    )

    lrs = []
    for _ in range(total_epochs * steps_per_epoch):
        opt.step()
        lrs.append(opt.param_groups[0]["lr"])
        sched.step()

    print(f"  base_lr             = {base_lr:.3e}")
    print(f"  lr at step 0        = {lrs[0]:.3e}  (expect ~ base_lr * {cfg['training']['scheduler']['warmup_start_factor']})")
    print(f"  lr at step warmup-1 = {lrs[warmup_epochs * steps_per_epoch - 1]:.3e}")
    print(f"  lr at end           = {lrs[-1]:.3e}  (eta_min = {cfg['training']['scheduler']['eta_min']:.0e})")

    expected_start = base_lr * cfg["training"]["scheduler"]["warmup_start_factor"]
    assert abs(lrs[0] - expected_start) < 1e-9, f"warmup start wrong: {lrs[0]} vs {expected_start}"
    assert lrs[-1] <= base_lr + 1e-9, "final LR exceeds base"
    assert lrs[-1] < lrs[warmup_epochs * steps_per_epoch], "cosine should decrease after warmup"

    # ---------- RegWeight schedule ----------
    print("\n--- RegWeight schedule ---")
    rw = RegWeightSchedule(cfg["training"]["reg_warmup"])
    seen = [(e, rw.value(e)) for e in [0, 5, 10, 50]]
    for e, w in seen:
        print(f"  epoch {e:2d}  reg_weight = {w:.4f}")
    assert abs(rw.value(0) - 0.1) < 1e-9
    assert abs(rw.value(10) - 0.01) < 1e-9
    assert abs(rw.value(50) - 0.01) < 1e-9
    assert rw.value(5) > rw.value(10)

    # ---------- Visualization ----------
    print("\n--- Visualization ---")
    theta5 = torch.zeros(2, 25, 2) + 0.05 * torch.randn(2, 25, 2)
    theta10 = torch.zeros(2, 100, 2) + 0.02 * torch.randn(2, 100, 2)
    grid_img_5 = visualize_tps_grid(theta5, grid_size=5)
    grid_img_10 = visualize_tps_grid(theta10, grid_size=10)
    print(f"  TPS-grid 5x5  image shape  = {tuple(grid_img_5.shape)}")
    print(f"  TPS-grid 10x10 image shape = {tuple(grid_img_10.shape)}")
    assert grid_img_5.dim() == 3 and grid_img_5.shape[0] == 3

    cloth = torch.rand(4, 3, 64, 48)
    warped = torch.rand(4, 3, 64, 48)
    target = torch.rand(4, 3, 64, 48)
    person = torch.rand(4, 3, 64, 48)
    side_by_side = visualize_warp_result(cloth, warped, target, person, n_samples=2)
    print(f"  side-by-side shape         = {tuple(side_by_side.shape)}")
    assert side_by_side.shape == (3, 2 * 64, 4 * 48), side_by_side.shape

    cm = torch.rand(2, 1, 64, 48)
    cm_img = visualize_comp_mask(cm)
    print(f"  comp-mask heatmap shape    = {tuple(cm_img.shape)}")

    # ---------- TensorBoard logger ----------
    with tempfile.TemporaryDirectory() as tdir:
        logger = Logger(tdir, run_name="smoketest")
        logger.scalar("train/loss", 0.5, 0)
        logger.scalars("loss", {"a": 1.0, "b": torch.tensor(2.0)}, 0)
        logger.image("viz/grid5", grid_img_5, 0)
        logger.images("viz", {"warp": side_by_side, "compmask": cm_img}, 0)
        logger.lr(opt, 0)
        logger.close()
        # event file should exist
        event_files = list(Path(tdir).rglob("events.out.tfevents.*"))
        print(f"  TB event files: {len(event_files)}")
        assert event_files, "No TensorBoard event file written"

    # ---------- ValidationTracker + checkpoint roundtrip ----------
    print("\n--- ValidationTracker / checkpoint roundtrip ---")
    with tempfile.TemporaryDirectory() as tdir:
        tracker = ValidationTracker(
            save_dir=tdir,
            primary_metric=cfg["validation"]["primary_metric"],
            primary_metric_mode=cfg["validation"]["primary_metric_mode"],
            early_stopping_patience=cfg["validation"]["early_stopping_patience"],
            device=device,
        )

        m = torch.nn.Conv2d(3, 3, 1).to(device)
        opt2 = torch.optim.Adam(m.parameters(), lr=1e-4)
        scaler = make_amp_scaler(enabled=False)

        # Fake validation: pred close to target (good metrics)
        fake_metrics = {"lpips": 0.05, "ssim": 0.9, "l1": 0.02}
        improved = tracker.save_if_best(
            m, opt2, scheduler=None, scaler=scaler,
            epoch=0, metrics=fake_metrics, config=cfg,
        )
        print(f"  improved on epoch 0 = {improved}")
        assert improved
        assert (Path(tdir) / "best_lpips.pth").exists()
        assert (Path(tdir) / "best_ssim.pth").exists()
        assert (Path(tdir) / "last.pth").exists()

        # Worse metrics — wait should increment, no new best
        worse = {"lpips": 0.10, "ssim": 0.85, "l1": 0.05}
        tracker.save_if_best(m, opt2, None, scaler, 1, worse, cfg)
        assert tracker.wait == 1, tracker.wait
        print(f"  wait after worse epoch = {tracker.wait}")

        # Roundtrip restore
        m2 = torch.nn.Conv2d(3, 3, 1).to(device)
        opt3 = torch.optim.Adam(m2.parameters(), lr=1e-4)
        ckpt = load_checkpoint(
            Path(tdir) / "best_lpips.pth",
            m2, optimizer=opt3, scaler=scaler, map_location=device,
        )
        for p1, p2 in zip(m.parameters(), m2.parameters()):
            assert torch.allclose(p1, p2), "parameter mismatch after load"
        assert ckpt["epoch"] == 0
        print("  checkpoint roundtrip OK")

    print("\n[OK] Step 4 smoke test passed.")


if __name__ == "__main__":
    main()
