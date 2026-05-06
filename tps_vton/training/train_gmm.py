"""Stage 1: train the multi-scale GMM on VITON-HD."""

from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# allow `python training/train_gmm.py ...` and `python -m training.train_gmm`
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.dataset import VitonHDDataset, deterministic_train_val_split
from models.gmm import MultiScaleGMM
from models.losses import GMMLossComputer
from training.scheduler import RegWeightSchedule, build_lr_scheduler
from training.validator import ValidationTracker, load_checkpoint
from utils.helpers import (
    count_parameters,
    get_device,
    load_state_dict_compat,
    make_amp_scaler,
    maybe_data_parallel,
    set_seed,
    state_dict_for_save,
    unwrap,
)
from utils.logger import Logger
from utils.visualization import visualize_tps_grid, visualize_warp_result


# ------------------------------------------------------------------ utils

def _move_batch(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


def _gmm_forward(model: MultiScaleGMM, batch: Dict[str, torch.Tensor]):
    return model(
        batch["cloth"], batch["cloth_mask"], batch["cloth_sem_mask"], batch["person_rep"],
    )


def _val_forward(model: MultiScaleGMM, batch: Dict[str, torch.Tensor]):
    """Adapter for ValidationTracker: returns (pred, target) in [0, 1]."""
    device = next(model.parameters()).device
    batch = _move_batch(batch, device)
    warped, _, _, _, _ = _gmm_forward(model, batch)
    target = batch["target_cloth"]
    return warped, target


# ------------------------------------------------------------------ training phase

def _build_dataloaders(cfg: Dict, resolution, train_pairs, val_pairs, batch_size: int):
    cfg_resized = copy.deepcopy(cfg)
    cfg_resized["data"]["resolution"] = list(resolution)

    train_set = VitonHDDataset(cfg_resized, split="train", augment=True, pairs_override=train_pairs)
    val_set = VitonHDDataset(cfg_resized, split="val", augment=False, pairs_override=val_pairs)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
        drop_last=True,
        persistent_workers=cfg["data"]["num_workers"] > 0,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=max(1, batch_size // 2),
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
        persistent_workers=cfg["data"]["num_workers"] > 0,
    )
    return train_set, val_set, train_loader, val_loader


def _train_phase(
    cfg: Dict,
    resolution,
    epochs: int,
    base_lr: float,
    batch_size: int,
    train_pairs,
    val_pairs,
    device: torch.device,
    logger: Logger,
    tracker: ValidationTracker,
    init_state_dict: Optional[Dict[str, torch.Tensor]] = None,
    phase_name: str = "phase",
    global_step_start: int = 0,
    resume_ckpt: Optional[Dict[str, Any]] = None,
) -> tuple[MultiScaleGMM, int]:
    print(f"\n========== [{phase_name}] resolution={resolution} epochs={epochs} lr={base_lr:.1e} bs={batch_size} ==========")

    H, W = resolution
    train_set, val_set, train_loader, val_loader = _build_dataloaders(
        cfg, resolution, train_pairs, val_pairs, batch_size
    )
    print(f"  train batches = {len(train_loader)}, val batches = {len(val_loader)}")

    model = MultiScaleGMM(
        H=H, W=W,
        cloth_in_ch=cfg["model"]["cloth_input_channels"],
        person_in_ch=cfg["model"]["person_rep_channels"],
        encoder_features=cfg["model"]["encoder_features"],
        coarse_grid=cfg["model"]["coarse_grid"],
        fine_grid=cfg["model"]["fine_grid"],
        regression_dropout=cfg["model"]["regression_dropout"],
    )
    model = maybe_data_parallel(model, device)
    if init_state_dict is not None:
        # When switching resolution the TPS buffers re-derive from the new (H, W); load
        # the rest of the parameters (encoder weights, regression head) from the prior
        # phase's best checkpoint.
        missing, unexpected = load_state_dict_compat(model, init_state_dict, strict=False)
        print(f"  loaded checkpoint: missing={len(missing)} unexpected={len(unexpected)}")

    print(f"  params = {count_parameters(model):,}")

    opt_cfg = cfg["training"]["optimizer"]
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=base_lr,
        betas=tuple(opt_cfg["betas"]),
        weight_decay=opt_cfg.get("weight_decay", 0.0),
    )

    sched_cfg = cfg["training"]["scheduler"]
    scheduler = build_lr_scheduler(
        optimizer,
        warmup_epochs=sched_cfg["warmup_epochs"],
        total_epochs=epochs,
        steps_per_epoch=max(1, len(train_loader)),
        warmup_start_factor=sched_cfg["warmup_start_factor"],
        eta_min=sched_cfg["eta_min"],
    )

    scaler = make_amp_scaler(enabled=cfg["training"]["amp"]["enabled"])
    loss_fn = GMMLossComputer(
        cfg, coarse_grid=cfg["model"]["coarse_grid"], fine_grid=cfg["model"]["fine_grid"]
    ).to(device)
    reg_schedule = RegWeightSchedule(cfg["training"]["reg_warmup"])

    grad_clip = cfg["training"]["gradient_clip"]["max_norm"]
    image_log_interval = cfg["logging"]["image_log_interval"]
    val_interval = cfg["validation"]["interval"]
    log_individual = cfg["logging"]["log_individual_losses"]

    global_step = global_step_start
    last_loss = None
    start_epoch = 0

    # ---- Full resume: model + optimizer + scheduler + scaler + tracker.best + start_epoch ----
    if resume_ckpt is not None:
        if "optimizer_state_dict" in resume_ckpt and resume_ckpt["optimizer_state_dict"] is not None:
            optimizer.load_state_dict(resume_ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in resume_ckpt and resume_ckpt["scheduler_state_dict"] is not None:
            scheduler.load_state_dict(resume_ckpt["scheduler_state_dict"])
        if "scaler_state_dict" in resume_ckpt and resume_ckpt["scaler_state_dict"] is not None:
            scaler.load_state_dict(resume_ckpt["scaler_state_dict"])
        if "best" in resume_ckpt and resume_ckpt["best"] is not None:
            tracker.best.update(resume_ckpt["best"])
        start_epoch = int(resume_ckpt.get("epoch", -1)) + 1
        print(f"  resumed full state -> starting at epoch {start_epoch}, "
              f"best_lpips={tracker.best['lpips']:.4f}, best_ssim={tracker.best['ssim']:.4f}")

    for epoch in range(start_epoch, epochs):
        reg_weight = reg_schedule.value(epoch)
        epoch_start = time.time()
        epoch_loss_sum = 0.0
        n_batches = 0

        model.train()
        pbar = tqdm(
            train_loader,
            desc=f"[{phase_name} ep{epoch:03d}/{epochs - 1}]",
            leave=False, dynamic_ncols=True,
        )
        for batch in pbar:
            batch = _move_batch(batch, device)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=cfg["training"]["amp"]["enabled"]):
                warped_cloth, warped_mask, coarse_theta, fine_theta, coarse_warped = _gmm_forward(model, batch)
                loss, parts = loss_fn(
                    warped_cloth, warped_mask, coarse_theta, fine_theta, coarse_warped,
                    target_cloth=batch["target_cloth"],
                    target_mask=batch["target_mask"],
                    reg_weight=reg_weight,
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            global_step += 1
            epoch_loss_sum += float(loss.detach())
            n_batches += 1
            last_loss = float(loss.detach())

            pbar.set_postfix(
                loss=f"{last_loss:.4f}",
                avg=f"{epoch_loss_sum / n_batches:.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                reg=f"{reg_weight:.3f}",
            )

            # ---- Per-step logging ----
            logger.scalar(f"{phase_name}/train/loss", last_loss, global_step)
            logger.lr(optimizer, global_step, tag=f"{phase_name}/train/lr")
            if log_individual:
                logger.scalars(f"{phase_name}/train/loss_parts", parts, global_step)

            # Image logging
            if cfg["logging"]["log_tps_grid"] and global_step % image_log_interval == 0:
                with torch.no_grad():
                    grid_c = visualize_tps_grid(coarse_theta, grid_size=cfg["model"]["coarse_grid"])
                    grid_f = visualize_tps_grid(fine_theta, grid_size=cfg["model"]["fine_grid"])
                    sxs = visualize_warp_result(
                        batch["cloth"], warped_cloth, batch["target_cloth"], batch["image"], n_samples=2,
                    )
                logger.image(f"{phase_name}/viz/coarse_tps_grid", grid_c, global_step)
                logger.image(f"{phase_name}/viz/fine_tps_grid", grid_f, global_step)
                logger.image(f"{phase_name}/viz/cloth_warp_target_person", sxs, global_step)

        epoch_avg = epoch_loss_sum / max(1, n_batches)
        elapsed = time.time() - epoch_start
        print(f"  [Epoch {epoch:3d}] avg_loss={epoch_avg:.4f}  reg_weight={reg_weight:.4f}  ({elapsed:.1f}s)")
        logger.scalar(f"{phase_name}/train/epoch_loss", epoch_avg, epoch)
        logger.scalar(f"{phase_name}/train/reg_weight", reg_weight, epoch)

        # ---- Validation + best-only checkpoint ----
        if (epoch + 1) % val_interval == 0 or epoch == epochs - 1:
            metrics = tracker.validate(model, val_loader, _val_forward)
            tracker.log_metrics(epoch, metrics)
            for k, v in metrics.items():
                logger.scalar(f"{phase_name}/val/{k}", v, epoch)
            improved = tracker.save_if_best(
                model, optimizer, scheduler, scaler,
                epoch=epoch, metrics=metrics, config=cfg,
                extra={"resolution": list(resolution), "phase": phase_name},
            )
            if improved:
                print(f"    [best] checkpoint updated at epoch {epoch}")
            if tracker.should_stop():
                print(f"  Early stopping at epoch {epoch}")
                break

        torch.cuda.empty_cache()

    return model, global_step


# ------------------------------------------------------------------ main entry

def train_gmm(cfg_path: str, resume_from: Optional[str] = None, smoke: bool = False) -> None:
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["training"]["seed"])
    device = get_device(cfg["training"].get("device"))
    print(f"[info] device = {device}")

    # ---- Resolve dataset paths ----
    repo_root = Path(__file__).resolve().parents[2]
    data_root = Path(cfg["data"]["root"])
    if not data_root.is_absolute():
        data_root = (repo_root / data_root).resolve()
    cfg["data"]["root"] = str(data_root)

    train_pairs_file = data_root / cfg["data"]["train_pairs"]
    if not train_pairs_file.exists():
        alt = repo_root / cfg["data"]["train_pairs"]
        if alt.exists():
            train_pairs_file = alt
    val_pairs_file = data_root / cfg["data"]["val_pairs"]

    train_pairs, val_pairs = deterministic_train_val_split(
        train_pairs_file, val_pairs_file,
        val_split=cfg["data"]["val_split"],
        seed=cfg["training"]["seed"],
    )

    if smoke:
        # Tiny subset for the Step 5 smoke test
        train_pairs = train_pairs[:16]
        val_pairs = val_pairs[:4]
        cfg["training"]["gmm_epochs"] = 2
        cfg["training"]["batch_size"] = 4
        cfg["validation"]["interval"] = 1
        cfg["training"]["progressive_resolution"]["switch_after_plateau"] = False
        cfg["logging"]["image_log_interval"] = 1
        cfg["data"]["num_workers"] = 0          # Windows multiprocessing flakes with tiny sets
        print(f"[smoke] train pairs = {len(train_pairs)}, val pairs = {len(val_pairs)}")

    print(f"[info] train pairs = {len(train_pairs)}, val pairs = {len(val_pairs)}")

    # ---- Logger + tracker ----
    run_name = f"gmm_{int(time.time())}"
    log_dir = Path(cfg["logging"]["log_dir"]) / run_name
    ckpt_dir = Path(cfg["logging"]["ckpt_dir"]) / run_name
    if not log_dir.is_absolute():
        log_dir = (repo_root / log_dir).resolve()
    if not ckpt_dir.is_absolute():
        ckpt_dir = (repo_root / ckpt_dir).resolve()
    print(f"[info] log_dir  = {log_dir}")
    print(f"[info] ckpt_dir = {ckpt_dir}")

    logger = Logger(log_dir)
    tracker = ValidationTracker(
        save_dir=ckpt_dir,
        primary_metric=cfg["validation"]["primary_metric"],
        primary_metric_mode=cfg["validation"]["primary_metric_mode"],
        early_stopping_patience=cfg["validation"]["early_stopping_patience"],
        device=device,
    )

    # ---- Optional resume from a best checkpoint (full state) ----
    init_state = None
    resume_ckpt = None
    if resume_from is not None:
        resume_ckpt = torch.load(resume_from, map_location="cpu")
        init_state = resume_ckpt["model_state_dict"]
        print(f"[info] resuming from {resume_from} (epoch={resume_ckpt.get('epoch', '?')})")

    # ---- Phase 1: low-res ----
    low_res = tuple(cfg["data"]["resolution"])
    base_lr = cfg["training"]["optimizer"]["lr"]
    bs = cfg["training"]["batch_size"]

    model, global_step = _train_phase(
        cfg, resolution=low_res, epochs=cfg["training"]["gmm_epochs"],
        base_lr=base_lr, batch_size=bs,
        train_pairs=train_pairs, val_pairs=val_pairs,
        device=device, logger=logger, tracker=tracker,
        init_state_dict=init_state, phase_name="phase1_lowres", global_step_start=0,
        resume_ckpt=resume_ckpt,
    )

    # ---- Phase 2: high-res (optional, skipped in smoke mode) ----
    pr = cfg["training"]["progressive_resolution"]
    if pr.get("switch_after_plateau", False) and not smoke:
        # Reload best-LPIPS weights from phase 1 (encoder + regression heads).
        best_path = ckpt_dir / "best_lpips.pth"
        if best_path.exists():
            best_state = torch.load(best_path, map_location="cpu")["model_state_dict"]
        else:
            best_state = state_dict_for_save(model)

        high_res = tuple(cfg["data"]["high_resolution"])
        hr_lr = base_lr * pr["high_res_lr_factor"]
        hr_bs = max(1, bs // 2)

        # Reset early-stopping for the new phase.
        tracker.wait = 0

        _train_phase(
            cfg, resolution=high_res, epochs=max(15, cfg["training"]["gmm_epochs"] // 3),
            base_lr=hr_lr, batch_size=hr_bs,
            train_pairs=train_pairs, val_pairs=val_pairs,
            device=device, logger=logger, tracker=tracker,
            init_state_dict=best_state, phase_name="phase2_highres",
            global_step_start=global_step,
        )

    logger.close()
    print("\n[done] GMM training complete.")
    print(f"  best LPIPS = {tracker.best['lpips']}")
    print(f"  best SSIM  = {tracker.best['ssim']}")
    print(f"  checkpoints in {ckpt_dir}")


# ------------------------------------------------------------------ CLI

def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--smoke", action="store_true",
                   help="2-epoch run on 16 train / 4 val samples (Step 5 smoke test)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train_gmm(args.config, resume_from=args.resume, smoke=args.smoke)
