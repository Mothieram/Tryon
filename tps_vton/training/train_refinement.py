"""Stage 2: train the UNet refinement network with a PatchGAN discriminator."""

from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.dataset import VitonHDDataset, deterministic_train_val_split
from models.discriminator import PatchDiscriminator
from models.gmm import MultiScaleGMM
from models.losses import RefinementLossComputer, lsgan_d_loss
from models.refinement import RefinementUNet, compose_output
from training.scheduler import build_lr_scheduler
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
from utils.visualization import visualize_comp_mask, visualize_warp_result


# ------------------------------------------------------------------ helpers

def _move(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
    return out


def _build_dataloaders(cfg: Dict, train_pairs, val_pairs, batch_size: int):
    train_set = VitonHDDataset(cfg, split="train", augment=True, pairs_override=train_pairs)
    val_set = VitonHDDataset(cfg, split="val", augment=False, pairs_override=val_pairs)
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=cfg["data"]["num_workers"], pin_memory=True, drop_last=True,
        persistent_workers=cfg["data"]["num_workers"] > 0,
    )
    val_loader = DataLoader(
        val_set, batch_size=max(1, batch_size // 2), shuffle=False,
        num_workers=cfg["data"]["num_workers"], pin_memory=True,
        persistent_workers=cfg["data"]["num_workers"] > 0,
    )
    return train_loader, val_loader


# ------------------------------------------------------------------ main

def train_refinement(
    cfg_path: str,
    gmm_checkpoint: Optional[str],
    resume_from: Optional[str] = None,
    smoke: bool = False,
) -> None:
    if not gmm_checkpoint:
        raise SystemExit("--gmm_checkpoint is required for refinement training.")
    if not Path(gmm_checkpoint).exists():
        raise SystemExit(f"GMM checkpoint not found: {gmm_checkpoint}")

    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["training"]["seed"])
    device = get_device(cfg["training"].get("device"))
    print(f"[info] device = {device}")

    # ---- resolve dataset paths ----
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
        val_split=cfg["data"]["val_split"], seed=cfg["training"]["seed"],
    )

    if smoke:
        train_pairs = train_pairs[:16]
        val_pairs = val_pairs[:4]
        cfg["training"]["refine_epochs"] = 2
        cfg["training"]["batch_size"] = 4
        cfg["validation"]["interval"] = 1
        cfg["data"]["num_workers"] = 0
        cfg["logging"]["image_log_interval"] = 1
        print(f"[smoke] train pairs = {len(train_pairs)}, val pairs = {len(val_pairs)}")

    bs = cfg["training"]["batch_size"]
    train_loader, val_loader = _build_dataloaders(cfg, train_pairs, val_pairs, bs)
    H, W = cfg["data"]["resolution"]

    # ---- frozen GMM ----
    gmm = MultiScaleGMM(
        H=H, W=W,
        cloth_in_ch=cfg["model"]["cloth_input_channels"],
        person_in_ch=cfg["model"]["person_rep_channels"],
        encoder_features=cfg["model"]["encoder_features"],
        coarse_grid=cfg["model"]["coarse_grid"],
        fine_grid=cfg["model"]["fine_grid"],
        regression_dropout=cfg["model"]["regression_dropout"],
    )
    gmm = maybe_data_parallel(gmm, device)
    load_checkpoint(gmm_checkpoint, gmm, map_location=device, strict=False)
    gmm.eval()
    for p in gmm.parameters():
        p.requires_grad = False
    print(f"[info] frozen GMM loaded from {gmm_checkpoint}")

    # ---- refinement + discriminator ----
    refine = RefinementUNet(
        in_ch=7,
        features=cfg["model"]["encoder_features"],
        gmm_features=cfg["model"]["encoder_features"],
    )
    refine = maybe_data_parallel(refine, device)
    disc = PatchDiscriminator(
        in_ch=cfg["discriminator"]["in_channels"],
        features=cfg["discriminator"]["features"],
    )
    disc = maybe_data_parallel(disc, device)
    print(f"[info] refine params = {count_parameters(refine):,}")
    print(f"[info] disc   params = {count_parameters(disc):,}")

    # ---- optimizers + schedulers + scalers ----
    g_cfg = cfg["training"]["optimizer"]
    d_cfg = cfg["training"]["discriminator_optimizer"]
    opt_g = torch.optim.Adam(refine.parameters(), lr=g_cfg["lr"], betas=tuple(g_cfg["betas"]))
    opt_d = torch.optim.Adam(disc.parameters(), lr=d_cfg["lr"], betas=tuple(d_cfg["betas"]))

    sched_cfg = cfg["training"]["scheduler"]
    epochs = cfg["training"]["refine_epochs"]
    sched_g = build_lr_scheduler(
        opt_g, warmup_epochs=max(1, sched_cfg["warmup_epochs"] // 2),
        total_epochs=epochs, steps_per_epoch=max(1, len(train_loader)),
        warmup_start_factor=sched_cfg["warmup_start_factor"], eta_min=sched_cfg["eta_min"],
    )

    scaler_g = make_amp_scaler(enabled=cfg["training"]["amp"]["enabled"])
    scaler_d = make_amp_scaler(enabled=cfg["training"]["amp"]["enabled"])
    refine_loss = RefinementLossComputer(cfg).to(device)

    # ---- logger + tracker ----
    run_name = "refine"
    log_dir = (repo_root / cfg["logging"]["log_dir"] / run_name).resolve()
    ckpt_dir = (repo_root / cfg["logging"]["ckpt_dir"] / run_name).resolve()
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

    start_epoch = 0
    if resume_from is None:
        auto = ckpt_dir / "last.pth"
        if auto.exists():
            resume_from = str(auto)
            print(f"[info] auto-resuming from {auto}")
    if resume_from is not None:
        ck = torch.load(resume_from, map_location="cpu")
        load_state_dict_compat(refine, ck["model_state_dict"], strict=False)
        if "discriminator_state_dict" in ck:
            load_state_dict_compat(disc, ck["discriminator_state_dict"], strict=False)
        if ck.get("optimizer_state_dict") is not None:
            opt_g.load_state_dict(ck["optimizer_state_dict"])
        if ck.get("discriminator_optimizer_state_dict") is not None:
            opt_d.load_state_dict(ck["discriminator_optimizer_state_dict"])
        if ck.get("scheduler_state_dict") is not None:
            sched_g.load_state_dict(ck["scheduler_state_dict"])
        if ck.get("scaler_state_dict") is not None:
            scaler_g.load_state_dict(ck["scaler_state_dict"])
        if ck.get("best") is not None:
            tracker.best.update(ck["best"])
        start_epoch = int(ck.get("epoch", -1)) + 1
        print(f"[info] resumed refinement from {resume_from} -> starting at epoch {start_epoch}")

    # ---- D-stability watchdog state ----
    d_lr_halved = False
    grad_clip = cfg["training"]["gradient_clip"]["max_norm"]
    image_log_interval = cfg["logging"]["image_log_interval"]
    val_interval = cfg["validation"]["interval"]
    log_individual = cfg["logging"]["log_individual_losses"]

    global_step = 0

    for epoch in range(start_epoch, epochs):
        epoch_start = time.time()
        sum_g = sum_d = 0.0
        sum_g_gan = 0.0
        n_batches = 0

        refine.train()
        disc.train()
        pbar = tqdm(
            train_loader,
            desc=f"[refine ep{epoch:03d}/{epochs - 1}]",
            leave=False, dynamic_ncols=True,
        )
        for batch in pbar:
            batch = _move(batch, device)
            person_image = batch["image"]
            agnostic_mask = batch["agnostic_flow_mask"]

            # ---- frozen GMM warp + person features (for skip injection) ----
            with torch.no_grad():
                warped_cloth, warped_mask, _, _, _ = gmm(
                    batch["cloth"], batch["cloth_mask"], batch["cloth_sem_mask"], batch["person_rep"],
                )
                gmm_person_feats = unwrap(gmm).encode_person(batch["person_rep"])

            # ============================================================
            # Generator (refinement) update
            # ============================================================
            opt_g.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=cfg["training"]["amp"]["enabled"]):
                refined, comp_mask = refine(
                    warped_cloth, person_image, agnostic_mask, gmm_feats=gmm_person_feats,
                )
                composed = compose_output(refined, person_image, comp_mask)
                pred_fake_for_g = disc(composed, person_image)
                loss_g, parts_g = refine_loss(
                    composed, comp_mask, person_image, agnostic_mask, pred_fake=pred_fake_for_g,
                )

            scaler_g.scale(loss_g).backward()
            scaler_g.unscale_(opt_g)
            torch.nn.utils.clip_grad_norm_(refine.parameters(), max_norm=grad_clip)
            scaler_g.step(opt_g)
            sched_g.step()

            # ============================================================
            # Discriminator update
            # ============================================================
            opt_d.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=cfg["training"]["amp"]["enabled"]):
                pred_real = disc(person_image, person_image)
                pred_fake_for_d = disc(composed.detach(), person_image)
                loss_d = lsgan_d_loss(pred_real, pred_fake_for_d)

            scaler_d.scale(loss_d).backward()
            scaler_d.step(opt_d)

            scaler_g.update()
            scaler_d.update()

            global_step += 1
            n_batches += 1
            sum_g += float(loss_g.detach())
            sum_d += float(loss_d.detach())
            sum_g_gan += float(parts_g["G_gan"].detach())

            pbar.set_postfix(
                G=f"{sum_g / n_batches:.4f}",
                D=f"{sum_d / n_batches:.4f}",
                G_gan=f"{sum_g_gan / n_batches:.4f}",
                lrG=f"{opt_g.param_groups[0]['lr']:.2e}",
            )

            # ---- logging ----
            logger.scalar("refine/train/loss_G", float(loss_g.detach()), global_step)
            logger.scalar("refine/train/loss_D", float(loss_d.detach()), global_step)
            logger.lr(opt_g, global_step, tag="refine/train/lr_G")
            logger.lr(opt_d, global_step, tag="refine/train/lr_D")
            if log_individual:
                logger.scalars("refine/train/G_parts", parts_g, global_step)

            if global_step % image_log_interval == 0:
                with torch.no_grad():
                    viz = visualize_warp_result(
                        warped_cloth, refined, composed, person_image, n_samples=2,
                    )
                    cm_viz = visualize_comp_mask(comp_mask)
                logger.image("refine/viz/warped_refined_composed_person", viz, global_step)
                logger.image("refine/viz/comp_mask", cm_viz, global_step)

        avg_g = sum_g / max(1, n_batches)
        avg_d = sum_d / max(1, n_batches)
        avg_g_gan = sum_g_gan / max(1, n_batches)
        elapsed = time.time() - epoch_start
        print(f"  [Epoch {epoch:3d}] G={avg_g:.4f}  D={avg_d:.4f}  G_gan={avg_g_gan:.4f}  ({elapsed:.1f}s)")
        logger.scalar("refine/train/epoch_G", avg_g, epoch)
        logger.scalar("refine/train/epoch_D", avg_d, epoch)
        logger.scalar("refine/train/epoch_G_gan", avg_g_gan, epoch)

        # ---- D stability watchdog: D dying while G_gan still high -> halve D LR ----
        if not d_lr_halved and avg_d < 0.05 and avg_g_gan > 0.5 and epoch >= 1:
            for g in opt_d.param_groups:
                g["lr"] *= 0.5
            d_lr_halved = True
            print(f"    [watchdog] D collapsing — halved D learning rate.")

        # ---- Validation + best-only checkpoint ----
        if (epoch + 1) % val_interval == 0 or epoch == epochs - 1:
            def _val_forward(_model, b):
                b = _move(b, device)
                with torch.no_grad():
                    wc, _, _, _, _ = gmm(b["cloth"], b["cloth_mask"], b["cloth_sem_mask"], b["person_rep"])
                    gp = unwrap(gmm).encode_person(b["person_rep"])
                    r, cm = _model(wc, b["image"], b["agnostic_flow_mask"], gmm_feats=gp)
                    cmp = compose_output(r, b["image"], cm)
                return cmp, b["image"]

            metrics = tracker.validate(refine, val_loader, _val_forward)
            tracker.log_metrics(epoch, metrics)
            for k, v in metrics.items():
                logger.scalar(f"refine/val/{k}", v, epoch)
            extra = {
                "discriminator_state_dict": state_dict_for_save(disc),
                "discriminator_optimizer_state_dict": opt_d.state_dict(),
                "phase": "refinement",
            }
            improved = tracker.save_if_best(
                refine, opt_g, sched_g, scaler_g,
                epoch=epoch, metrics=metrics, config=cfg, extra=extra,
            )
            if improved:
                print(f"    [best] checkpoint updated at epoch {epoch}")
            if tracker.should_stop():
                print(f"  Early stopping at epoch {epoch}")
                tracker.save_last(
                    refine, opt_g, sched_g, scaler_g,
                    epoch=epoch, metrics=metrics, config=cfg, extra=extra,
                )
                break

        # last.pth every epoch — survives Kaggle session timeouts.
        tracker.save_last(
            refine, opt_g, sched_g, scaler_g,
            epoch=epoch, metrics=None, config=cfg,
            extra={
                "discriminator_state_dict": state_dict_for_save(disc),
                "discriminator_optimizer_state_dict": opt_d.state_dict(),
                "phase": "refinement",
            },
        )

        torch.cuda.empty_cache()

    logger.close()
    print("\n[done] Refinement training complete.")
    print(f"  best LPIPS = {tracker.best['lpips']}")
    print(f"  best SSIM  = {tracker.best['ssim']}")
    print(f"  checkpoints in {ckpt_dir}")


# ------------------------------------------------------------------ CLI

def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--gmm_checkpoint", type=str, required=True)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train_refinement(args.config, gmm_checkpoint=args.gmm_checkpoint,
                     resume_from=args.resume, smoke=args.smoke)
