"""Stage 3: short joint fine-tuning of GMM + Refinement at very low LR."""

from __future__ import annotations

import argparse
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
from models.losses import GMMLossComputer, RefinementLossComputer, lsgan_d_loss
from models.refinement import RefinementUNet, compose_output
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
from utils.visualization import visualize_warp_result


def _move(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
    return out


def train_joint(
    cfg_path: str,
    gmm_checkpoint: str,
    refine_checkpoint: str,
    resume_from: Optional[str] = None,
    smoke: bool = False,
) -> None:
    if not (gmm_checkpoint and refine_checkpoint):
        raise SystemExit("--gmm_checkpoint and --refine_checkpoint are both required for joint training.")

    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["training"]["seed"])
    device = get_device(cfg["training"].get("device"))
    print(f"[info] device = {device}")

    # ---- dataset paths ----
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
        cfg["training"]["joint_epochs"] = 2
        cfg["training"]["batch_size"] = 4
        cfg["validation"]["interval"] = 1
        cfg["data"]["num_workers"] = 0
        cfg["logging"]["image_log_interval"] = 1
        print(f"[smoke] train pairs = {len(train_pairs)}, val pairs = {len(val_pairs)}")

    bs = cfg["training"]["batch_size"]
    train_set = VitonHDDataset(cfg, split="train", augment=True, pairs_override=train_pairs)
    val_set = VitonHDDataset(cfg, split="val", augment=False, pairs_override=val_pairs)
    train_loader = DataLoader(
        train_set, batch_size=bs, shuffle=True, drop_last=True,
        num_workers=cfg["data"]["num_workers"], pin_memory=True,
        persistent_workers=cfg["data"]["num_workers"] > 0,
    )
    val_loader = DataLoader(
        val_set, batch_size=max(1, bs // 2), shuffle=False,
        num_workers=cfg["data"]["num_workers"], pin_memory=True,
        persistent_workers=cfg["data"]["num_workers"] > 0,
    )

    H, W = cfg["data"]["resolution"]

    # ---- models (now BOTH trainable) ----
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
    gmm.train()
    for p in gmm.parameters():
        p.requires_grad = True

    refine = RefinementUNet(
        in_ch=7,
        features=cfg["model"]["encoder_features"],
        gmm_features=cfg["model"]["encoder_features"],
    )
    refine = maybe_data_parallel(refine, device)
    load_checkpoint(refine_checkpoint, refine, map_location=device, strict=False)
    refine.train()

    # Discriminator: warm-start from refinement checkpoint if available, else fresh.
    disc = PatchDiscriminator(
        in_ch=cfg["discriminator"]["in_channels"], features=cfg["discriminator"]["features"],
    )
    disc = maybe_data_parallel(disc, device)
    refine_ckpt = torch.load(refine_checkpoint, map_location="cpu")
    if "discriminator_state_dict" in refine_ckpt:
        load_state_dict_compat(disc, refine_ckpt["discriminator_state_dict"], strict=False)
        print("[info] discriminator warm-started from refinement checkpoint")

    print(f"[info] GMM params    = {count_parameters(gmm):,}")
    print(f"[info] Refine params = {count_parameters(refine):,}")
    print(f"[info] Disc   params = {count_parameters(disc):,}")

    # ---- single low-LR optimizer for GMM + refinement ----
    joint_lr = 1e-5
    opt_g = torch.optim.Adam(
        list(gmm.parameters()) + list(refine.parameters()),
        lr=joint_lr, betas=tuple(cfg["training"]["optimizer"]["betas"]),
    )
    opt_d = torch.optim.Adam(
        disc.parameters(),
        lr=cfg["training"]["discriminator_optimizer"]["lr"] * 0.5,
        betas=tuple(cfg["training"]["discriminator_optimizer"]["betas"]),
    )

    epochs = cfg["training"]["joint_epochs"]
    sched_g = build_lr_scheduler(
        opt_g, warmup_epochs=0, total_epochs=epochs,
        steps_per_epoch=max(1, len(train_loader)),
        warmup_start_factor=1.0, eta_min=1e-7,
    )

    scaler_g = make_amp_scaler(enabled=cfg["training"]["amp"]["enabled"])
    scaler_d = make_amp_scaler(enabled=cfg["training"]["amp"]["enabled"])

    gmm_loss = GMMLossComputer(
        cfg, coarse_grid=cfg["model"]["coarse_grid"], fine_grid=cfg["model"]["fine_grid"],
    ).to(device)
    refine_loss = RefinementLossComputer(cfg).to(device)
    reg_schedule = RegWeightSchedule(cfg["training"]["reg_warmup"])

    # ---- logger + tracker (aggressive: every epoch) ----
    run_name = "joint"
    log_dir = (repo_root / cfg["logging"]["log_dir"] / run_name).resolve()
    ckpt_dir = (repo_root / cfg["logging"]["ckpt_dir"] / run_name).resolve()
    print(f"[info] log_dir  = {log_dir}")
    print(f"[info] ckpt_dir = {ckpt_dir}")
    logger = Logger(log_dir)
    tracker = ValidationTracker(
        save_dir=ckpt_dir,
        primary_metric=cfg["validation"]["primary_metric"],
        primary_metric_mode=cfg["validation"]["primary_metric_mode"],
        early_stopping_patience=max(epochs, 3),
        device=device,
    )

    grad_clip = cfg["training"]["gradient_clip"]["max_norm"]
    image_log_interval = cfg["logging"]["image_log_interval"]
    log_individual = cfg["logging"]["log_individual_losses"]

    global_step = 0
    start_epoch = 0

    # ---- Full resume: explicit --resume wins; else auto-pick last.pth ----
    if resume_from is None:
        auto = ckpt_dir / "last.pth"
        if auto.exists():
            resume_from = str(auto)
            print(f"[info] auto-resuming from {auto}")
    if resume_from is not None:
        rk = torch.load(resume_from, map_location="cpu")
        if "gmm_state_dict" in rk:
            load_state_dict_compat(gmm, rk["gmm_state_dict"], strict=False)
        if "refine_state_dict" in rk:
            load_state_dict_compat(refine, rk["refine_state_dict"], strict=False)
        if "discriminator_state_dict" in rk:
            load_state_dict_compat(disc, rk["discriminator_state_dict"], strict=False)
        if rk.get("optimizer_state_dict") is not None:
            opt_g.load_state_dict(rk["optimizer_state_dict"])
        if rk.get("discriminator_optimizer_state_dict") is not None:
            opt_d.load_state_dict(rk["discriminator_optimizer_state_dict"])
        if rk.get("scheduler_state_dict") is not None:
            sched_g.load_state_dict(rk["scheduler_state_dict"])
        if rk.get("scaler_state_dict") is not None:
            scaler_g.load_state_dict(rk["scaler_state_dict"])
        if rk.get("best") is not None:
            tracker.best.update(rk["best"])
        start_epoch = int(rk.get("epoch", -1)) + 1
        print(f"[info] resumed joint training from {resume_from} -> starting at epoch {start_epoch}")

    for epoch in range(start_epoch, epochs):
        reg_weight = reg_schedule.value(epoch + cfg["training"]["reg_warmup"]["warmup_epochs"])  # already at final
        epoch_start = time.time()
        sum_g = sum_d = 0.0
        n_batches = 0

        gmm.train()
        refine.train()
        disc.train()
        pbar = tqdm(
            train_loader,
            desc=f"[joint ep{epoch:03d}/{epochs - 1}]",
            leave=False, dynamic_ncols=True,
        )
        for batch in pbar:
            batch = _move(batch, device)
            person_image = batch["image"]
            agnostic_mask = batch["agnostic_flow_mask"]

            # ===== Generator path (GMM + Refine, both trained) =====
            opt_g.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=cfg["training"]["amp"]["enabled"]):
                warped_cloth, warped_mask, coarse_theta, fine_theta, coarse_warped = gmm(
                    batch["cloth"], batch["cloth_mask"], batch["cloth_sem_mask"], batch["person_rep"],
                )
                gmm_person_feats = unwrap(gmm).encode_person(batch["person_rep"])
                refined, comp_mask = refine(
                    warped_cloth, person_image, agnostic_mask, gmm_feats=gmm_person_feats,
                )
                composed = compose_output(refined, person_image, comp_mask)

                pred_fake_for_g = disc(composed, person_image)

                gmm_total, gmm_parts = gmm_loss(
                    warped_cloth, warped_mask, coarse_theta, fine_theta, coarse_warped,
                    target_cloth=batch["target_cloth"], target_mask=batch["target_mask"],
                    reg_weight=reg_weight,
                )
                refine_total, refine_parts = refine_loss(
                    composed, comp_mask, person_image, agnostic_mask, pred_fake=pred_fake_for_g,
                )
                loss_g = gmm_total + refine_total

            scaler_g.scale(loss_g).backward()
            scaler_g.unscale_(opt_g)
            torch.nn.utils.clip_grad_norm_(
                list(gmm.parameters()) + list(refine.parameters()), max_norm=grad_clip,
            )
            scaler_g.step(opt_g)
            sched_g.step()

            # ===== Discriminator =====
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

            pbar.set_postfix(
                G=f"{sum_g / n_batches:.4f}",
                D=f"{sum_d / n_batches:.4f}",
                lrG=f"{opt_g.param_groups[0]['lr']:.2e}",
            )

            logger.scalar("joint/train/loss_G", float(loss_g.detach()), global_step)
            logger.scalar("joint/train/loss_D", float(loss_d.detach()), global_step)
            logger.lr(opt_g, global_step, tag="joint/train/lr_G")
            logger.lr(opt_d, global_step, tag="joint/train/lr_D")
            if log_individual:
                logger.scalars("joint/train/gmm_parts", gmm_parts, global_step)
                logger.scalars("joint/train/refine_parts", refine_parts, global_step)

            if global_step % image_log_interval == 0:
                with torch.no_grad():
                    viz = visualize_warp_result(
                        batch["cloth"], warped_cloth, composed, person_image, n_samples=2,
                    )
                logger.image("joint/viz/cloth_warped_composed_person", viz, global_step)

        avg_g = sum_g / max(1, n_batches)
        avg_d = sum_d / max(1, n_batches)
        elapsed = time.time() - epoch_start
        print(f"  [Epoch {epoch:3d}] G={avg_g:.4f}  D={avg_d:.4f}  ({elapsed:.1f}s)")

        # ---- Validation (every epoch) ----
        def _val_forward(_unused, b):
            b = _move(b, device)
            with torch.no_grad():
                wc, _, _, _, _ = gmm(b["cloth"], b["cloth_mask"], b["cloth_sem_mask"], b["person_rep"])
                gp = unwrap(gmm).encode_person(b["person_rep"])
                r, cm = refine(wc, b["image"], b["agnostic_flow_mask"], gmm_feats=gp)
                cmp = compose_output(r, b["image"], cm)
            return cmp, b["image"]

        # ValidationTracker uses model.eval()/train(), so wrap a holder module
        class _Holder(torch.nn.Module):
            def __init__(self, *modules):
                super().__init__()
                self.children_list = torch.nn.ModuleList(modules)
        holder = _Holder(gmm, refine)
        metrics = tracker.validate(holder, val_loader, _val_forward)
        tracker.log_metrics(epoch, metrics)
        for k, v in metrics.items():
            logger.scalar(f"joint/val/{k}", v, epoch)

        # Save with all three state dicts so a joint checkpoint is fully restartable.
        joint_extra = {
            "gmm_state_dict": state_dict_for_save(gmm),
            "refine_state_dict": state_dict_for_save(refine),
            "discriminator_state_dict": state_dict_for_save(disc),
            "discriminator_optimizer_state_dict": opt_d.state_dict(),
            "phase": "joint",
        }
        improved = tracker.save_if_best(
            gmm, opt_g, sched_g, scaler_g,
            epoch=epoch, metrics=metrics, config=cfg, extra=joint_extra,
        )
        if improved:
            print(f"    [best] checkpoint updated at epoch {epoch}")

        # last.pth every epoch — survives Kaggle session timeouts.
        tracker.save_last(
            gmm, opt_g, sched_g, scaler_g,
            epoch=epoch, metrics=metrics, config=cfg, extra=joint_extra,
        )

        torch.cuda.empty_cache()

    logger.close()
    print("\n[done] Joint training complete.")
    print(f"  best LPIPS = {tracker.best['lpips']}")
    print(f"  best SSIM  = {tracker.best['ssim']}")
    print(f"  checkpoints in {ckpt_dir}")


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--gmm_checkpoint", type=str, required=True)
    p.add_argument("--refine_checkpoint", type=str, required=True)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train_joint(
        args.config,
        gmm_checkpoint=args.gmm_checkpoint,
        refine_checkpoint=args.refine_checkpoint,
        resume_from=args.resume,
        smoke=args.smoke,
    )
