"""Validation tracker + checkpoint manager for the GMM and refinement stages."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as sk_ssim
from tqdm.auto import tqdm

try:
    import lpips as _lpips_pkg                # noqa: F401
except ImportError:
    _lpips_pkg = None

from utils.helpers import load_state_dict_compat, state_dict_for_save


class ValidationTracker:
    """Run validation, compute SSIM/LPIPS/L1, and save best checkpoints.

    Use `forward_fn(batch)` to abstract over GMM / Refinement forward passes.
    It must return `(pred_image, target_image)` tensors in [0, 1].
    """

    def __init__(
        self,
        save_dir: str | Path,
        primary_metric: str = "lpips",
        primary_metric_mode: str = "min",
        early_stopping_patience: int = 10,
        device: torch.device | str = "cuda",
    ):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.primary_metric = primary_metric
        self.primary_metric_mode = primary_metric_mode
        self.patience = early_stopping_patience
        self.wait = 0

        self.best: Dict[str, float] = {
            "lpips": float("inf"),
            "ssim": -float("inf"),
            "l1": float("inf"),
        }
        self.history: list[Dict[str, float]] = []

        self.device = torch.device(device)
        self.lpips_fn = None
        if _lpips_pkg is not None:
            try:
                self.lpips_fn = _lpips_pkg.LPIPS(net="alex").to(self.device).eval()
                for p in self.lpips_fn.parameters():
                    p.requires_grad = False
            except Exception as exc:
                print(f"[ValidationTracker] LPIPS init failed: {exc}; LPIPS will be reported as NaN.")
                self.lpips_fn = None

    # ------------------------------------------------------------------
    @staticmethod
    def _to_lpips_input(x: torch.Tensor) -> torch.Tensor:
        # LPIPS expects (B, 3, H, W) in [-1, 1]
        return x.clamp(0, 1) * 2 - 1

    @staticmethod
    def _ssim_batch(pred: torch.Tensor, target: torch.Tensor) -> float:
        p = pred.detach().cpu().numpy().transpose(0, 2, 3, 1).astype(np.float32)
        t = target.detach().cpu().numpy().transpose(0, 2, 3, 1).astype(np.float32)
        scores = []
        for i in range(p.shape[0]):
            scores.append(sk_ssim(p[i], t[i], channel_axis=-1, data_range=1.0))
        return float(np.mean(scores))

    # ------------------------------------------------------------------
    @torch.no_grad()
    def validate(
        self,
        model: torch.nn.Module,
        val_loader,
        forward_fn: Callable[[torch.nn.Module, Dict[str, torch.Tensor]], Tuple[torch.Tensor, torch.Tensor]],
    ) -> Dict[str, float]:
        model.eval()
        l_sum = 0.0
        s_sum = 0.0
        l1_sum = 0.0
        count = 0

        for batch in tqdm(val_loader, desc="[val]", leave=False, dynamic_ncols=True):
            pred, target = forward_fn(model, batch)
            pred = pred.clamp(0, 1)
            target = target.clamp(0, 1)
            B = pred.size(0)

            if self.lpips_fn is not None:
                lp = self.lpips_fn(self._to_lpips_input(pred), self._to_lpips_input(target))
                l_sum += lp.mean().item() * B
            else:
                l_sum = float("nan")

            s_sum += self._ssim_batch(pred, target) * B
            l1_sum += F.l1_loss(pred, target, reduction="mean").item() * B
            count += B

        metrics = {
            "lpips": (l_sum / count) if (count and not np.isnan(l_sum)) else float("nan"),
            "ssim": (s_sum / count) if count else 0.0,
            "l1": (l1_sum / count) if count else 0.0,
        }
        self.history.append(metrics)
        model.train()
        return metrics

    # ------------------------------------------------------------------
    def _build_state(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        scaler: Optional[torch.cuda.amp.GradScaler],
        epoch: int,
        metrics: Optional[Dict[str, float]],
        config: Dict[str, Any],
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        state = {
            "epoch": epoch,
            "model_state_dict": state_dict_for_save(model),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
            "metrics": metrics,
            "config": config,
            "best": dict(self.best),
        }
        if extra:
            state.update(extra)
        return state

    def save_if_best(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        scaler: Optional[torch.cuda.amp.GradScaler],
        epoch: int,
        metrics: Dict[str, float],
        config: Dict[str, Any],
        extra: Optional[Dict[str, Any]] = None,
    ) -> bool:
        # Track best LPIPS / SSIM internally for early-stopping memory, but write
        # only one file: best.pth (the checkpoint at the best primary-metric epoch).
        lp = metrics.get("lpips", float("nan"))
        ss = metrics.get("ssim", 0.0)
        if not np.isnan(lp) and lp < self.best["lpips"]:
            self.best["lpips"] = lp
        if ss > self.best["ssim"]:
            self.best["ssim"] = ss

        improved = self._primary_improved(metrics)
        if improved:
            ckpt_state = self._build_state(model, optimizer, scheduler, scaler, epoch, metrics, config, extra)
            torch.save(ckpt_state, self.save_dir / "best.pth")
            self.wait = 0
        else:
            self.wait += 1
        return improved

    def save_last(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        scaler: Optional[torch.cuda.amp.GradScaler],
        epoch: int,
        metrics: Optional[Dict[str, float]],
        config: Dict[str, Any],
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Write last.pth at the end of every epoch so timeouts never lose more
        than one epoch of work."""
        ckpt_state = self._build_state(model, optimizer, scheduler, scaler, epoch, metrics, config, extra)
        torch.save(ckpt_state, self.save_dir / "last.pth")

    def _primary_improved(self, metrics: Dict[str, float]) -> bool:
        v = metrics.get(self.primary_metric, float("nan"))
        if np.isnan(v):
            return False
        if self.primary_metric_mode == "min":
            return v <= self.best[self.primary_metric] + 1e-12
        return v >= self.best[self.primary_metric] - 1e-12

    def should_stop(self) -> bool:
        return self.wait >= self.patience

    def log_metrics(self, epoch: int, metrics: Dict[str, float]) -> None:
        msg = f"[Epoch {epoch}] " + " | ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        print(msg)


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
) -> Dict[str, Any]:
    """Restore model + (optionally) optimizer/scheduler/scaler from a checkpoint."""
    ckpt = torch.load(path, map_location=map_location)
    load_state_dict_compat(model, ckpt["model_state_dict"], strict=strict)
    if optimizer is not None and ckpt.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    if scaler is not None and ckpt.get("scaler_state_dict") is not None:
        scaler.load_state_dict(ckpt["scaler_state_dict"])
    return ckpt
