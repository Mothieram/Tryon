"""Learning-rate and regularization-weight schedules."""

from __future__ import annotations

from typing import Dict

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_epochs: int,
    total_epochs: int,
    steps_per_epoch: int,
    warmup_start_factor: float = 0.01,
    eta_min: float = 1e-6,
) -> torch.optim.lr_scheduler._LRScheduler:
    """Linear warmup for `warmup_epochs`, then cosine annealing to `eta_min`.

    The scheduler steps **per iteration** (not per epoch) so the schedule is
    smooth even with small training sets.
    """
    warmup_iters = max(1, warmup_epochs * steps_per_epoch)
    cosine_iters = max(1, (total_epochs - warmup_epochs) * steps_per_epoch)

    if warmup_epochs <= 0:
        return CosineAnnealingLR(optimizer, T_max=cosine_iters, eta_min=eta_min)

    warmup = LinearLR(
        optimizer,
        start_factor=warmup_start_factor,
        end_factor=1.0,
        total_iters=warmup_iters,
    )
    cosine = CosineAnnealingLR(optimizer, T_max=cosine_iters, eta_min=eta_min)
    return SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_iters],
    )


class RegWeightSchedule:
    """Linear warmdown of TPS regularization weight.

    reg_weight starts at `start_weight` and decays linearly to `final_weight`
    over `warmup_epochs` epochs, then stays constant. Call `.value(epoch)`
    each epoch to fetch the current weight.
    """

    def __init__(self, cfg: Dict):
        self.start_weight = float(cfg["start_weight"])
        self.final_weight = float(cfg["final_weight"])
        self.warmup_epochs = int(cfg["warmup_epochs"])

    def value(self, epoch: int) -> float:
        if epoch >= self.warmup_epochs:
            return self.final_weight
        frac = epoch / max(1, self.warmup_epochs)
        return self.start_weight - (self.start_weight - self.final_weight) * frac
