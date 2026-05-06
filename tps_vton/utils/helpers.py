"""Common helpers: seeding, device selection, AMP scaler."""

from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device(prefer: Optional[str] = None) -> torch.device:
    if prefer is not None:
        return torch.device(prefer)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_amp_scaler(enabled: bool):
    """torch.cuda.amp.GradScaler with a no-op fallback when AMP is disabled."""
    return torch.cuda.amp.GradScaler(enabled=enabled)


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def maybe_data_parallel(model: torch.nn.Module, device: torch.device) -> torch.nn.Module:
    """Move `model` to `device` and wrap with `nn.DataParallel` if more than one
    CUDA device is visible. Returns the wrapped (or plain) module.
    """
    model = model.to(device)
    if device.type == "cuda" and torch.cuda.device_count() > 1:
        device_ids = list(range(torch.cuda.device_count()))
        print(f"[info] DataParallel across {len(device_ids)} GPUs: {device_ids}")
        return torch.nn.DataParallel(model, device_ids=device_ids)
    return model


def unwrap(model: torch.nn.Module) -> torch.nn.Module:
    """Strip a DataParallel/DistributedDataParallel wrapper if present."""
    return model.module if isinstance(model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)) else model


def state_dict_for_save(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    """state_dict that's load-compatible with both wrapped and unwrapped models."""
    return unwrap(model).state_dict()


def load_state_dict_compat(model: torch.nn.Module, state: Dict[str, torch.Tensor], strict: bool = False):
    """Load `state` regardless of whether `model` is DataParallel-wrapped or not."""
    target = unwrap(model)
    # If the saved keys still have a 'module.' prefix from an older DP save, strip it.
    if state and all(k.startswith("module.") for k in list(state.keys())[:8]):
        state = {k[len("module."):]: v for k, v in state.items()}
    return target.load_state_dict(state, strict=strict)
