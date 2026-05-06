"""Step 1 smoke test: load one batch, print shapes, verify channel counts."""

from __future__ import annotations

import sys
from pathlib import Path

import yaml
from torch.utils.data import DataLoader

# allow running as `python scripts/smoke_test_data.py` from the tps_vton dir
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.dataset import VitonHDDataset, deterministic_train_val_split


def main() -> None:
    cfg_path = Path(__file__).resolve().parents[1] / "configs" / "config.yaml"
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    # If run from repo root, dataset path is ./dataset; resolve relative to repo root.
    repo_root = Path(__file__).resolve().parents[2]
    cfg["data"]["root"] = str(repo_root / "dataset")

    train_pairs_file = Path(cfg["data"]["root"]) / cfg["data"]["train_pairs"]
    val_pairs_file = Path(cfg["data"]["root"]) / cfg["data"]["val_pairs"]
    if not train_pairs_file.exists():
        # fall back to the repo-root copy (the user keeps train_pairs.txt at repo root)
        alt = repo_root / cfg["data"]["train_pairs"]
        if alt.exists():
            train_pairs_file = alt
            cfg["data"]["root"] = str(repo_root / "dataset")
            val_pairs_file = repo_root / cfg["data"]["val_pairs"]

    print(f"[info] root         = {cfg['data']['root']}")
    print(f"[info] train_pairs  = {train_pairs_file}")

    train_pairs, val_pairs = deterministic_train_val_split(
        train_pairs_file, val_pairs_file,
        val_split=cfg["data"]["val_split"],
        seed=cfg["training"]["seed"],
    )
    print(f"[info] train pairs  = {len(train_pairs)}")
    print(f"[info] val pairs    = {len(val_pairs)}")

    ds = VitonHDDataset(cfg, split="train", augment=True, pairs_override=train_pairs)
    print(f"[info] dataset size = {len(ds)}")

    loader = DataLoader(ds, batch_size=2, shuffle=False, num_workers=0)
    batch = next(iter(loader))

    print("\n=== Batch tensor shapes ===")
    for k, v in batch.items():
        if hasattr(v, "shape"):
            print(f"  {k:<22} {tuple(v.shape)}  dtype={v.dtype}")
        else:
            print(f"  {k:<22} {type(v).__name__} (len={len(v) if hasattr(v, '__len__') else '?'})")

    # ---- Assertions ----
    H, W = cfg["data"]["resolution"]
    expected_person_ch = cfg["model"]["person_rep_channels"]
    expected_cloth_ch = cfg["model"]["cloth_input_channels"]

    assert batch["cloth"].shape == (2, 3, H, W), batch["cloth"].shape
    assert batch["cloth_mask"].shape == (2, 1, H, W), batch["cloth_mask"].shape
    assert batch["cloth_sem_mask"].shape == (2, 3, H, W), batch["cloth_sem_mask"].shape

    person_rep = batch["person_rep"]
    assert person_rep.shape == (2, expected_person_ch, H, W), \
        f"person_rep {person_rep.shape}, expected {(2, expected_person_ch, H, W)}"

    cloth_input_ch = batch["cloth"].shape[1] + batch["cloth_mask"].shape[1] + batch["cloth_sem_mask"].shape[1]
    assert cloth_input_ch == expected_cloth_ch, \
        f"cloth input channels {cloth_input_ch}, expected {expected_cloth_ch}"

    print("\n[OK] Step 1 smoke test passed.")
    print(f"     person_rep channels = {person_rep.shape[1]} (target {expected_person_ch})")
    print(f"     cloth input channels = {cloth_input_ch} (target {expected_cloth_ch})")


if __name__ == "__main__":
    main()
