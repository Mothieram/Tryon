# TPS-VITON — Running & Resuming Guide

End-to-end reference for running each training stage on Kaggle (or any GPU machine) and resuming after a session timeout.

---

## TL;DR

| What you want | What to do |
|---|---|
| Start GMM from scratch | Set `STAGE = "gmm"`, run all cells. |
| Start refinement | Set `STAGE = "refinement"`, set `GMM_CHECKPOINT`, run all cells. |
| Start joint | Set `STAGE = "joint"`, set both `GMM_CHECKPOINT` and `REFINE_CHECKPOINT`, run all cells. |
| Resume any stage after a timeout | **Just re-run.** Auto-resume finds `last.pth` in `/kaggle/working/` or `/kaggle/input/`. |

---

## 1. The three stages

The pipeline trains in three phases. Each consumes the previous stage's `best.pth`.

```
Stage 1: GMM           →  best.pth  →  used as frozen backbone in Stage 2
Stage 2: Refinement    →  best.pth  →  used together with GMM in Stage 3
Stage 3: Joint         →  best.pth  →  the final shippable checkpoint
```

| Stage | Default epochs | What it learns | Checkpoint dir |
|---|---|---|---|
| `gmm` | 60 | Cloth-to-person geometric warping (TPS) | `checkpoints/gmm/` |
| `refinement` | 40 | Texture, occlusions, composition | `checkpoints/refine/` |
| `joint` | 15 | Joint fine-tuning of both networks | `checkpoints/joint/` |

---

## 2. Running each stage

All commands run from `tps_vton/`. The Kaggle notebook builds these for you in cell 7 — you only edit the *Paths* cell.

### Stage 1 — GMM

```bash
python train.py --config configs/config_kaggle.yaml --stage gmm
```

**Notebook *Paths* cell:**
```python
STAGE             = "gmm"
GMM_CHECKPOINT    = None
REFINE_CHECKPOINT = None
RESUME_FROM       = None
```

### Stage 2 — Refinement

Needs the GMM checkpoint from stage 1.

```bash
python train.py --config configs/config_kaggle.yaml --stage refinement \
                --gmm_checkpoint /kaggle/working/checkpoints/gmm/best.pth
```

**Notebook *Paths* cell:**
```python
STAGE             = "refinement"
GMM_CHECKPOINT    = "/kaggle/working/checkpoints/gmm/best.pth"
REFINE_CHECKPOINT = None
RESUME_FROM       = None
```

If your GMM checkpoint is from a previous Kaggle session attached as input:
```python
GMM_CHECKPOINT = "/kaggle/input/<your-notebook-slug>/checkpoints/gmm/best.pth"
```

### Stage 3 — Joint

Needs both previous checkpoints.

```bash
python train.py --config configs/config_kaggle.yaml --stage joint \
                --gmm_checkpoint /kaggle/working/checkpoints/gmm/best.pth \
                --refine_checkpoint /kaggle/working/checkpoints/refine/best.pth
```

**Notebook *Paths* cell:**
```python
STAGE             = "joint"
GMM_CHECKPOINT    = "/kaggle/working/checkpoints/gmm/best.pth"
REFINE_CHECKPOINT = "/kaggle/working/checkpoints/refine/best.pth"
RESUME_FROM       = None
```

### Smoke test (sanity check, ~2 min)

```bash
python train.py --config configs/config_kaggle.yaml --stage gmm --smoke
```

Or set `SMOKE = True` in the notebook *Paths* cell.

---

## 3. Resuming after a Kaggle timeout

Kaggle GPU sessions cap at ~9–12 hours. GMM at 60 epochs × 15 min/epoch ≈ 15 hours, so you **will** time out. The training script auto-resumes — you don't need to do anything tricky.

### What gets saved

After every epoch the script writes:

```
/kaggle/working/checkpoints/<stage>/
  best.pth   ← updated only when validation LPIPS improves (best model so far)
  last.pth   ← updated every epoch (always the latest state for resume)
```

Both are full state: model weights + optimizer + scheduler + AMP scaler + epoch + best-metric memory.

### Auto-resume search order

When training starts, the script looks for a checkpoint in this order:

1. `--resume <path>` (if provided)
2. `/kaggle/working/checkpoints/<stage>/last.pth` — same session
3. `/kaggle/input/**/<stage>/last.pth` — saved-version output or uploaded dataset
4. `/kaggle/input/**/last.pth` — any `last.pth` anywhere under inputs (fallback)

The first match wins. If nothing's found, it starts fresh from epoch 0.

### Resume — same session

You stopped the cell, kernel still alive. Just re-run the launch cell. Auto-resume picks up `last.pth` from `/kaggle/working/`. Done.

### Resume — fresh session, using "Save Version"

1. **Before** the previous session ended, you clicked **Save Version → Save & Run All**. The whole `/kaggle/working/` is now the notebook's output.
2. New session: right sidebar **+ Add Input → Notebook Output → pick this notebook → latest version**.
3. Run all cells. Auto-resume finds `/kaggle/input/<your-slug>/checkpoints/<stage>/last.pth`.

### Resume — fresh session, using a downloaded checkpoint

1. Download `last.pth` (single file, ~250 MB).
2. On Kaggle: **Datasets → New Dataset → drag in `last.pth`** → name it (e.g. `tps-vton-ckpt`).
3. New session: **+ Add Input → Datasets → pick `tps-vton-ckpt`**.
4. Run all cells. Auto-resume finds `/kaggle/input/tps-vton-ckpt/last.pth`.

### Override the auto-discovery

Set an explicit path in the *Paths* cell to skip the search:
```python
RESUME_FROM = "/kaggle/input/tps-vton-ckpt/last.pth"
```
or on the command line:
```bash
python train.py --config configs/config_kaggle.yaml --stage gmm \
                --resume /kaggle/input/tps-vton-ckpt/last.pth
```

### Verify resume worked

Watch the launch cell's first lines of output:

```
[info] auto-resuming from /kaggle/input/.../checkpoints/gmm/last.pth
[info] resuming from /kaggle/input/.../checkpoints/gmm/last.pth (epoch=4)
  resumed full state -> starting at epoch 5, best_lpips=0.3217, best_ssim=0.4561
```

If you see `epoch 0` instead, run this in a debug cell:
```python
!find /kaggle/input -name "last.pth" 2>/dev/null
!ls -lah /kaggle/working/checkpoints/*/ 2>/dev/null
```

---

## 4. Checkpoint format — one file, ZIP inside

`last.pth` and `best.pth` are single self-contained files. PyTorch's `torch.save` uses ZIP format internally:

```
last.pth (ZIP)
├── data.pkl                ← Python metadata
├── data/0.bin              ← model weights tensor 0
├── data/1.bin              ← model weights tensor 1
├── ...                     ← Adam optimizer state, scheduler, scaler, etc.
```

You can verify with `unzip -l last.pth`. **Don't wrap it in another `.zip`** — it's already an archive.

Each `.pth` contains:
| Key | Purpose |
|---|---|
| `model_state_dict` | network weights |
| `optimizer_state_dict` | Adam momentum |
| `scheduler_state_dict` | warmup + cosine LR position |
| `scaler_state_dict` | AMP grad scaler |
| `epoch` | the epoch this checkpoint was saved at |
| `best` | `{lpips, ssim}` so early-stopping memory survives resume |
| `metrics` | validation metrics at this epoch (if any) |
| `config` | full training config |

For joint training, additional keys: `gmm_state_dict`, `refine_state_dict`, `discriminator_state_dict`, `discriminator_optimizer_state_dict`.

---

## 5. Stage-to-stage handoff cheat sheet

After each stage finishes, before starting the next, **copy the `best.pth` to a stable name** so it's preserved when the next stage's directory gets written into:

```python
# After GMM finishes
!cp /kaggle/working/checkpoints/gmm/best.pth /kaggle/working/gmm_best.pth

# Then for stage 2:
GMM_CHECKPOINT = "/kaggle/working/gmm_best.pth"
```

Or just point at the original location — `checkpoints/gmm/best.pth` only gets overwritten if you re-run stage 1 in the same session.

---

## 6. Validation schedule & what scores mean

| Setting | Value | Source |
|---|---|---|
| `validation.interval` | 2 | runs every other epoch |
| `validation.primary_metric` | `lpips` | best.pth saved on LPIPS improvement |
| `validation.early_stopping_patience` | 10 | stop after 10 validations with no LPIPS improvement |

Validation runs on epochs **1, 3, 5, …, 59** (odd) plus the final epoch.

Healthy targets:

| Metric | Random init | Well-trained |
|---|---|---|
| LPIPS (↓) | ~0.80 | 0.10–0.18 |
| SSIM (↑)  | ~0.05 | 0.85+ |
| L1 (↓)    | ~0.45 | 0.03–0.05 |

---

## 7. Common issues

**"prepare_data finished with non-zero exit"**
Safe to ignore on Kaggle — `/kaggle/input/` is read-only so the pairs cache can't be written. The split is deterministic via the seed; runs are still reproducible.

**"NameError: name 'Path' is not defined" in the *Paths* cell**
You ran the *Paths* cell before cell 1 (which imports `Path`). Run cells in order, or add `from pathlib import Path; import os` at the top of the *Paths* cell.

**Loss is NaN or LPIPS getting worse**
LR too high or AMP instability. Try `AMP_ENABLED = False`, or lower the optimizer LR in `config.yaml`.

**No `best.pth` after several epochs**
Validation runs every 2 epochs (epoch 1, 3, 5…). `best.pth` is only written when LPIPS improves. Watch for `last.pth` instead — that updates every epoch.

**Two `gmm_<timestamp>/` folders from old runs**
You're on outdated code. After pulling the latest, the dir is just `checkpoints/gmm/`. Delete the old timestamped dirs:
```python
import shutil, glob
for d in glob.glob("/kaggle/working/checkpoints/gmm_*"):
    shutil.rmtree(d)
```
