"""
sam2_engine.py - Body silhouette via Meta SAM2 (HuggingFace transformers).

Pixel-precise body segmentation prompted by MediaPipe pose keypoints.
The pipeline pushes 4-6 anchor points onto the chest / belly / hips and
SAM2 returns a clean torso mask that tracks the body silhouette far
better than DensePose's coarse semantic mask.

Live performance depends on the selected SAM2 checkpoint. The default
SAM2.1 large checkpoint favors mask quality over raw frame rate.

Falls back gracefully when:
  - `transformers` is too old to ship SAM2 (logs a clear install hint),
  - the model fails to download (network, HF token, etc.),
  - the pose keypoints aren't usable for prompting.
"""

from __future__ import annotations

import time
import importlib.util
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from engine.coreutils import setup_logger, PoseKeypoints

logger = setup_logger("sam2")


def _dependency_debug_hint() -> str:
    spec = importlib.util.find_spec("filelock")
    origin = getattr(spec, "origin", None) if spec is not None else None
    search_locations = (
        list(spec.submodule_search_locations or [])
        if spec is not None and spec.submodule_search_locations is not None
        else []
    )
    return (
        f"python={sys.executable}; filelock_origin={origin}; "
        f"filelock_locations={search_locations}"
    )

try:
    import torch
except Exception:
    torch = None  # type: ignore

# SAM2 transformers integration. Falls back to SAM v1 if SAM2 isn't
# available in the installed transformers — both expose a similar
# point-prompted segmentation API, so the pipeline still works.
_SAM2_BACKEND: Optional[str] = None
_SAM2_IMPORT_ERROR: Optional[str] = None
SamProcessorClass: Any = None
SamModelClass: Any = None
try:
    from transformers import Sam2Model, Sam2Processor
    SamProcessorClass = Sam2Processor
    SamModelClass = Sam2Model
    _SAM2_BACKEND = "sam2"
except Exception as exc1:
    try:
        from transformers import SamModel, SamProcessor
        SamProcessorClass = SamProcessor
        SamModelClass = SamModel
        _SAM2_BACKEND = "sam"
        _SAM2_IMPORT_ERROR = (
            f"SAM2 not present in transformers ({exc1}); falling back to SAM v1. "
            "For SAM2 install: pip install -U transformers"
        )
    except Exception as exc2:
        _SAM2_IMPORT_ERROR = (
            f"Neither SAM2 nor SAM is importable from transformers: "
            f"{exc1} | {exc2}. Install SAM2 dependencies in the same venv "
            f"used to run main.py. {_dependency_debug_hint()}"
        )


@dataclass
class SAM2Status:
    loaded: bool = False
    backend: str = ""               # "sam2", "sam", or ""
    model_id: str = ""
    device: str = "cpu"
    last_inference_ms: float = 0.0
    error: Optional[str] = None


class SAM2Engine:
    """Body-silhouette extractor backed by SAM2 (or SAM v1 fallback).

    Usage:
        engine = SAM2Engine()           # auto-loads
        if engine.available:
            mask = engine.segment_body(frame, pose)   # uint8 (H,W), 0 or 255

    The returned mask is pixel-precise and aligned to the original frame
    resolution; resize is handled internally.
    """

    def __init__(
        self,
        model_id: Optional[str] = None,
        device: str = "auto",
        max_image_dim: int = 768,
        use_half: Optional[bool] = None,
    ):
        # Default model: SAM2.1 large if SAM2 backend, sam-vit-base for SAM v1.
        if model_id is None:
            model_id = (
                "facebook/sam2.1-hiera-large" if _SAM2_BACKEND == "sam2"
                else "facebook/sam-vit-base"
            )
        self.model_id = model_id
        self.max_image_dim = int(max_image_dim)
        self._device = self._resolve_device(device)
        # FP16 on GPU for speed; SAM/SAM2 both tolerate it.
        if use_half is None:
            use_half = self._device.startswith("cuda")
        self._use_half = bool(use_half)
        self._processor: Any = None
        self._model: Any = None
        self._status = SAM2Status(
            backend=_SAM2_BACKEND or "", model_id=model_id, device=self._device,
        )
        if _SAM2_BACKEND is None:
            self._status.error = _SAM2_IMPORT_ERROR
            logger.warning("SAM2 unavailable: %s", self._status.error)
            return
        if _SAM2_IMPORT_ERROR:
            logger.info(_SAM2_IMPORT_ERROR)
        self._try_load()

    # ── Public ──────────────────────────────────────────────────────────────

    @property
    def available(self) -> bool:
        return bool(self._status.loaded and self._model is not None)

    def get_status(self) -> Dict[str, Any]:
        return {
            "loaded": self._status.loaded,
            "backend": self._status.backend,
            "model_id": self._status.model_id,
            "device": self._status.device,
            "last_inference_ms": self._status.last_inference_ms,
            "error": self._status.error,
        }

    def segment_body(
        self,
        frame: np.ndarray,
        pose: PoseKeypoints,
    ) -> Optional[np.ndarray]:
        """Run SAM2 with pose-keypoint prompts to extract a body silhouette.

        Returns a uint8 mask of shape (H, W) where 255=body, 0=background.
        Returns None when SAM2 is unavailable or no usable prompts could
        be derived from the pose.
        """
        if not self.available or torch is None:
            return None
        prompts = self._build_prompts(pose)
        if prompts is None:
            return None

        h, w = frame.shape[:2]
        # Downscale large frames so SAM2 inference stays in budget.
        scale = 1.0
        work_frame = frame
        if max(h, w) > self.max_image_dim:
            scale = self.max_image_dim / float(max(h, w))
            ws = max(64, int(round(w * scale)))
            hs = max(64, int(round(h * scale)))
            work_frame = cv2.resize(frame, (ws, hs), interpolation=cv2.INTER_AREA)

        # Convert BGR → RGB for HuggingFace processor.
        rgb = cv2.cvtColor(work_frame, cv2.COLOR_BGR2RGB)

        # Scale prompt coordinates into the work-frame space.
        scaled_pts = [
            [int(round(x * scale)), int(round(y * scale))]
            for (x, y) in prompts["points"]
        ]
        labels = list(prompts["labels"])

        try:
            t0 = time.perf_counter()
            mask_small = self._infer(rgb, scaled_pts, labels)
            self._status.last_inference_ms = (time.perf_counter() - t0) * 1000.0
        except Exception as exc:
            logger.warning("SAM2 inference failed: %s", exc)
            return None

        if mask_small is None:
            return None
        # Resize mask back to original frame resolution.
        if mask_small.shape[:2] != (h, w):
            mask = cv2.resize(
                mask_small.astype(np.uint8), (w, h),
                interpolation=cv2.INTER_NEAREST,
            )
        else:
            mask = mask_small.astype(np.uint8)
        # Convert binary mask to 0/255.
        mask = (mask > 0).astype(np.uint8) * 255
        # Light morphological cleanup so any single-pixel speckles disappear.
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return mask

    # ── Internals ───────────────────────────────────────────────────────────

    def _resolve_device(self, device: str) -> str:
        if device != "auto":
            return device
        if torch is not None and torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def _try_load(self) -> None:
        if SamProcessorClass is None or SamModelClass is None:
            return
        try:
            self._processor = SamProcessorClass.from_pretrained(self.model_id)
            self._model = SamModelClass.from_pretrained(self.model_id)
            self._model = self._model.to(self._device)
            if self._use_half and self._device.startswith("cuda"):
                try:
                    self._model = self._model.half()
                except Exception:
                    self._use_half = False
            self._model.eval()
            self._status.loaded = True
            logger.info(
                "SAM2 ready | backend=%s | model=%s | device=%s | half=%s",
                self._status.backend, self.model_id, self._device, self._use_half,
            )
        except Exception as exc:
            self._status.loaded = False
            self._status.error = str(exc)
            logger.warning("SAM2 model load failed: %s", exc)

    def _build_prompts(
        self, pose: PoseKeypoints,
    ) -> Optional[Dict[str, List[Any]]]:
        """Derive 4-6 positive prompt points on the torso from pose keypoints."""
        ls = pose.left_shoulder
        rs = pose.right_shoulder
        if not (ls and rs and ls.valid and rs.valid):
            return None
        sw = max(20.0, abs(rs.x - ls.x))
        cx = (ls.x + rs.x) * 0.5
        cy = (ls.y + rs.y) * 0.5
        lh = pose.left_hip
        rh = pose.right_hip
        # Hip line (extrapolate when not detected so we still get prompts).
        if lh and rh and lh.valid and rh.valid:
            hcx = (lh.x + rh.x) * 0.5
            hcy = (lh.y + rh.y) * 0.5
        else:
            hcx = cx
            hcy = cy + sw * 1.4
        # Positive prompts placed inside the torso so SAM2 segments the
        # whole body silhouette around them.
        points: List[Tuple[float, float]] = [
            (cx,             cy + sw * 0.18),  # upper chest
            (cx,             (cy + hcy) * 0.5),  # belly button area
            (hcx,            hcy - sw * 0.10),  # waistband
            (cx - sw * 0.18, cy + sw * 0.30),  # left side of chest
            (cx + sw * 0.18, cy + sw * 0.30),  # right side of chest
        ]
        return {
            "points": [(int(round(x)), int(round(y))) for (x, y) in points],
            "labels": [1] * len(points),
        }

    def _infer(
        self,
        rgb: np.ndarray,
        points: List[List[int]],
        labels: List[int],
    ) -> Optional[np.ndarray]:
        """Run the underlying HF SAM/SAM2 forward pass."""
        # HuggingFace SAM API expects nested lists:
        #   input_points  : (batch, point_batch, n_points, 2)
        #   input_labels  : (batch, point_batch, n_points)
        input_points = [[points]]
        input_labels = [[labels]]
        inputs = self._processor(
            rgb,
            input_points=input_points,
            input_labels=input_labels,
            return_tensors="pt",
        )
        # Move to device (fp16 conversion only for floating tensors).
        moved = {}
        for k, v in inputs.items():
            if torch.is_tensor(v):
                v_dev = v.to(self._device)
                if self._use_half and v_dev.is_floating_point():
                    v_dev = v_dev.half()
                moved[k] = v_dev
            else:
                moved[k] = v
        with torch.inference_mode():
            outputs = self._model(**moved, multimask_output=False)
        # Post-process to original image resolution.
        try:
            post_process_masks = getattr(self._processor, "post_process_masks", None)
            if post_process_masks is None:
                post_process_masks = self._processor.image_processor.post_process_masks
            original_sizes = (
                inputs["original_sizes"].cpu()
                if "original_sizes" in inputs else None
            )
            reshaped_input_sizes = (
                inputs["reshaped_input_sizes"].cpu()
                if "reshaped_input_sizes" in inputs else None
            )
            try:
                masks = post_process_masks(outputs.pred_masks.cpu(), original_sizes)
            except TypeError:
                masks = post_process_masks(
                    outputs.pred_masks.cpu(),
                    original_sizes,
                    reshaped_input_sizes,
                )
        except Exception:
            # Some processors expose the helper differently; fall back to
            # a simple resize of the raw logit prediction.
            pred = outputs.pred_masks.detach().cpu().float().numpy()
            mask = pred[0, 0] if pred.ndim == 4 else pred
            return (mask > 0.0).astype(np.uint8)

        if not masks:
            return None
        m = masks[0]
        # masks[0] shape: (1, num_masks, H, W) torch tensor or numpy.
        if hasattr(m, "numpy"):
            m = m.numpy()
        m = np.asarray(m)
        # Squeeze to (H, W). Take the first mask if multimask was returned.
        while m.ndim > 2:
            m = m[0]
        return (m > 0).astype(np.uint8)


__all__ = ["SAM2Engine", "SAM2Status"]
