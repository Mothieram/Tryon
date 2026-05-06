"""
densepose_engine.py - DensePose torso surface mapping with safe fallback.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np

# Ensure project root is importable when run as:
#   python engine/densepose_engine.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from engine.coreutils import Keypoint, PoseKeypoints, setup_logger

logger = setup_logger("densepose")


@dataclass
class TorsoMap:
    torso_mask: np.ndarray
    arm_masks: Dict[str, np.ndarray]
    neck_mask: np.ndarray
    uv_map: Optional[np.ndarray]
    method: str = "keypoint"
    confidence: float = 1.0


class DensePoseEngine:
    """DensePose wrapper with parsing/keypoint fallback."""

    def __init__(self, use_densepose: bool = True, device: str = "auto"):
        self._densepose_available = False
        self._predictor = None
        self._device = self._resolve_device(device)
        self._project_root = Path(__file__).resolve().parents[1]

        if use_densepose:
            self._try_load_densepose()

        if not self._densepose_available:
            logger.info("DensePose unavailable - using fallback body mapping")

    def _resolve_device(self, device: str) -> str:
        if device != "auto":
            return "cuda" if str(device).startswith("cuda") else str(device)
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    def _try_load_densepose(self) -> None:
        """Attempt loading detectron2 + DensePose model."""
        try:
            densepose_root = self._project_root / "DensePose"
            if densepose_root.exists() and str(densepose_root) not in sys.path:
                sys.path.insert(0, str(densepose_root))

            from detectron2.config import get_cfg
            from detectron2.engine import DefaultPredictor
            from densepose import add_densepose_config

            cfg = get_cfg()
            add_densepose_config(cfg)
            cfg_path = densepose_root / "configs" / "densepose_rcnn_R_50_FPN_s1x.yaml"
            weights_path = self._project_root / "models" / "densepose" / "model_final.pkl"

            if not cfg_path.exists():
                raise FileNotFoundError(f"DensePose config not found: {cfg_path}")
            if not weights_path.exists():
                raise FileNotFoundError(f"DensePose weights not found: {weights_path}")

            cfg.merge_from_file(str(cfg_path))
            cfg.MODEL.WEIGHTS = str(weights_path)
            cfg.MODEL.DEVICE = "cuda" if self._device == "cuda" else "cpu"
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

            self._predictor = DefaultPredictor(cfg)
            self._densepose_available = True
            logger.info("DensePose loaded successfully")
        except Exception as exc:
            self._densepose_available = False
            logger.warning(f"DensePose load failed: {exc}")

    def estimate(
        self,
        frame: np.ndarray,
        pose: PoseKeypoints,
        parsing_mask: Optional[Any] = None,
    ) -> TorsoMap:
        h, w = frame.shape[:2]

        if self._densepose_available and self._predictor is not None:
            dense = self._densepose_estimate(frame)
            if dense is not None:
                return dense

        if parsing_mask is not None:
            parsed = self._parsing_estimate(parsing_mask, h, w)
            if parsed is not None:
                return parsed

        return self._keypoint_estimate(pose, h, w)

    def _densepose_estimate(self, frame: np.ndarray) -> Optional[TorsoMap]:
        try:
            outputs = self._predictor(frame)
            instances = outputs["instances"].to("cpu")
            if len(instances) == 0 or not instances.has("pred_densepose"):
                return None

            h, w = frame.shape[:2]
            torso_mask = np.zeros((h, w), dtype=np.uint8)
            left_arm = np.zeros((h, w), dtype=np.uint8)
            right_arm = np.zeros((h, w), dtype=np.uint8)
            neck_mask = np.zeros((h, w), dtype=np.uint8)

            box = instances.pred_boxes.tensor.numpy()[0].astype(int)
            x1, y1, x2, y2 = box
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            bw, bh = max(1, x2 - x1), max(1, y2 - y1)

            dp = instances.pred_densepose[0]

            coarse = dp.coarse_segm.cpu()
            if coarse.dim() == 4:
                coarse = coarse.squeeze(0)
            coarse_map = np.argmax(coarse.numpy(), axis=0).astype(np.uint8)

            fine = dp.fine_segm.cpu()
            if fine.dim() == 4:
                fine = fine.squeeze(0)
            fine_np = fine.numpy()
            if fine_np.ndim != 3:
                return None
            fine_map = np.argmax(fine_np, axis=0).astype(np.uint8)

            coarse_map = cv2.resize(coarse_map, (bw, bh), interpolation=cv2.INTER_NEAREST)
            fine_map = cv2.resize(fine_map, (bw, bh), interpolation=cv2.INTER_NEAREST)
            labels = fine_map.copy()
            labels[coarse_map == 0] = 0

            torso_roi = np.isin(labels, [1, 2]).astype(np.uint8) * 255
            left_roi = np.isin(labels, [16, 18]).astype(np.uint8) * 255
            right_roi = np.isin(labels, [15, 17]).astype(np.uint8) * 255

            y2c, x2c = y1 + torso_roi.shape[0], x1 + torso_roi.shape[1]
            torso_mask[y1:y2c, x1:x2c] = torso_roi
            left_arm[y1:y2c, x1:x2c] = left_roi
            right_arm[y1:y2c, x1:x2c] = right_roi

            ys, xs = np.where(torso_mask > 0)
            if len(xs) > 0:
                cx = int(np.mean(xs))
                top = int(np.min(ys))
                cv2.ellipse(neck_mask, (cx, max(0, top - 10)), (18, 14), 0, 0, 360, 255, -1)

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            torso_mask = cv2.morphologyEx(torso_mask, cv2.MORPH_CLOSE, kernel)
            left_arm = cv2.morphologyEx(left_arm, cv2.MORPH_CLOSE, kernel)
            right_arm = cv2.morphologyEx(right_arm, cv2.MORPH_CLOSE, kernel)

            return TorsoMap(
                torso_mask=torso_mask,
                arm_masks={"left_arm": left_arm, "right_arm": right_arm},
                neck_mask=neck_mask,
                uv_map=None,
                method="densepose",
                confidence=1.0,
            )
        except Exception as exc:
            logger.debug(f"DensePose inference failed, falling back: {exc}")
            return None

    def _parsing_estimate(self, parsing_mask: Any, h: int, w: int) -> Optional[TorsoMap]:
        """Build torso map from either ParsedRegions object or label map."""
        torso_mask = np.zeros((h, w), dtype=np.uint8)
        left_arm = np.zeros((h, w), dtype=np.uint8)
        right_arm = np.zeros((h, w), dtype=np.uint8)
        neck_mask = np.zeros((h, w), dtype=np.uint8)

        # ParsedRegions object from parsing engine.
        if hasattr(parsing_mask, "torso"):
            torso_mask = parsing_mask.torso.copy()
            left_arm = parsing_mask.left_arm.copy()
            right_arm = parsing_mask.right_arm.copy()
            if hasattr(parsing_mask, "head_region"):
                neck_mask = parsing_mask.head_region.copy()
            elif hasattr(parsing_mask, "face"):
                neck_mask = parsing_mask.face.copy()
        else:
            # Integer label mask fallback (ATR labels).
            labels = np.asarray(parsing_mask)
            if labels.ndim != 2:
                return None
            torso_mask = np.isin(labels, [4, 7]).astype(np.uint8) * 255
            left_arm = np.isin(labels, [14]).astype(np.uint8) * 255
            right_arm = np.isin(labels, [15]).astype(np.uint8) * 255
            neck_mask = np.isin(labels, [11, 1, 2]).astype(np.uint8) * 255

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        torso_mask = cv2.morphologyEx(torso_mask, cv2.MORPH_CLOSE, kernel)
        torso_mask = cv2.morphologyEx(torso_mask, cv2.MORPH_OPEN, kernel)

        return TorsoMap(
            torso_mask=torso_mask,
            arm_masks={"left_arm": left_arm, "right_arm": right_arm},
            neck_mask=neck_mask,
            uv_map=None,
            method="parsing",
            confidence=0.9,
        )

    def _keypoint_estimate(self, pose: PoseKeypoints, h: int, w: int) -> TorsoMap:
        torso_mask = np.zeros((h, w), dtype=np.uint8)
        left_arm = np.zeros((h, w), dtype=np.uint8)
        right_arm = np.zeros((h, w), dtype=np.uint8)
        neck_mask = np.zeros((h, w), dtype=np.uint8)

        ls = pose.left_shoulder
        rs = pose.right_shoulder
        lh = pose.left_hip
        rh = pose.right_hip

        if ls and rs and ls.valid and rs.valid:
            sw = max(20.0, pose.shoulder_width)
            expand = sw * 0.12
            pts = [[ls.x - expand, ls.y], [rs.x + expand, rs.y]]
            if rh and rh.valid:
                pts.append([rh.x + expand * 0.5, rh.y])
            else:
                pts.append([rs.x, rs.y + sw * 1.3])
            if lh and lh.valid:
                pts.append([lh.x - expand * 0.5, lh.y])
            else:
                pts.append([ls.x, ls.y + sw * 1.3])
            cv2.fillConvexPoly(torso_mask, np.array(pts, dtype=np.int32), 255)

        def draw_arm(mask: np.ndarray, p1, p2, p3=None) -> None:
            if p1 and p2 and p1.valid and p2.valid:
                t = max(10, int(max(20.0, pose.shoulder_width) * 0.16))
                cv2.line(mask, p1.to_tuple(), p2.to_tuple(), 255, t)
                if p3 and p3.valid:
                    cv2.line(mask, p2.to_tuple(), p3.to_tuple(), 255, int(t * 0.85))

        draw_arm(left_arm, pose.left_shoulder, pose.left_elbow, pose.left_wrist)
        draw_arm(right_arm, pose.right_shoulder, pose.right_elbow, pose.right_wrist)

        nose = pose.nose
        if nose and nose.valid and ls and rs and ls.valid and rs.valid:
            cx = int((ls.x + rs.x) / 2)
            top = int(nose.y)
            bot = int((ls.y + rs.y) / 2)
            nw = max(10, int(max(20.0, pose.shoulder_width) * 0.18))
            nh = max(8, abs(bot - top) // 2 + 8)
            cv2.ellipse(neck_mask, (cx, (top + bot) // 2), (nw, nh), 0, 0, 360, 255, -1)

        torso_mask = cv2.GaussianBlur(torso_mask, (15, 15), 0)
        left_arm = cv2.GaussianBlur(left_arm, (11, 11), 0)
        right_arm = cv2.GaussianBlur(right_arm, (11, 11), 0)

        return TorsoMap(
            torso_mask=torso_mask,
            arm_masks={"left_arm": left_arm, "right_arm": right_arm},
            neck_mask=neck_mask,
            uv_map=None,
            method="keypoint",
            confidence=0.75,
        )

    @property
    def has_densepose(self) -> bool:
        return self._densepose_available


__all__ = ["DensePoseEngine", "TorsoMap"]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Quick densepose engine smoke test.")
    parser.add_argument(
        "--image",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "person.jpg"),
        help="Path to input image.",
    )
    parser.add_argument(
        "--no-densepose",
        action="store_true",
        help="Force fallback mode for quick testing.",
    )
    args = parser.parse_args()

    img = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"Could not read image: {args.image}")

    h, w = img.shape[:2]
    # Minimal synthetic pose for standalone testing.
    kps = [Keypoint(0.0, 0.0, 0.0) for _ in range(17)]
    kps[0] = Keypoint(w * 0.50, h * 0.20, 1.0)   # nose
    kps[5] = Keypoint(w * 0.36, h * 0.34, 1.0)   # left shoulder
    kps[6] = Keypoint(w * 0.64, h * 0.34, 1.0)   # right shoulder
    kps[7] = Keypoint(w * 0.30, h * 0.48, 1.0)   # left elbow
    kps[8] = Keypoint(w * 0.70, h * 0.48, 1.0)   # right elbow
    kps[9] = Keypoint(w * 0.27, h * 0.62, 1.0)   # left wrist
    kps[10] = Keypoint(w * 0.73, h * 0.62, 1.0)  # right wrist
    kps[11] = Keypoint(w * 0.42, h * 0.72, 1.0)  # left hip
    kps[12] = Keypoint(w * 0.58, h * 0.72, 1.0)  # right hip
    pose = PoseKeypoints(keypoints=kps, confidence=1.0)

    engine = DensePoseEngine(use_densepose=not args.no_densepose)
    torso = engine.estimate(img, pose, parsing_mask=None)

    vis = np.zeros_like(img)
    vis[torso.torso_mask > 0] = (255, 120, 0)
    vis[torso.arm_masks.get("left_arm", np.zeros((h, w), np.uint8)) > 0] = (0, 255, 255)
    vis[torso.arm_masks.get("right_arm", np.zeros((h, w), np.uint8)) > 0] = (0, 255, 0)
    vis[torso.neck_mask > 0] = (255, 0, 255)

    overlay = cv2.addWeighted(img, 0.6, vis, 0.4, 0)
    out_dir = Path(__file__).resolve().parents[1]
    color_out = out_dir / "densepose_test_colored.png"
    overlay_out = out_dir / "densepose_test_overlay.png"
    cv2.imwrite(str(color_out), vis)
    cv2.imwrite(str(overlay_out), overlay)
    print(f"Method: {torso.method} | DensePose loaded: {engine.has_densepose}")
    print(f"Saved: {color_out}")
    print(f"Saved: {overlay_out}")
