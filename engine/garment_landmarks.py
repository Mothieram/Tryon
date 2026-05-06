"""
garment_landmarks.py - Auto-Detect Garment Control Points
Analyzes shirt PNG assets to find collar, shoulders, sleeves,
chest, waist and hem landmarks for TPS warping alignment.
"""

import cv2
import numpy as np
import logging
from typing import Optional, Dict, Tuple, List
from pathlib import Path
from dataclasses import dataclass, field

from engine.coreutils import setup_logger, GarmentMeta

logger = setup_logger("garment_landmarks")


@dataclass
class GarmentLandmarks:
    """
    Semantic control points on a garment image.
    All coordinates in pixel space of the shirt image.
    """
    # Source shirt image dimensions
    width: int
    height: int

    # Key anatomical landmarks
    collar_center: Tuple[int, int] = (0, 0)   # Top of collar/neck opening
    collar_left: Tuple[int, int] = (0, 0)     # Left collar edge
    collar_right: Tuple[int, int] = (0, 0)    # Right collar edge

    shoulder_left: Tuple[int, int] = (0, 0)   # Left shoulder seam point
    shoulder_right: Tuple[int, int] = (0, 0)  # Right shoulder seam point

    sleeve_left_end: Tuple[int, int] = (0, 0)  # Left sleeve cuff center
    sleeve_right_end: Tuple[int, int] = (0, 0) # Right sleeve cuff center

    chest_left: Tuple[int, int] = (0, 0)       # Left chest mid point
    chest_right: Tuple[int, int] = (0, 0)      # Right chest mid point
    chest_center: Tuple[int, int] = (0, 0)     # Center chest

    waist_left: Tuple[int, int] = (0, 0)
    waist_right: Tuple[int, int] = (0, 0)
    waist_center: Tuple[int, int] = (0, 0)

    hem_left: Tuple[int, int] = (0, 0)         # Bottom-left hem corner
    hem_right: Tuple[int, int] = (0, 0)        # Bottom-right hem corner
    hem_center: Tuple[int, int] = (0, 0)

    # Garment bounding box within image
    content_bbox: Tuple[int, int, int, int] = (0, 0, 1, 1)  # x1,y1,x2,y2

    @property
    def shirt_width(self) -> int:
        x1, y1, x2, y2 = self.content_bbox
        return x2 - x1

    @property
    def shirt_height(self) -> int:
        x1, y1, x2, y2 = self.content_bbox
        return y2 - y1

    def source_points(self, include_sleeves: bool = True) -> np.ndarray:
        """Return ordered array of source control points for TPS warping."""
        pts = [
            self.collar_center,
            self.collar_left,
            self.collar_right,
            self.shoulder_left,
            self.shoulder_right,
            self.chest_left,
            self.chest_right,
            self.chest_center,
            self.waist_left,
            self.waist_right,
            self.waist_center,
            self.hem_left,
            self.hem_right,
            self.hem_center,
        ]
        if include_sleeves:
            pts += [self.sleeve_left_end, self.sleeve_right_end]
        return np.array(pts, dtype=np.float32)


class GarmentAnalyzer:
    """
    Automatically detects semantic landmarks on shirt assets.

    Uses a combination of:
    - Alpha channel analysis for garment boundary detection
    - Morphological operations to find collar/shoulder regions
    - Geometric reasoning with anatomical priors
    - GarmentMeta ratios for initial estimates (corrected by analysis)
    """

    def __init__(self):
        self._cache: Dict[str, GarmentLandmarks] = {}

    def analyze(
        self,
        shirt_image: np.ndarray,
        meta: Optional[GarmentMeta] = None,
        cache_key: Optional[str] = None,
    ) -> GarmentLandmarks:
        """
        Detect garment landmarks from shirt PNG.

        Args:
            shirt_image: BGRA shirt image (transparent background)
            meta: Optional metadata with ratio hints
            cache_key: Cache key to avoid re-analysis

        Returns:
            GarmentLandmarks with control points
        """
        if cache_key and cache_key in self._cache:
            return self._cache[cache_key]

        h, w = shirt_image.shape[:2]
        landmarks = GarmentLandmarks(width=w, height=h)

        # Ensure BGRA
        if shirt_image.shape[2] == 3:
            shirt_image = cv2.cvtColor(shirt_image, cv2.COLOR_BGR2BGRA)

        alpha = shirt_image[:, :, 3]

        # Find garment bounding box
        bbox = self._find_content_bbox(alpha)
        landmarks.content_bbox = bbox
        x1, y1, x2, y2 = bbox
        cw = x2 - x1  # content width
        ch = y2 - y1  # content height

        # Use meta ratios or defaults
        collar_y_ratio = meta.collar_y_ratio if meta else 0.08
        shoulder_y_ratio = meta.shoulder_y_ratio if meta else 0.15
        sleeve_ratio = meta.sleeve_end_ratio if meta else 0.55
        hem_ratio = meta.hem_y_ratio if meta else 0.95
        neck_x_ratio = meta.neck_x_ratio if meta else 0.5

        # ── Collar ──────────────────────────────────────────────
        collar_y = int(y1 + ch * collar_y_ratio)
        collar_cx = int(x1 + cw * neck_x_ratio)
        collar_cx = self._refine_collar_x(alpha, collar_y, x1, x2, collar_cx)

        # Find collar width by scanning alpha along collar_y
        collar_width = self._measure_collar_width(alpha, collar_y, x1, x2)
        half_cw = collar_width // 2

        landmarks.collar_center = (collar_cx, collar_y)
        landmarks.collar_left = (collar_cx - half_cw, collar_y)
        landmarks.collar_right = (collar_cx + half_cw, collar_y)

        # ── Shoulders ────────────────────────────────────────────
        shoulder_y = int(y1 + ch * shoulder_y_ratio)
        left_x, right_x = self._find_shoulder_points(alpha, shoulder_y, x1, x2)

        landmarks.shoulder_left = (left_x, shoulder_y)
        landmarks.shoulder_right = (right_x, shoulder_y)

        # ── Sleeves ──────────────────────────────────────────────
        sleeve_y = int(y1 + ch * sleeve_ratio)
        sl_x, sr_x = self._find_sleeve_ends(alpha, sleeve_y, shoulder_y, x1, x2)

        landmarks.sleeve_left_end = (sl_x, sleeve_y)
        landmarks.sleeve_right_end = (sr_x, sleeve_y)

        # ── Chest ────────────────────────────────────────────────
        chest_y = int(y1 + ch * 0.35)
        cl_x, cr_x = self._find_torso_edges(alpha, chest_y, x1, x2)
        chest_cx = (cl_x + cr_x) // 2

        landmarks.chest_left = (cl_x, chest_y)
        landmarks.chest_right = (cr_x, chest_y)
        landmarks.chest_center = (chest_cx, chest_y)

        # ── Waist ─────────────────────────────────────────────────
        waist_y = int(y1 + ch * 0.65)
        wl_x, wr_x = self._find_torso_edges(alpha, waist_y, x1, x2)
        waist_cx = (wl_x + wr_x) // 2

        landmarks.waist_left = (wl_x, waist_y)
        landmarks.waist_right = (wr_x, waist_y)
        landmarks.waist_center = (waist_cx, waist_y)

        # ── Hem ───────────────────────────────────────────────────
        hem_y = int(y1 + ch * hem_ratio)
        hl_x, hr_x = self._find_torso_edges(alpha, hem_y, x1, x2)
        hem_cx = (hl_x + hr_x) // 2

        landmarks.hem_left = (hl_x, hem_y)
        landmarks.hem_right = (hr_x, hem_y)
        landmarks.hem_center = (hem_cx, hem_y)

        if cache_key:
            self._cache[cache_key] = landmarks

        logger.debug(
            f"Landmarks: collar={landmarks.collar_center}, "
            f"L-shoulder={landmarks.shoulder_left}, "
            f"R-shoulder={landmarks.shoulder_right}"
        )
        return landmarks

    def _find_content_bbox(self, alpha: np.ndarray) -> Tuple[int, int, int, int]:
        """Find tight bounding box of non-transparent content."""
        _, thresh = cv2.threshold(alpha, 10, 255, cv2.THRESH_BINARY)
        rows = np.any(thresh > 0, axis=1)
        cols = np.any(thresh > 0, axis=0)
        if not rows.any():
            h, w = alpha.shape
            return 0, 0, w, h
        y1, y2 = np.where(rows)[0][[0, -1]]
        x1, x2 = np.where(cols)[0][[0, -1]]
        return int(x1), int(y1), int(x2) + 1, int(y2) + 1

    def _refine_collar_x(
        self,
        alpha: np.ndarray,
        collar_y: int,
        x1: int,
        x2: int,
        default_cx: int,
    ) -> int:
        """Find horizontal center of collar opening."""
        if collar_y >= alpha.shape[0]:
            return default_cx

        row = alpha[collar_y, x1:x2]
        # Collar opening = transparent (low alpha) in center
        transparent = row < 30
        if transparent.any():
            t_idxs = np.where(transparent)[0]
            cx = int(np.mean(t_idxs)) + x1
            return cx
        return default_cx

    def _measure_collar_width(
        self,
        alpha: np.ndarray,
        collar_y: int,
        x1: int,
        x2: int,
    ) -> int:
        """Measure width of collar/neck opening."""
        if collar_y >= alpha.shape[0]:
            return (x2 - x1) // 4

        row = alpha[collar_y, x1:x2]
        transparent = row < 30
        t_count = np.sum(transparent)
        return max(20, min(int(t_count), (x2 - x1) // 3))

    def _find_shoulder_points(
        self,
        alpha: np.ndarray,
        y: int,
        x1: int,
        x2: int,
    ) -> Tuple[int, int]:
        """Find leftmost and rightmost shirt edges at shoulder height."""
        y = min(y, alpha.shape[0] - 1)
        row = alpha[y, x1:x2]
        opaque = row > 30
        if not opaque.any():
            return x1, x2
        left = int(np.where(opaque)[0][0])+ x1
        right = int(np.where(opaque)[0][-1])+ x1
        return left, right

    def _find_sleeve_ends(
        self,
        alpha: np.ndarray,
        sleeve_y: int,
        shoulder_y: int,
        x1: int,
        x2: int,
    ) -> Tuple[int, int]:
        """Find sleeve cuff endpoints by scanning vertically from shoulder level."""
        h = alpha.shape[0]

        # Left sleeve: find leftmost opaque pixel in left half at sleeve_y
        sleeve_y = min(sleeve_y, h - 1)
        mid_x = (x1 + x2) // 2

        # Scan the left half
        left_col = alpha[shoulder_y:sleeve_y, x1:mid_x]
        left_mask = left_col > 30
        if left_mask.any():
            rows, cols = np.where(left_mask)
            if len(cols) > 0:
                sl_x = max(x1, int(np.min(cols)) + x1 - 5) 
            else:
                sl_x = x1
        else:
            sl_x = x1

        # Right sleeve: find rightmost opaque in right half
        right_col = alpha[shoulder_y:sleeve_y, mid_x:x2]
        right_mask = right_col > 30
        if right_mask.any():
            rows, cols = np.where(right_mask)
            if len(cols) > 0:
                sr_x = min(x2, int(np.max(cols)) + mid_x + 5)
            else:
                sr_x = x2
        else:
            sr_x = x2

        sl_y = sleeve_y
        sr_y = sleeve_y

        return sl_x, sr_x

    def _find_torso_edges(
        self,
        alpha: np.ndarray,
        y: int,
        x1: int,
        x2: int,
    ) -> Tuple[int, int]:
        """Find left/right edges of torso at given y position."""
        y = min(y, alpha.shape[0] - 1)
        row = alpha[y, x1:x2]
        opaque = row > 30
        if not opaque.any():
            return x1, x2
        left = int(np.where(opaque)[0][0]) + x1
        right = int(np.where(opaque)[0][-1]) + x1
        return left, right

    def visualize_landmarks(
        self,
        shirt_image: np.ndarray,
        landmarks: GarmentLandmarks,
    ) -> np.ndarray:
        """Draw detected landmarks on shirt for debugging."""
        vis = shirt_image.copy()
        if vis.shape[2] == 4:
            # Composite over white
            alpha = vis[:, :, 3:4].astype(np.float32) / 255
            bgr = vis[:, :, :3].astype(np.float32)
            white = np.ones_like(bgr) * 255
            vis_bgr = (bgr * alpha + white * (1 - alpha)).astype(np.uint8)
        else:
            vis_bgr = vis

        # Draw landmark groups
        groups = [
            ([landmarks.collar_center, landmarks.collar_left, landmarks.collar_right], (255, 165, 0), "Collar"),
            ([landmarks.shoulder_left, landmarks.shoulder_right], (0, 255, 0), "Shoulder"),
            ([landmarks.sleeve_left_end, landmarks.sleeve_right_end], (255, 0, 255), "Sleeve"),
            ([landmarks.chest_left, landmarks.chest_right, landmarks.chest_center], (0, 255, 255), "Chest"),
            ([landmarks.waist_left, landmarks.waist_right, landmarks.waist_center], (255, 255, 0), "Waist"),
            ([landmarks.hem_left, landmarks.hem_right, landmarks.hem_center], (0, 165, 255), "Hem"),
        ]

        for pts, color, label in groups:
            for pt in pts:
                cv2.circle(vis_bgr, pt, 5, color, -1)
                cv2.circle(vis_bgr, pt, 6, (0, 0, 0), 1)

        return vis_bgr


__all__ = ["GarmentAnalyzer", "GarmentLandmarks"]