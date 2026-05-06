"""
utils.py - Shared utilities for Virtual Try-On System
Provides image processing helpers, math utilities, logging setup,
performance profiling, and common data structures.
"""

import cv2
import numpy as np
import logging
import time
import functools
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass, field
from pathlib import Path
import colorsys


# ─────────────────────────────────────────────
# Logging Setup
# ─────────────────────────────────────────────

def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Create a colored, well-formatted logger for a module."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(level)

        try:
            import colorlog
            fmt = colorlog.ColoredFormatter(
                "%(log_color)s[%(asctime)s] %(name)s %(levelname)s%(reset)s - %(message)s",
                datefmt="%H:%M:%S",
                log_colors={
                    "DEBUG": "cyan",
                    "INFO": "green",
                    "WARNING": "yellow",
                    "ERROR": "red",
                    "CRITICAL": "red,bg_white",
                },
            )
        except ImportError:
            fmt = logging.Formatter(
                "[%(asctime)s] %(name)s %(levelname)s - %(message)s",
                datefmt="%H:%M:%S",
            )

        handler.setFormatter(fmt)
        logger.addHandler(handler)

    return logger


logger = setup_logger("utils")


# ─────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────

@dataclass
class Keypoint:
    """Single body keypoint with confidence."""
    x: float
    y: float
    confidence: float = 1.0

    @property
    def valid(self) -> bool:
        return self.confidence > 0.3

    def to_tuple(self) -> Tuple[int, int]:
        return (int(self.x), int(self.y))

    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y])


@dataclass
class PoseKeypoints:
    """
    Full body pose with COCO 17-keypoint format.
    Indices: 0=nose, 1=left_eye, 2=right_eye, 3=left_ear, 4=right_ear,
             5=left_shoulder, 6=right_shoulder, 7=left_elbow, 8=right_elbow,
             9=left_wrist, 10=right_wrist, 11=left_hip, 12=right_hip,
             13=left_knee, 14=right_knee, 15=left_ankle, 16=right_ankle
    """
    keypoints: List[Keypoint] = field(default_factory=list)
    confidence: float = 0.0

    # Named accessors
    @property
    def nose(self) -> Optional[Keypoint]:
        return self.keypoints[0] if len(self.keypoints) > 0 else None

    @property
    def left_shoulder(self) -> Optional[Keypoint]:
        return self.keypoints[5] if len(self.keypoints) > 5 else None

    @property
    def right_shoulder(self) -> Optional[Keypoint]:
        return self.keypoints[6] if len(self.keypoints) > 6 else None

    @property
    def left_elbow(self) -> Optional[Keypoint]:
        return self.keypoints[7] if len(self.keypoints) > 7 else None

    @property
    def right_elbow(self) -> Optional[Keypoint]:
        return self.keypoints[8] if len(self.keypoints) > 8 else None

    @property
    def left_wrist(self) -> Optional[Keypoint]:
        return self.keypoints[9] if len(self.keypoints) > 9 else None

    @property
    def right_wrist(self) -> Optional[Keypoint]:
        return self.keypoints[10] if len(self.keypoints) > 10 else None

    @property
    def left_hip(self) -> Optional[Keypoint]:
        return self.keypoints[11] if len(self.keypoints) > 11 else None

    @property
    def right_hip(self) -> Optional[Keypoint]:
        return self.keypoints[12] if len(self.keypoints) > 12 else None

    @property
    def valid_keypoints(self) -> int:
        return sum(1 for kp in self.keypoints if kp.valid)

    def is_usable(self, min_keypoints: int = 6) -> bool:
        return self.valid_keypoints >= min_keypoints

    def midpoint(self, kp1: Optional[Keypoint], kp2: Optional[Keypoint]) -> Optional[np.ndarray]:
        if kp1 and kp2 and kp1.valid and kp2.valid:
            return (kp1.to_array() + kp2.to_array()) / 2
        return None

    @property
    def shoulder_midpoint(self) -> Optional[np.ndarray]:
        return self.midpoint(self.left_shoulder, self.right_shoulder)

    @property
    def hip_midpoint(self) -> Optional[np.ndarray]:
        return self.midpoint(self.left_hip, self.right_hip)

    @property
    def shoulder_width(self) -> float:
        ls, rs = self.left_shoulder, self.right_shoulder
        if ls and rs and ls.valid and rs.valid:
            return float(np.linalg.norm(ls.to_array() - rs.to_array()))
        return 0.0

    @property
    def torso_height(self) -> float:
        sm = self.shoulder_midpoint
        hm = self.hip_midpoint
        if sm is not None and hm is not None:
            return float(np.linalg.norm(sm - hm))
        return 0.0

    @property
    def torso_angle(self) -> float:
        """Returns torso rotation angle in degrees."""
        ls, rs = self.left_shoulder, self.right_shoulder
        if ls and rs and ls.valid and rs.valid:
            dx = rs.x - ls.x
            dy = rs.y - ls.y
            return float(np.degrees(np.arctan2(dy, dx)))
        return 0.0


@dataclass
class GarmentMeta:
    """Metadata for a shirt asset."""
    path: str
    name: str
    collar_y_ratio: float = 0.08   # collar top as fraction of shirt height
    shoulder_y_ratio: float = 0.15  # shoulder line ratio
    sleeve_end_ratio: float = 0.55  # sleeve end position ratio
    hem_y_ratio: float = 0.95       # bottom hem ratio
    neck_x_ratio: float = 0.5       # neck center x ratio


# ─────────────────────────────────────────────
# Image Utilities
# ─────────────────────────────────────────────

def ensure_bgra(img: np.ndarray) -> np.ndarray:
    """Convert image to BGRA if not already."""
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    if img.shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    return img


def ensure_bgr(img: np.ndarray) -> np.ndarray:
    """Convert image to BGR."""
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


def alpha_blend(
    background: np.ndarray,
    overlay: np.ndarray,
    alpha_mask: Optional[np.ndarray] = None,
    x: int = 0,
    y: int = 0,
) -> np.ndarray:
    """
    Blend overlay onto background at position (x, y).
    Uses overlay's alpha channel or provided alpha_mask.
    Returns modified background (BGR).
    """
    bg = background.copy()
    h, w = overlay.shape[:2]
    bh, bw = bg.shape[:2]

    # Clip to background bounds
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(bw, x + w), min(bh, y + h)

    if x2 <= x1 or y2 <= y1:
        return bg

    # Slice regions
    ov_x1 = x1 - x
    ov_y1 = y1 - y
    ov_x2 = ov_x1 + (x2 - x1)
    ov_y2 = ov_y1 + (y2 - y1)

    bg_region = bg[y1:y2, x1:x2].astype(np.float32)
    ov_region = overlay[ov_y1:ov_y2, ov_x1:ov_x2]

    # Get alpha
    if alpha_mask is not None:
        alpha = alpha_mask[ov_y1:ov_y2, ov_x1:ov_x2].astype(np.float32) / 255.0
    elif overlay.shape[2] == 4:
        alpha = ov_region[:, :, 3].astype(np.float32) / 255.0
    else:
        alpha = np.ones((ov_y2 - ov_y1, ov_x2 - ov_x1), dtype=np.float32)

    alpha_3 = np.stack([alpha, alpha, alpha], axis=2)

    ov_bgr = ov_region[:, :, :3].astype(np.float32)
    blended = bg_region * (1 - alpha_3) + ov_bgr * alpha_3
    bg[y1:y2, x1:x2] = np.clip(blended, 0, 255).astype(np.uint8)

    return bg


def feather_mask(mask: np.ndarray, radius: int = 15) -> np.ndarray:
    """Apply Gaussian feathering to a binary/alpha mask."""
    if radius <= 0:
        return mask
    ksize = radius * 2 + 1
    return cv2.GaussianBlur(mask.astype(np.float32), (ksize, ksize), radius / 3)


def resize_with_aspect(
    img: np.ndarray,
    target_w: int,
    target_h: int,
    interpolation: int = cv2.INTER_LINEAR,
) -> np.ndarray:
    """Resize image to fit within target dimensions preserving aspect ratio."""
    h, w = img.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(img, (new_w, new_h), interpolation=interpolation)


def get_brightness(frame: np.ndarray) -> float:
    """Estimate perceptual brightness of a BGR frame [0-1]."""
    gray = cv2.cvtColor(ensure_bgr(frame), cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray)) / 255.0


def adjust_brightness_contrast(
    img: np.ndarray,
    brightness_factor: float,
    target: float = 0.5,
) -> np.ndarray:
    """Adjust image brightness to match a target level."""
    if abs(brightness_factor - target) < 0.05:
        return img
    alpha = target / max(brightness_factor, 0.01)
    alpha = np.clip(alpha, 0.6, 1.4)
    return np.clip(img.astype(np.float32) * alpha, 0, 255).astype(np.uint8)


# ─────────────────────────────────────────────
# Geometry Utilities
# ─────────────────────────────────────────────

def point_distance(p1: np.ndarray, p2: np.ndarray) -> float:
    return float(np.linalg.norm(p1 - p2))


def rotate_point(point: np.ndarray, center: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate a 2D point around a center."""
    angle_rad = np.radians(angle_deg)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    shifted = point - center
    rotated = np.array([
        cos_a * shifted[0] - sin_a * shifted[1],
        sin_a * shifted[0] + cos_a * shifted[1],
    ])
    return rotated + center


def interpolate_points(
    p1: np.ndarray,
    p2: np.ndarray,
    t: float,
) -> np.ndarray:
    """Linear interpolation between two points."""
    return p1 * (1 - t) + p2 * t


def smooth_value(current: float, new: float, alpha: float = 0.3) -> float:
    """Exponential moving average smoothing."""
    return current * (1 - alpha) + new * alpha


def smooth_array(current: np.ndarray, new: np.ndarray, alpha: float = 0.3) -> np.ndarray:
    """EMA smoothing for arrays/keypoints."""
    return current * (1 - alpha) + new * alpha


def build_rotation_matrix(angle_deg: float, center: Tuple[float, float]) -> np.ndarray:
    """Build 2x3 rotation matrix for cv2.warpAffine."""
    return cv2.getRotationMatrix2D(center, angle_deg, 1.0)


# ─────────────────────────────────────────────
# Performance Utilities
# ─────────────────────────────────────────────

class FPSCounter:
    """Rolling-window FPS counter."""

    def __init__(self, window: int = 30):
        self.window = window
        self._times: List[float] = []

    def tick(self) -> float:
        now = time.perf_counter()
        self._times.append(now)
        if len(self._times) > self.window:
            self._times.pop(0)
        if len(self._times) < 2:
            return 0.0
        elapsed = self._times[-1] - self._times[0]
        return (len(self._times) - 1) / elapsed if elapsed > 0 else 0.0

    @property
    def fps(self) -> float:
        if len(self._times) < 2:
            return 0.0
        elapsed = self._times[-1] - self._times[0]
        return (len(self._times) - 1) / elapsed if elapsed > 0 else 0.0


def timeit(func):
    """Decorator to log function execution time."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = (time.perf_counter() - start) * 1000
        logger.debug(f"{func.__name__} took {elapsed:.1f}ms")
        return result
    return wrapper


class FrameCache:
    """Cache warped shirt overlay to avoid recomputation on static poses."""

    def __init__(self, change_threshold: float = 5.0):
        self.threshold = change_threshold
        self._cached_frame: Optional[np.ndarray] = None
        self._cached_pose: Optional[np.ndarray] = None
        self._cached_shirt_idx: int = -1

    def is_valid(self, pose_array: np.ndarray, shirt_idx: int) -> bool:
        if self._cached_frame is None:
            return False
        if shirt_idx != self._cached_shirt_idx:
            return False
        if self._cached_pose is None:
            return False
        diff = np.mean(np.abs(pose_array - self._cached_pose))
        return diff < self.threshold

    def update(self, frame: np.ndarray, pose_array: np.ndarray, shirt_idx: int):
        self._cached_frame = frame.copy()
        self._cached_pose = pose_array.copy()
        self._cached_shirt_idx = shirt_idx

    @property
    def cached(self) -> Optional[np.ndarray]:
        return self._cached_frame


# ─────────────────────────────────────────────
# Color / Appearance
# ─────────────────────────────────────────────

def estimate_ambient_color(frame: np.ndarray) -> Tuple[int, int, int]:
    """Estimate dominant ambient light color from frame borders."""
    h, w = frame.shape[:2]
    border_pixels = np.concatenate([
        frame[:10, :, :3].reshape(-1, 3),
        frame[-10:, :, :3].reshape(-1, 3),
        frame[:, :10, :3].reshape(-1, 3),
        frame[:, -10:, :3].reshape(-1, 3),
    ])
    avg = np.mean(border_pixels, axis=0)
    return int(avg[0]), int(avg[1]), int(avg[2])


def create_placeholder_shirt(size: Tuple[int, int] = (400, 500), color: Tuple[int, int, int] = (30, 80, 160)) -> np.ndarray:
    """Create a simple placeholder shirt PNG for testing."""
    w, h = size
    img = np.zeros((h, w, 4), dtype=np.uint8)

    # Body
    body_pts = np.array([
        [w * 0.15, h * 0.2],
        [w * 0.85, h * 0.2],
        [w * 0.95, h * 0.4],
        [w * 0.90, h * 1.0],
        [w * 0.10, h * 1.0],
        [w * 0.05, h * 0.4],
    ], dtype=np.int32)

    # Left sleeve
    left_sleeve = np.array([
        [w * 0.15, h * 0.2],
        [w * 0.05, h * 0.4],
        [w * -0.1, h * 0.55],
        [w * -0.05, h * 0.6],
        [w * 0.15, h * 0.45],
    ], dtype=np.int32)

    # Right sleeve
    right_sleeve = np.array([
        [w * 0.85, h * 0.2],
        [w * 0.95, h * 0.4],
        [w * 1.1, h * 0.55],
        [w * 1.05, h * 0.6],
        [w * 0.85, h * 0.45],
    ], dtype=np.int32)

    # Collar
    collar_pts = np.array([
        [w * 0.35, h * 0.2],
        [w * 0.5, h * 0.12],
        [w * 0.65, h * 0.2],
        [w * 0.6, h * 0.28],
        [w * 0.5, h * 0.3],
        [w * 0.4, h * 0.28],
    ], dtype=np.int32)

    # Draw
    cv2.fillPoly(img, [body_pts], (*color, 255))
    cv2.fillPoly(img, [left_sleeve], (*color, 255))
    cv2.fillPoly(img, [right_sleeve], (*color, 255))
    darker = tuple(max(0, c - 40) for c in color)
    cv2.fillPoly(img, [collar_pts], (*darker, 255))

    # Add subtle texture lines
    for i in range(5, h, 20):
        cv2.line(img, (int(w * 0.1), i), (int(w * 0.9), i),
                 (min(255, color[0] + 20), min(255, color[1] + 20), min(255, color[2] + 20), 180), 1)

    return img


# ─────────────────────────────────────────────
# Exports
# ─────────────────────────────────────────────

__all__ = [
    "setup_logger",
    "Keypoint",
    "PoseKeypoints",
    "GarmentMeta",
    "ensure_bgra",
    "ensure_bgr",
    "alpha_blend",
    "feather_mask",
    "resize_with_aspect",
    "get_brightness",
    "adjust_brightness_contrast",
    "point_distance",
    "rotate_point",
    "interpolate_points",
    "smooth_value",
    "smooth_array",
    "build_rotation_matrix",
    "FPSCounter",
    "timeit",
    "FrameCache",
    "estimate_ambient_color",
    "create_placeholder_shirt",
]
