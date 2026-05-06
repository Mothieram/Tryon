"""
shadow_engine.py - Dynamic Shadow and Lighting System
Creates realistic cloth shadows under arms and at collar,
and adapts shirt appearance to scene lighting conditions.
"""

import cv2
import numpy as np
import logging
from typing import Optional, Tuple

from engine.coreutils import setup_logger, PoseKeypoints, get_brightness, estimate_ambient_color

logger = setup_logger("shadow")


class ShadowEngine:
    """
    Creates realistic shadow effects and lighting adaptation for shirt overlay.

    Features:
    - Soft shadow under arms/collar
    - Ambient light color temperature matching
    - Brightness normalization between shirt and scene
    - Directional shadow from estimated light source
    - Frame-to-frame stable shadow (no flickering)
    """

    def __init__(
        self,
        shadow_intensity: float = 0.45,
        shadow_blur: int = 25,
        light_adaptation: bool = True,
        light_sample_every_n: int = 5,
    ):
        self.shadow_intensity = shadow_intensity
        self.shadow_blur = shadow_blur
        self.light_adaptation = light_adaptation
        # Re-sample scene brightness/colour every N frames. The smoothing
        # weight is 0.2, so 95% of the value already comes from history —
        # sampling every frame is pure overhead.
        self.light_sample_every_n = max(1, int(light_sample_every_n))

        # Cached brightness for stability
        self._avg_brightness: float = 0.5
        self._avg_color: Tuple[int, int, int] = (128, 128, 128)
        self._light_sample_counter: int = 0

    def apply_shadows(
        self,
        frame: np.ndarray,
        shirt_region: np.ndarray,
        pose: PoseKeypoints,
        warped_shirt: np.ndarray,
    ) -> np.ndarray:
        """
        Apply shadow effects under arms and collar onto frame.

        Args:
            frame: Current BGR frame (shirt already blended)
            shirt_region: Mask of where shirt was rendered
            pose: Body pose keypoints
            warped_shirt: The warped shirt BGRA image

        Returns:
            Frame with shadows applied
        """
        result = frame.copy()
        h, w = frame.shape[:2]
        shirt_norm = None
        if shirt_region is not None:
            sr = np.asarray(shirt_region)
            if sr.ndim == 3:
                sr = sr[:, :, 0]
            if sr.shape != (h, w):
                sr = cv2.resize(sr, (w, h), interpolation=cv2.INTER_NEAREST)
            shirt_norm = np.clip(sr.astype(np.float32) / 255.0, 0.0, 1.0)

        # ── Arm shadow ────────────────────────────────────────────
        arm_shadow = self._compute_arm_shadow(pose, h, w, shirt_region)
        if arm_shadow is not None:
            result = self._apply_shadow_layer(result, arm_shadow)

        # ── Collar shadow ─────────────────────────────────────────
        collar_shadow = self._compute_collar_shadow(pose, h, w)
        if collar_shadow is not None:
            if shirt_norm is not None:
                collar_shadow = collar_shadow * shirt_norm
            result = self._apply_shadow_layer(result, collar_shadow, intensity=0.3)

        return result

    def adapt_shirt_lighting(
        self,
        shirt_bgra: np.ndarray,
        frame: np.ndarray,
        shirt_x: int,
        shirt_y: int,
    ) -> np.ndarray:
        """
        Adapt shirt colors to match scene lighting conditions.

        Args:
            shirt_bgra: Warped shirt BGRA
            frame: Camera frame for brightness reference
            shirt_x, shirt_y: Placement coordinates

        Returns:
            Lighting-adjusted shirt BGRA
        """
        if not self.light_adaptation:
            return shirt_bgra

        # Sample brightness from frame area where shirt will appear
        sh, sw = shirt_bgra.shape[:2]
        fh, fw = frame.shape[:2]

        x1, y1 = max(0, shirt_x), max(0, shirt_y)
        x2, y2 = min(fw, shirt_x + sw), min(fh, shirt_y + sh)

        sample_now = (self._light_sample_counter % self.light_sample_every_n) == 0
        self._light_sample_counter += 1
        if sample_now and x2 > x1 and y2 > y1:
            region = frame[y1:y2, x1:x2]
            scene_brightness = get_brightness(region)
            self._avg_brightness = self._avg_brightness * 0.8 + scene_brightness * 0.2

            ambient = estimate_ambient_color(frame)
            self._avg_color = tuple(
                int(self._avg_color[i] * 0.8 + ambient[i] * 0.2)
                for i in range(3)
            )

        result = shirt_bgra.copy()
        shirt_brightness = self._estimate_shirt_brightness(shirt_bgra)

        # Adjust only if there's a significant difference. Tighter clamp
        # (0.85, 1.15) than before (0.65, 1.35) — the wide range was washing
        # the shirt out so much it looked like a flat colour patch instead
        # of fabric. We give up some scene-matching to preserve the shirt's
        # own texture / tonal range.
        if shirt_brightness > 0.05 and abs(shirt_brightness - self._avg_brightness) > 0.1:
            ratio = self._avg_brightness / shirt_brightness
            ratio = np.clip(ratio, 0.85, 1.15)

            # Apply to BGR channels only, preserve alpha
            bgr = result[:, :, :3].astype(np.float32)
            bgr = np.clip(bgr * ratio, 0, 255).astype(np.uint8)
            result[:, :, :3] = bgr

        # Apply subtle ambient color tint. Halved from 0.08 → 0.04 for the
        # same reason — the tint was visibly muddying the shirt's hue.
        if self._avg_color != (128, 128, 128):
            tint = np.array(self._avg_color, dtype=np.float32)
            neutral = np.array([128.0, 128.0, 128.0])
            tint_strength = 0.04

            bgr = result[:, :, :3].astype(np.float32)
            tint_offset = (tint - neutral) * tint_strength
            bgr = np.clip(bgr + tint_offset, 0, 255).astype(np.uint8)
            result[:, :, :3] = bgr

        return result

    def _compute_arm_shadow(
        self,
        pose: PoseKeypoints,
        h: int,
        w: int,
        shirt_region: Optional[np.ndarray],
    ) -> Optional[np.ndarray]:
        """
        Compute soft shadow cast by arms onto shirt fabric.
        Shadow appears below/around arm positions over the shirt.
        """
        ls = pose.left_shoulder
        rs = pose.right_shoulder
        le = pose.left_elbow
        re = pose.right_elbow
        lw = pose.left_wrist
        rw = pose.right_wrist

        shadow_map = np.zeros((h, w), dtype=np.float32)
        sw_body = pose.shoulder_width
        if sw_body < 5:
            return None

        # Slimmer shoulder-elbow band (was 0.12 — produced a visible dark
        # rectangle on the upper chest when the arms were extended). The
        # forearm band stays a bit thinner still.
        upper_arm_width = max(6, int(sw_body * 0.07))
        forearm_width = max(5, int(sw_body * 0.06))

        # Offset the shadow downward proportional to shoulder width so the
        # cast shadow falls UNDER the arm (on the lower chest / side of
        # torso) rather than directly on the shoulder. A fixed 4 px offset
        # was too small for typical webcam framing and put the shadow on
        # top of the shirt's shoulder.
        shadow_offset = max(6, int(sw_body * 0.08))

        def draw_shadow_line(p1, p2, width):
            if p1 and p2 and p1.valid and p2.valid:
                pt1 = (p1.to_tuple()[0], p1.to_tuple()[1] + shadow_offset)
                pt2 = (p2.to_tuple()[0], p2.to_tuple()[1] + shadow_offset)
                cv2.line(shadow_map, pt1, pt2, 1.0, width)
                cv2.circle(shadow_map, pt1, width // 2, 1.0, -1)
                cv2.circle(shadow_map, pt2, width // 2, 1.0, -1)

        # NOTE: only the forearm (elbow → wrist) gets a shadow now. The
        # shoulder → elbow segment was darkening the chest in T-pose,
        # because for a horizontal arm the shoulder and elbow share a y
        # coordinate and any vertical offset still drops the shadow onto
        # the upper torso. Forearm shadow alone reads as a real cast
        # shadow when the arm is in front of the body, and is invisible
        # when the arm is extended away.
        draw_shadow_line(le, lw, forearm_width)
        draw_shadow_line(re, rw, forearm_width)
        # Suppress unused variable warning while keeping the var available
        # in case a future config wants to reintroduce upper-arm shadow.
        _ = upper_arm_width

        # Blur for soft shadow
        ksize = self.shadow_blur | 1
        shadow_map = cv2.GaussianBlur(shadow_map, (ksize, ksize), self.shadow_blur / 4)

        # Only apply shadow over shirt region
        if shirt_region is not None:
            shirt_norm = shirt_region.astype(np.float32) / 255.0
            shadow_map = shadow_map * shirt_norm

        shadow_map = np.clip(shadow_map * self.shadow_intensity, 0, 0.5)
        return shadow_map

    def _compute_collar_shadow(
        self,
        pose: PoseKeypoints,
        h: int,
        w: int,
    ) -> Optional[np.ndarray]:
        """Small shadow under collar/neck area."""
        ls = pose.left_shoulder
        rs = pose.right_shoulder
        nose = pose.nose

        if not (ls and rs and ls.valid and rs.valid):
            return None

        shadow_map = np.zeros((h, w), dtype=np.float32)
        sw_body = pose.shoulder_width
        neck_cx = int((ls.x + rs.x) / 2)
        collar_y = int((ls.y + rs.y) / 2 - sw_body * 0.02)

        collar_w = int(sw_body * 0.2)
        collar_h = int(sw_body * 0.06)

        cv2.ellipse(
            shadow_map,
            (neck_cx, collar_y + collar_h),
            (collar_w, collar_h),
            0, 0, 360, 0.8, -1
        )

        shadow_map = cv2.GaussianBlur(shadow_map, (15, 15), 0)
        return shadow_map

    def _apply_shadow_layer(
        self,
        frame: np.ndarray,
        shadow: np.ndarray,
        intensity: float = 1.0,
    ) -> np.ndarray:
        """Darken frame pixels according to shadow map (bbox-local)."""
        # Find the shadow's bounding box; outside it the multiplier is 1
        # so there's nothing to do. Saves a full-frame float32 cast.
        thresh = max(1e-3, 0.01 * float(intensity))
        ys, xs = np.where(shadow > thresh)
        if xs.size == 0:
            return frame
        x1 = int(xs.min())
        x2 = int(xs.max()) + 1
        y1 = int(ys.min())
        y2 = int(ys.max()) + 1
        roi = frame[y1:y2, x1:x2].astype(np.float32, copy=False)
        s = shadow[y1:y2, x1:x2] * intensity
        roi *= (1.0 - s)[:, :, None]
        out = frame.copy()
        out[y1:y2, x1:x2] = np.clip(roi, 0, 255).astype(np.uint8)
        return out

    def _estimate_shirt_brightness(self, shirt_bgra: np.ndarray) -> float:
        """Estimate mean brightness of visible shirt pixels."""
        if shirt_bgra.shape[2] == 4:
            alpha = shirt_bgra[:, :, 3] > 30
            if not alpha.any():
                return 0.5
            bgr = shirt_bgra[:, :, :3]
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            return float(np.mean(gray[alpha])) / 255.0
        gray = cv2.cvtColor(shirt_bgra[:, :, :3], cv2.COLOR_BGR2GRAY)
        return float(np.mean(gray)) / 255.0


__all__ = ["ShadowEngine"]