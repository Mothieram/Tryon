"""
occlusion_engine.py - Natural Body Part Occlusion System  [FIXED]
Ensures arms, hands, neck and head appear in front of the shirt
using layered compositing and alpha masking.

FIXES (2026-04-24):
- shirt_region mask no longer cuts holes: we use a UNION of parsing-torso
  and densepose-torso, heavily dilated, so the shirt always has a full mask.
- Import uses engine.coreutils (consistent with the rest of the project).
- composite() no longer floors shirt_alpha to 0 outside torso — instead it
  blends down gradually so edges are soft, not hard-cut.
- Arm/head occlusion masks are properly feathered before compositing.
"""

import cv2
import numpy as np
from typing import Optional, Dict, Tuple

from engine.coreutils import setup_logger, PoseKeypoints, feather_mask
from engine.parsing_engine import ParsedRegions
from engine.densepose_engine import TorsoMap

logger = setup_logger("occlusion")


class OcclusionEngine:
    """
    Natural occlusion compositing for virtual try-on.

    Layering order:
        Background → Shirt (on torso) → Arms → Neck/Face/Hair
    """

    def __init__(
        self,
        feather_radius: int = 14,
        trust_parser_for_body: bool = False,
        trust_parser_for_foreground: bool = True,
        trust_densepose: bool = True,
        trust_sam2: bool = True,
        densepose_polygon_pad_ratio: float = 0.15,
        inner_core_inset_ratio: float = 0.18,
        sleeve_coverage_ratio: float = 0.78,
        sleeve_thickness_ratio: float = 0.28,
        polygon_fill_strength: float = 0.95,
        trust_parser: Optional[bool] = None,
    ):
        # Three flags govern occlusion:
        #   - trust_densepose: use DensePose torso mask as the body region.
        #     ON by default — DensePose is reliable regardless of clothing.
        #   - trust_parser_for_body: same idea using SCHP. OFF by default —
        #     SCHP misclassifies bare chest as arm/face and carves holes.
        #   - trust_parser_for_foreground: parser arms/face/hair refine the
        #     in-front-of-shirt layer. ON. Safety-clipped to outside the
        #     shirt inner core so misclassifications can't paste skin on
        #     top of the shirt.
        if trust_parser is not None:
            trust_parser_for_body = bool(trust_parser)
            trust_parser_for_foreground = bool(trust_parser)
        self.feather_radius = feather_radius
        self.trust_parser_for_body = bool(trust_parser_for_body)
        self.trust_parser_for_foreground = bool(trust_parser_for_foreground)
        self.trust_densepose = bool(trust_densepose)
        self.trust_sam2 = bool(trust_sam2)
        self.densepose_polygon_pad_ratio = float(densepose_polygon_pad_ratio)
        self.inner_core_inset_ratio = float(inner_core_inset_ratio)
        # Sleeve coverage: how far down the arm the polygon extends
        # (0.55 = mid-bicep) and how thick the sleeve stroke is
        # (0.22 of shoulder width). Composite uses polygon_fill_strength
        # to render the dehaloed shirt color in polygon area where the
        # shirt PNG has no native alpha — that gives a tank-top PNG
        # visible sleeves on the body without editing the asset.
        self.sleeve_coverage_ratio = float(sleeve_coverage_ratio)
        self.sleeve_thickness_ratio = float(sleeve_thickness_ratio)
        self.polygon_fill_strength = float(np.clip(polygon_fill_strength, 0.0, 1.0))

    @property
    def trust_parser(self) -> bool:
        """Back-compat: True only when both granular flags are True."""
        return self.trust_parser_for_body and self.trust_parser_for_foreground

    @trust_parser.setter
    def trust_parser(self, value: bool) -> None:
        self.trust_parser_for_body = bool(value)
        self.trust_parser_for_foreground = bool(value)

    def _ensure_mask(self, mask, h: int, w: int) -> np.ndarray:
        if mask is None:
            return np.zeros((h, w), dtype=np.uint8)
        m = np.asarray(mask)
        if m.ndim == 3:
            m = m[:, :, 0]
        elif m.ndim != 2:
            return np.zeros((h, w), dtype=np.uint8)
        if m.shape != (h, w):
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
        return np.clip(m, 0, 255).astype(np.uint8)

    def _largest_component(self, mask: np.ndarray, min_area: int = 0) -> np.ndarray:
        binary = (mask > 0).astype(np.uint8)
        if not np.any(binary):
            return np.zeros_like(mask, dtype=np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        if num_labels <= 1:
            return (binary * 255).astype(np.uint8)
        areas = stats[1:, cv2.CC_STAT_AREA]
        best = int(np.argmax(areas)) + 1
        if int(stats[best, cv2.CC_STAT_AREA]) < max(1, int(min_area)):
            return np.zeros_like(mask, dtype=np.uint8)
        out = np.zeros_like(mask, dtype=np.uint8)
        out[labels == best] = 255
        return out

    def build_occlusion_masks(
        self,
        frame: np.ndarray,
        pose: PoseKeypoints,
        parsed: ParsedRegions,
        torso_map: TorsoMap,
        sam2_mask: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        h, w = frame.shape[:2]

        # Normalise all external masks first
        parsed.torso     = self._ensure_mask(parsed.torso, h, w)
        parsed.left_arm  = self._ensure_mask(parsed.left_arm, h, w)
        parsed.right_arm = self._ensure_mask(parsed.right_arm, h, w)
        parsed.face      = self._ensure_mask(parsed.face, h, w)
        parsed.hair      = self._ensure_mask(parsed.hair, h, w)
        parsed.legs      = self._ensure_mask(parsed.legs, h, w)
        torso_map.torso_mask = self._ensure_mask(torso_map.torso_mask, h, w)
        torso_map.neck_mask  = self._ensure_mask(torso_map.neck_mask, h, w)
        if torso_map.arm_masks:
            torso_map.arm_masks = {
                k: self._ensure_mask(v, h, w)
                for k, v in torso_map.arm_masks.items()
            }
        if sam2_mask is not None:
            sam2_mask = self._ensure_mask(sam2_mask, h, w)

        masks = {}

        # ── Shirt region ──────────────────────────────────────────────────────
        shirt_region = self._compute_shirt_region(
            torso_map, parsed, pose, h, w, sam2_mask=sam2_mask,
        )
        masks["shirt_region"] = shirt_region

        # ── Arm occlusion (arms appear IN FRONT of shirt) ─────────────────────
        arm_mask = self._compute_arm_occlusion(frame, parsed, pose, h, w)
        masks["arm_occlusion"] = arm_mask

        # ── Head/neck occlusion (head appears IN FRONT of shirt) ──────────────
        head_mask = self._compute_head_occlusion(parsed, pose, h, w)
        masks["head_occlusion"] = head_mask

        # ── Combined foreground ───────────────────────────────────────────────
        arm_mask  = self._ensure_mask(arm_mask, h, w)
        head_mask = self._ensure_mask(head_mask, h, w)
        foreground = cv2.bitwise_or(arm_mask, head_mask)
        foreground = feather_mask(foreground, self.feather_radius // 2)
        foreground = np.clip(foreground, 0, 255).astype(np.uint8)
        masks["foreground"] = foreground

        return masks

    def _compute_shirt_region(
        self,
        torso_map: TorsoMap,
        parsed: ParsedRegions,
        pose: PoseKeypoints,
        h: int,
        w: int,
        sam2_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Build the body region the shirt may occupy.

        Four signals can drive the silhouette, in order of preference:
          1. SAM2 body mask (PIXEL-PRECISE — best when available).
          2. DensePose torso (semantic but coarser).
          3. SCHP torso (opt-in via `trust_parser_for_body`; risky when
             shirtless because SCHP misclassifies bare chest as arm/face).
          4. Pose polygon (always available, used as both safety bound and
             fallback when no neural signal is reliable).

        The chosen torso mask is intersected with an EXPANDED pose polygon
        so a runaway neural mask can't push the shirt off the body.
        """
        pose_polygon = self._geometric_shirt_region(pose, h, w)
        shirt_region = pose_polygon

        # Build a DIRECTIONAL safety zone around the pose polygon:
        #   - extend UP by ~20% of shoulder width, so DensePose can fill the
        #     clavicle / upper chest area (covers the hard horizontal cut
        #     visible at the shoulders).
        #   - extend SIDES by ~8% for pose-noise tolerance.
        #   - do NOT extend DOWN — uniform dilation would let the
        #     DensePose torso (which often spills into the lap) drag the
        #     shirt hem past the hip line.
        sw = max(20.0, float(pose.shoulder_width)) if pose is not None else 80.0
        top_ext  = max(10, int(sw * 0.20))
        side_ext = max(6,  int(sw * 0.08))

        # Shift the polygon up by top_ext px and OR with the original to
        # extend the top boundary only.
        M_up = np.float32([[1, 0, 0], [0, 1, -top_ext]])
        shifted_up = cv2.warpAffine(
            pose_polygon, M_up, (w, h),
            flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )
        # Side-only dilation (small uniform kernel).
        side_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (side_ext * 2 + 1, side_ext * 2 + 1),
        )
        side_dilated = cv2.dilate(pose_polygon, side_kernel, iterations=1)
        # Combined safety zone: extended top + extended sides.
        safe_zone = cv2.bitwise_or(side_dilated, shifted_up)

        # 1) SAM2 body mask — preferred when available (pixel-precise).
        sam2_clean: Optional[np.ndarray] = None
        if self.trust_sam2 and sam2_mask is not None:
            sm = self._ensure_mask(sam2_mask, h, w)
            if np.any(sm > 0):
                sam2_clean = sm

        # 2) DensePose path — fallback when SAM2 isn't available.
        densepose_mask = None
        if self.trust_densepose and torso_map is not None:
            dp = self._ensure_mask(getattr(torso_map, "torso_mask", None), h, w)
            if np.any(dp > 0):
                densepose_mask = dp

        # 3) SCHP torso (opt-in).
        schp_mask = None
        if self.trust_parser_for_body and parsed is not None:
            sp = self._ensure_mask(getattr(parsed, "torso", None), h, w)
            if np.any(sp > 0):
                schp_mask = sp

        # Merge signals. SAM2 (when present) is unioned with DensePose +
        # SCHP for max coverage, then intersected with the safe zone.
        merged: Optional[np.ndarray] = None
        for src in (sam2_clean, densepose_mask, schp_mask):
            if src is None:
                continue
            merged = src if merged is None else cv2.bitwise_or(merged, src)

        if merged is not None:
            refined = cv2.bitwise_and(merged, safe_zone)
            # Adopt the neural silhouette only if it covers a meaningful
            # fraction of the polygon — otherwise the model was confused
            # and we fall back to the pose polygon.
            min_coverage = 0.30 if sam2_clean is not None else 0.40
            if cv2.countNonZero(refined) > min_coverage * cv2.countNonZero(pose_polygon):
                k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
                refined = cv2.morphologyEx(refined, cv2.MORPH_CLOSE, k_close)
                shirt_region = refined

        # When SAM2 is present and trusted, FINAL clip the shirt region
        # to SAM2's body mask. Without this clip, the polygon's sleeve
        # strokes (drawn in pose space) can extend past the actual body
        # in the camera frame and produce visible cyan "wings" floating
        # in empty air. SAM2's mask says where the body really is, so
        # clipping to it constrains the synthesised sleeves to land on
        # actual skin / body pixels.
        if sam2_clean is not None:
            # Slight dilation so SAM2's pixel-edge isn't too tight.
            sam2_dilated = cv2.dilate(
                sam2_clean,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
                iterations=1,
            )
            shirt_region = cv2.bitwise_and(shirt_region, sam2_dilated)

        # Subtract head from the body region so the collar opening is clean.
        if self.trust_parser_for_body and parsed is not None:
            face = self._ensure_mask(getattr(parsed, "face", None), h, w)
            hair = self._ensure_mask(getattr(parsed, "hair", None), h, w)
            head = cv2.bitwise_or(face, hair)
        else:
            head = self._geometric_head_mask(pose, h, w)
        if np.any(head > 0):
            head_exp = cv2.dilate(
                head,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
                iterations=1,
            )
            # Clip the head mask to ABOVE the shoulder line — the neck
            # ellipse otherwise extends to shoulder_y and would carve a
            # flat horizontal cut out of the polygon top.
            ls = pose.left_shoulder if pose is not None else None
            rs = pose.right_shoulder if pose is not None else None
            if ls and rs and ls.valid and rs.valid:
                shoulder_y = int(min(ls.y, rs.y))
                head_exp[shoulder_y:, :] = 0
            shirt_region = cv2.bitwise_and(shirt_region, cv2.bitwise_not(head_exp))

        # Edge smoothing whenever a neural mask was used (it's noisier).
        if merged is not None:
            shirt_region = cv2.GaussianBlur(shirt_region, (7, 7), 0)
            _, shirt_region = cv2.threshold(shirt_region, 30, 255, cv2.THRESH_BINARY)
        return shirt_region

    def _shirt_inner_core(
        self,
        pose: PoseKeypoints,
        h: int,
        w: int,
    ) -> np.ndarray:
        """Eroded shirt polygon — the region we trust as 'definitely shirt'.

        Parser-derived foreground (arms, face/hair) gets clipped to OUTSIDE
        this core, so even if SCHP labels bare chest as 'arm' the chest
        pixels can't be pasted back on top of the shirt. Includes the
        sleeve extensions so virtual sleeves are also protected.
        """
        sw = max(20.0, float(pose.shoulder_width)) if pose is not None else 80.0
        inset = max(8, int(sw * self.inner_core_inset_ratio))
        # Use the FULL polygon (chest + sleeves) so the upper arm sleeve
        # coverage is also protected from being overpainted by parser arms.
        core = self._geometric_shirt_region(pose, h, w)
        if not np.any(core):
            return core
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (inset, inset))
        return cv2.erode(core, kernel, iterations=1)

    def _geometric_shirt_region(self, pose: PoseKeypoints, h: int, w: int) -> np.ndarray:
        """Pose-only T-shirt silhouette polygon.

        Concave 10-vertex polygon shaped like a real T-shirt: sloped
        shoulders, a scooped neckline at the top center, and a slightly
        dropped hem at the bottom center. Replaces the previous flat
        rectangle that made the shirt look "pasted on".
        """
        mask = np.zeros((h, w), dtype=np.uint8)
        ls = pose.left_shoulder
        rs = pose.right_shoulder
        lh = pose.left_hip
        rh = pose.right_hip
        if not (ls and rs and ls.valid and rs.valid):
            return mask

        sw = max(20.0, float(pose.shoulder_width))

        if lh and rh and lh.valid and rh.valid:
            lh_x, lh_y = float(lh.x), float(lh.y)
            rh_x, rh_y = float(rh.x), float(rh.y)
        else:
            lh_x, lh_y = float(ls.x) + sw * 0.05, float(ls.y) + sw * 1.45
            rh_x, rh_y = float(rs.x) - sw * 0.05, float(rs.y) + sw * 1.45

        x_left  = min(float(ls.x), float(rs.x))
        x_right = max(float(ls.x), float(rs.x))
        hip_left  = min(lh_x, rh_x)
        hip_right = max(lh_x, rh_x)
        body_mid_x = (x_left + x_right) * 0.5

        # Tighter pads — was 0.18/0.10 which made the polygon ~36px wider
        # than the body at the shoulders, producing the "ghost" cyan edge
        # outside the actual shoulder line. 0.10/0.04 sits just at the
        # shoulder-seam line.
        shoulder_pad  = sw * 0.10
        hip_pad       = sw * 0.04
        # Slight inward indents at the armpit and waist so the polygon
        # follows real body curvature (was straight-vertical sides).
        armpit_indent = sw * 0.04
        waist_indent  = sw * 0.07

        shoulder_y   = (float(ls.y) + float(rs.y)) * 0.5
        # Lift the top edge by ~3% so the shirt covers the clavicle area
        # instead of cutting at the shoulder bone (which left a strip of
        # skin visible above the shirt in your screenshots).
        top_y        = shoulder_y - sw * 0.03
        armpit_y     = shoulder_y + sw * 0.18
        waist_y      = shoulder_y + sw * 0.50
        hem_y        = (lh_y + rh_y) * 0.5 + sw * 0.05

        # Neckline scoop at the top center.
        neck_cx     = (float(ls.x) + float(rs.x)) * 0.5
        neck_half   = sw * 0.13
        neck_dip_y  = shoulder_y + sw * 0.08

        hem_cx       = (hip_left + hip_right) * 0.5
        hem_drop_y   = hem_y + sw * 0.04

        # Side x-coords at each contour level. The polygon now follows the
        # body: widest at shoulders, narrower at armpits, narrowest at
        # waist, back out at hips, slight tuck at hem.
        l_top_x    = x_left  - shoulder_pad
        r_top_x    = x_right + shoulder_pad
        l_armpit_x = x_left  + armpit_indent
        r_armpit_x = x_right - armpit_indent
        l_waist_x  = body_mid_x - max(sw * 0.30, (body_mid_x - x_left)  - waist_indent)
        r_waist_x  = body_mid_x + max(sw * 0.30, (x_right - body_mid_x) - waist_indent)
        l_hip_x    = hip_left  - hip_pad
        r_hip_x    = hip_right + hip_pad
        l_hem_x    = hip_left  - hip_pad * 0.5
        r_hem_x    = hip_right + hip_pad * 0.5

        # Vertices clockwise from upper-left, with neckline scoop concavity
        # at top center and body curve indents on the sides.
        pts = np.array([
            (l_top_x,             top_y),
            (neck_cx - neck_half, shoulder_y),
            (neck_cx,             neck_dip_y),
            (neck_cx + neck_half, shoulder_y),
            (r_top_x,             top_y),
            (r_armpit_x,          armpit_y),
            (r_waist_x,           waist_y),
            (r_hip_x,             (lh_y + rh_y) * 0.5),
            (r_hem_x,             hem_y),
            (hem_cx,              hem_drop_y),
            (l_hem_x,             hem_y),
            (l_hip_x,             (lh_y + rh_y) * 0.5),
            (l_waist_x,           waist_y),
            (l_armpit_x,          armpit_y),
        ], dtype=np.int32)
        pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
        cv2.fillPoly(mask, [pts], 255)

        # Sleeve coverage: extend the polygon over each upper arm so the
        # shirt visibly continues onto the bicep, not just the chest.
        # Drawn as a thick line from shoulder to mid-bicep (or mid-elbow
        # if the elbow is detected). Composite blends the shirt color in
        # this region using `polygon_fill_strength`.
        thickness = max(12, int(sw * self.sleeve_thickness_ratio))
        coverage  = float(np.clip(self.sleeve_coverage_ratio, 0.0, 1.0))

        def _draw_sleeve(shoulder, elbow):
            """Draw a sleeve stroke from shoulder toward elbow.

            Guards against the "horizontal wing" artefact that appears when
            the elbow is off-frame: if the shoulder→elbow vector points
            mostly sideways or upward (i.e. arm not naturally hanging),
            the elbow estimate is unreliable and we fall back to a short
            straight-down sleeve instead of drawing into empty space.
            """
            if shoulder is None or not shoulder.valid:
                return
            sx, sy = float(shoulder.x), float(shoulder.y)
            ex = ey = None
            if elbow is not None and elbow.valid:
                edx = float(elbow.x) - sx
                edy = float(elbow.y) - sy
                # Reject elbow when:
                #   (a) it sits at/above the shoulder (sleeve would point
                #       up — wings into the air),
                #   (b) it's mostly horizontal (|dx| > |dy| * 1.2),
                #   (c) it's farther than 1.4 * shoulder-width away
                #       (extrapolation from off-frame keypoint).
                horizontal = abs(edx) > abs(edy) * 1.2
                too_far    = (edx * edx + edy * edy) > (sw * 1.4) ** 2
                if edy >= sw * 0.10 and not horizontal and not too_far:
                    ex, ey = float(elbow.x), float(elbow.y)
            if ex is None:
                # Fallback: short cap-sleeve straight down.
                ex, ey = sx, sy + sw * 0.45
            tx = int(round(sx + (ex - sx) * coverage))
            ty = int(round(sy + (ey - sy) * coverage))
            cv2.line(mask, (int(sx), int(sy)), (tx, ty), 255, thickness)
            cv2.circle(mask, (tx, ty), thickness // 2, 255, -1)

        _draw_sleeve(ls, getattr(pose, "left_elbow", None))
        _draw_sleeve(rs, getattr(pose, "right_elbow", None))
        return mask

    def _compute_arm_occlusion(
        self,
        frame: np.ndarray,
        parsed: ParsedRegions,
        pose: PoseKeypoints,
        h: int,
        w: int,
    ) -> np.ndarray:
        """Arms that should appear in front of the shirt.

        Pose-keypoint based only. Optionally augmented by parser arms /
        skin-detection when explicitly enabled. The aggressive default would
        otherwise put bare-chest pixels in front of the shirt for shirtless
        users (SCHP confuses chest as arm without a garment to anchor on).
        """
        arm_mask = self._geometric_arm_mask(pose, h, w)

        # Parser-derived arms are clipped to OUTSIDE the shirt's inner core
        # so SCHP mis-labels of bare chest as 'arm' can't be pasted on top
        # of the shirt. The geometric stroke alone covers the actual arm
        # bone area; parser adds the silhouette detail outside.
        if self.trust_parser_for_foreground and parsed is not None:
            if np.any(parsed.left_arm > 0) or np.any(parsed.right_arm > 0):
                parser_arms = cv2.bitwise_or(parsed.left_arm, parsed.right_arm)
                inner_core = self._shirt_inner_core(pose, h, w)
                if np.any(inner_core):
                    parser_arms = cv2.bitwise_and(
                        parser_arms, cv2.bitwise_not(inner_core)
                    )
                arm_mask = cv2.bitwise_or(arm_mask, parser_arms)

        # Hands always in front (wrist circles).
        arm_mask = cv2.bitwise_or(arm_mask, self._compute_hand_region(pose, h, w))

        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        arm_mask = cv2.morphologyEx(arm_mask, cv2.MORPH_CLOSE, k)
        arm_mask = feather_mask(arm_mask, self.feather_radius)
        return np.clip(arm_mask, 0, 255).astype(np.uint8)

    def _compute_head_occlusion(
        self,
        parsed: ParsedRegions,
        pose: PoseKeypoints,
        h: int,
        w: int,
    ) -> np.ndarray:
        """Head/neck region that sits above / in front of the shirt collar.

        Parser-derived face/hair masks are only consulted when explicitly
        trusted; otherwise we use a pose-keypoint head ellipse. SCHP mis-
        labels bare chest as 'face/hair' and pasting those pixels back in
        front of the shirt was producing the chest-holes in the screenshots.
        """
        head_mask = self._geometric_head_mask(pose, h, w)
        if self.trust_parser_for_foreground and parsed is not None:
            if np.any(parsed.face > 0) or np.any(parsed.hair > 0):
                parser_head = cv2.bitwise_or(parsed.face, parsed.hair)
                # Clip parser head to ABOVE shoulder line — chin/face
                # spillover into the chest from misclassification is
                # silently dropped here.
                ls = pose.left_shoulder if pose is not None else None
                rs = pose.right_shoulder if pose is not None else None
                if ls and rs and ls.valid and rs.valid:
                    shoulder_y = int(min(ls.y, rs.y))
                    parser_head[shoulder_y:, :] = 0
                head_mask = cv2.bitwise_or(head_mask, parser_head)

        # Include neck keypoint region
        ls = pose.left_shoulder
        rs = pose.right_shoulder
        nose = pose.nose
        if ls and rs and ls.valid and rs.valid and nose and nose.valid:
            neck_cx = int((ls.x + rs.x) / 2)
            neck_top = int(nose.y)
            neck_bot = int((ls.y + rs.y) / 2)
            sw = max(10.0, pose.shoulder_width)
            neck_w = max(14, int(sw * 0.18))
            neck_h = max(10, abs(neck_bot - neck_top) // 2 + 12)
            cv2.ellipse(head_mask,
                        (neck_cx, (neck_top + neck_bot) // 2),
                        (neck_w, neck_h), 0, 0, 360, 255, -1)

        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        head_mask = cv2.morphologyEx(head_mask, cv2.MORPH_CLOSE, k)
        head_mask = feather_mask(head_mask, self.feather_radius)
        return np.clip(head_mask, 0, 255).astype(np.uint8)

    def _geometric_arm_mask(self, pose: PoseKeypoints, h: int, w: int) -> np.ndarray:
        """Pose-keypoint arm strokes used for arm-in-front occlusion.

        Upper-arm stroke STARTS a fraction of the way from the shoulder
        toward the elbow — drawing the full upper arm as a 32px-thick line
        from the shoulder bone carved into the shirt's shoulder seam.
        Forearm + wrist still draw fully so raised-arm occlusion still works.
        """
        mask = np.zeros((h, w), dtype=np.uint8)
        sw = pose.shoulder_width
        if sw < 5:
            return mask
        arm_thick = max(10, int(sw * 0.11))
        forearm_thick = max(9, int(sw * 0.10))

        def draw_segment(p1, p2, thickness, start_t: float = 0.0):
            if p1 and p2 and p1.valid and p2.valid:
                a = np.array(p1.to_tuple(), dtype=np.float32)
                b = np.array(p2.to_tuple(), dtype=np.float32)
                a2 = a + (b - a) * float(start_t)
                cv2.line(mask, tuple(a2.astype(int)), tuple(b.astype(int)), 255, thickness)
                cv2.circle(mask, tuple(b.astype(int)), thickness // 2, 255, -1)

        # Upper arm: skip the first ~25% (the shoulder-seam region) so the
        # shirt's shoulder seam stays visible when arms hang at the sides.
        draw_segment(pose.left_shoulder,  pose.left_elbow,  arm_thick, start_t=0.25)
        draw_segment(pose.right_shoulder, pose.right_elbow, arm_thick, start_t=0.25)
        # Forearm + wrist: full length, full occlusion.
        draw_segment(pose.left_elbow,  pose.left_wrist,  forearm_thick)
        draw_segment(pose.right_elbow, pose.right_wrist, forearm_thick)
        return mask

    def _compute_hand_region(self, pose: PoseKeypoints, h: int, w: int) -> np.ndarray:
        mask = np.zeros((h, w), dtype=np.uint8)
        sw = pose.shoulder_width
        if sw < 5:
            return mask
        hand_r = max(14, int(sw * 0.12))
        for wrist in [pose.left_wrist, pose.right_wrist]:
            if wrist and wrist.valid:
                cv2.circle(mask, wrist.to_tuple(), hand_r, 255, -1)
        return mask

    def _geometric_head_mask(self, pose: PoseKeypoints, h: int, w: int) -> np.ndarray:
        mask = np.zeros((h, w), dtype=np.uint8)
        nose = pose.nose
        ls   = pose.left_shoulder
        rs   = pose.right_shoulder
        if not (nose and nose.valid):
            return mask
        sw = max(50.0, pose.shoulder_width)
        head_r = int(sw * 0.24)
        face_center = (int(nose.x), int(nose.y - head_r * 0.1))
        cv2.ellipse(mask, face_center, (head_r, int(head_r * 1.25)), 0, 0, 360, 255, -1)
        if ls and rs and ls.valid and rs.valid:
            neck_cx  = int((ls.x + rs.x) / 2)
            neck_top = int(nose.y + head_r * 0.6)
            neck_bot = int((ls.y + rs.y) / 2)
            neck_w   = max(10, int(sw * 0.16))
            neck_h   = max(10, abs(neck_bot - neck_top) // 2 + 6)
            cv2.ellipse(mask,
                        (neck_cx, (neck_top + neck_bot) // 2),
                        (neck_w, neck_h), 0, 0, 360, 255, -1)
        return mask

    def composite(
        self,
        frame: np.ndarray,
        warped_shirt: np.ndarray,
        placement_x: int,
        placement_y: int,
        occlusion_masks: Dict[str, np.ndarray],
        opacity: float = 0.95,
    ) -> np.ndarray:
        """
        IMPROVED Composite: background → shirt (soft masked) → body parts (feathered)
        
        Key improvements:
        - Proper edge feathering on all layers
        - Smart mask constraint that follows body shape
        - Gradual transparency at edges (not hard cutoff)
        - Better foreground integration with natural blending
        """
        h, w = frame.shape[:2]

        if warped_shirt is None:
            return frame

        sh, sw_s = warped_shirt.shape[:2]

        # ── Clip to frame bounds ──────────────────────────────────────────
        x1 = max(0, placement_x)
        y1 = max(0, placement_y)
        x2 = min(w, placement_x + sw_s)
        y2 = min(h, placement_y + sh)

        if x2 <= x1 or y2 <= y1:
            return frame

        # Mutate `frame` in place from here on — the caller already passes
        # a `result` slice that they own. Skipping the full-frame copy
        # saves ~1ms per 1080p frame.
        result = frame
        sx1 = x1 - placement_x
        sy1 = y1 - placement_y
        sx2 = sx1 + (x2 - x1)
        sy2 = sy1 + (y2 - y1)

        shirt_roi = warped_shirt[sy1:sy2, sx1:sx2]
        frame_roi = result[y1:y2, x1:x2].astype(np.float32)
        # Snapshot the *original* shirt-ROI pixels so the foreground
        # re-paste step can restore them after we blend the shirt in.
        # This is the only copy we need — the rest of the frame is
        # untouched, so we don't need to clone it.
        orig_roi_bytes = result[y1:y2, x1:x2].copy()

        roi_h, roi_w = shirt_roi.shape[:2]

        # ── 1. Native shirt alpha + sampled "fabric color" ─────────────────
        # Native alpha = the cloth PNG's intrinsic silhouette (warped).
        # `avg_fabric_bgr` = mean BGR of the cloth INTERIOR (eroded so the
        # warp's black border pixels are excluded). We use it as the
        # cloth color anywhere the polygon extends past the cloth's own
        # alpha — that way the sleeve fill is always proper fabric color,
        # never a leaked-in black border pixel.
        if shirt_roi.shape[2] == 4:
            shirt_alpha = shirt_roi[:, :, 3].astype(np.float32) / 255.0
        else:
            shirt_alpha = np.ones((roi_h, roi_w), dtype=np.float32)
        avg_fabric_bgr = self._sample_fabric_color(shirt_roi)

        # ── 2. Combine native cloth + polygon body region ─────────────────
        shirt_region = occlusion_masks.get("shirt_region")
        if shirt_region is not None:
            shirt_region = self._ensure_mask(shirt_region, h, w)
            region_roi = shirt_region[y1:y2, x1:x2].astype(np.float32) / 255.0
            native = shirt_alpha * region_roi
            fill = region_roi * float(self.polygon_fill_strength) * (1.0 - native)
        else:
            region_roi = None
            native = shirt_alpha
            fill = np.zeros_like(shirt_alpha)

        # ── 3. Apply user opacity ─────────────────────────────────────────
        native = native * opacity
        fill = fill * opacity

        # ── 4. Feather alphas for clean edges ─────────────────────────────
        native = cv2.GaussianBlur(native, (9, 9), 0)
        fill = cv2.GaussianBlur(fill, (9, 9), 0)

        # ── 5. Build effective fabric BGR per pixel ───────────────────────
        # Where the cloth has high native alpha → use the cloth's actual
        # warped pixels (so seams, stripes, prints all show through).
        # Where native alpha is low (sleeve extension area) → use the
        # sampled solid fabric color. Avoids black-rectangle artefacts
        # from cv2.remap's borderValue=(0,0,0,0).
        cloth_bgr = shirt_roi[:, :, :3].astype(np.float32)
        avg_3 = avg_fabric_bgr.reshape(1, 1, 3)
        native_a3 = (shirt_alpha[:, :, None]).clip(0.0, 1.0)
        fabric_bgr = cloth_bgr * native_a3 + avg_3 * (1.0 - native_a3)

        # ── 6. Composite within shirt bbox ───────────────────────────────
        # Order: bg → fabric@native → fabric@fill (one-pass equivalent).
        total_a = (native + fill).clip(0.0, 1.0)[:, :, None]
        blended = frame_roi * (1.0 - total_a) + fabric_bgr * total_a
        result[y1:y2, x1:x2] = np.clip(blended, 0, 255).astype(np.uint8)

        # ── 6b. Polygon-fill OUTSIDE the shirt bbox (sleeve overflow) ─────
        # When the polygon extends past the warped shirt's rectangle,
        # render the sampled fabric color in that overflow area.
        if shirt_region is not None and self.polygon_fill_strength > 0:
            self._render_polygon_fill_outside_bbox(
                result, frame, shirt_region,
                avg_fabric_bgr=avg_fabric_bgr,
                bbox=(x1, y1, x2, y2), opacity=opacity,
            )

        # ── 6. Re-composite foreground elements with feathering ───────────
        foreground = occlusion_masks.get("foreground")
        if foreground is not None:
            foreground = self._ensure_mask(foreground, h, w)
            fg_roi = foreground[y1:y2, x1:x2]
            # Skip the entire foreground pass when no fg pixels land in the
            # shirt ROI — avoids 3 cv2 ops on an empty mask.
            if int(np.count_nonzero(fg_roi)) == 0:
                return result
            fg_soft = cv2.GaussianBlur(fg_roi.astype(np.float32) / 255.0, (5, 5), 0)
            
            fg_3 = fg_soft[:, :, None]
            orig_roi = orig_roi_bytes.astype(np.float32)
            current_roi = result[y1:y2, x1:x2].astype(np.float32)
            final_roi = current_roi * (1.0 - fg_3) + orig_roi * fg_3
            result[y1:y2, x1:x2] = np.clip(final_roi, 0, 255).astype(np.uint8)

        return result

    def _sample_fabric_color(self, shirt_roi: np.ndarray) -> np.ndarray:
        """Sample the cloth's INTERIOR mean BGR.

        Erodes the opaque mask before averaging so the warp's black border
        pixels (cv2.remap borderValue=(0,0,0,0)) aren't sampled. Returns
        a (3,) float32 BGR vector; defaults to a neutral mid-gray when
        the cloth has no detectably-opaque interior (rare).
        """
        if shirt_roi.ndim != 3 or shirt_roi.shape[2] < 4:
            # No alpha channel: just average everything (pre-warp PNGs).
            return np.mean(shirt_roi.reshape(-1, 3).astype(np.float32), axis=0)
        opaque = (shirt_roi[:, :, 3] > 220).astype(np.uint8) * 255
        # Erode so we only sample interior pixels — the warp's edge band
        # contains semi-transparent / black-bordered pixels that would
        # darken the average and produce visible black sleeves.
        if opaque.shape[0] > 12 and opaque.shape[1] > 12:
            opaque = cv2.erode(
                opaque,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
                iterations=1,
            )
        if cv2.countNonZero(opaque) < 50:
            return np.array([128.0, 128.0, 128.0], dtype=np.float32)
        sampled = shirt_roi[opaque > 0, :3].astype(np.float32)
        return np.mean(sampled, axis=0)

    def _render_polygon_fill_outside_bbox(
        self,
        result: np.ndarray,
        original_frame: np.ndarray,
        shirt_region: np.ndarray,
        avg_fabric_bgr: np.ndarray,
        bbox: Tuple[int, int, int, int],
        opacity: float,
    ) -> None:
        """Fill polygon area that lies OUTSIDE the warped shirt's bbox
        (sleeve overflow) with the precomputed fabric color.

        Uses `avg_fabric_bgr` (sampled from cloth interior) so we never
        bleed the warp's black border pixels into the sleeve extension.
        """
        bx1, by1, bx2, by2 = bbox

        outside = shirt_region.copy()
        outside[by1:by2, bx1:bx2] = 0
        ys, xs = np.where(outside > 0)
        if xs.size == 0:
            return
        ox1 = int(xs.min()); ox2 = int(xs.max()) + 1
        oy1 = int(ys.min()); oy2 = int(ys.max()) + 1

        sub_region = outside[oy1:oy2, ox1:ox2].astype(np.float32) / 255.0
        sub_alpha = cv2.GaussianBlur(sub_region, (9, 9), 0)
        sub_alpha *= float(self.polygon_fill_strength) * float(opacity)

        sub_frame = original_frame[oy1:oy2, ox1:ox2].astype(np.float32)
        a3 = sub_alpha[:, :, None]
        sub_blend = sub_frame * (1.0 - a3) + avg_fabric_bgr.reshape(1, 1, 3) * a3
        result[oy1:oy2, ox1:ox2] = np.clip(sub_blend, 0, 255).astype(np.uint8)


__all__ = ["OcclusionEngine"]