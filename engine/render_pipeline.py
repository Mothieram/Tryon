"""
render_pipeline.py - Master Render Pipeline  [FIXED]
Orchestrates all engines to produce final try-on frames.

FIXES (2026-04-24):
- Imports corrected: engine.coreutils throughout (no more engine.utils mismatch)
- Shirt lighting now actually applied (was a dead copy)
- Shadow block enabled behind flag (was `if False`)
- _process_frame_inner now correctly passes frame shape as tuple
- GarmentMeta defaults tuned for real shirt PNGs (collar at ~8–12% from top)
"""

import cv2
import numpy as np
import logging
import time
import threading
from typing import Optional, List, Dict, Tuple, Any
from pathlib import Path
from dataclasses import dataclass, field

from engine.coreutils import (
    setup_logger, PoseKeypoints, Keypoint, GarmentMeta,
    FPSCounter, FrameCache, ensure_bgra, create_placeholder_shirt,
)
from engine.mediapipe_holistic_pose import MediaPipeHolisticPoseEngine, AsyncPoseEngine
from engine.densepose_engine import DensePoseEngine, TorsoMap
from engine.parsing_engine import ParsingEngine, ParsedRegions
from engine.garment_landmarks import GarmentAnalyzer, GarmentLandmarks
from engine.hybrid_warper import HybridWarper
from engine.shadow_engine import ShadowEngine
from engine.occlusion_engine import OcclusionEngine
from engine.cloth_preprocessor import ClothPreprocessor
from engine.sam2_engine import SAM2Engine

logger = setup_logger("pipeline")


@dataclass
class PipelineStats:
    fps: float = 0.0
    pose_ms: float = 0.0
    warp_ms: float = 0.0
    render_ms: float = 0.0
    total_ms: float = 0.0
    pose_detected: bool = False
    active_shirt: str = ""
    engine_method: str = ""
    gpu_active: bool = False


@dataclass
class GarmentEntry:
    path: str
    name: str
    image: np.ndarray           # BGRA (dehaloed, centered into target_size)
    meta: GarmentMeta
    landmarks: Optional[GarmentLandmarks] = None
    thumbnail: Optional[np.ndarray] = None
    # Optional outputs from the offline ClothPreprocessor.
    mask: Optional[np.ndarray] = None             # uint8 binary cloth mask
    edges: Optional[np.ndarray] = None            # contour map
    control_points: Optional[np.ndarray] = None   # (N, 2) grid points
    content_bbox: Optional[Tuple[int, int, int, int]] = None


class RenderPipeline:
    """
    Master Virtual Try-On Render Pipeline.
    """

    def __init__(
        self,
        pose_model: Optional[str] = None,
        parsing_model: Optional[str] = None,
        device: str = "auto",
        target_fps: int = 30,
        enable_shadows: bool = True,
        enable_lighting: bool = True,
        opacity: float = 0.95,
    ):
        self.target_fps = target_fps
        self.enable_shadows = enable_shadows
        self.enable_lighting = enable_lighting
        self.opacity = opacity
        self.device = device

        self._pose_engine = MediaPipeHolisticPoseEngine(
            keypoint_conf=0.35,
            smooth_alpha=0.4,
        )
        self._async_pose = AsyncPoseEngine(self._pose_engine)
        self._densepose = DensePoseEngine(use_densepose=True)
        self._parsing = ParsingEngine(model_path=parsing_model, device=device)
        # SAM2 = pixel-precise body silhouette via HuggingFace transformers.
        # Loaded lazily; pipeline degrades gracefully to DensePose if SAM2
        # isn't installed or the model fails to download.
        self._sam2 = SAM2Engine(device=device)
        self._garment_analyzer = GarmentAnalyzer()
        # Offline cloth preprocessor: each shirt PNG is processed once into
        # a clean dehaloed asset with a mask, edges, and grid control
        # points. Cache lives in `assets/processed_shirts/<name>/`. The
        # runtime path (`HybridWarper`, `OcclusionEngine`) is unchanged.
        self._cloth_preprocessor = ClothPreprocessor()
        self._warper = HybridWarper(
            smooth_alpha=0.35,
            physics_lag=0.2,
            tps_smooth=0.1,           # kept for API compat (unused in V3)
            flow_pyramid_levels=3,
            flow_smooth_sigma=26.0,   # stiffer field — kills the "floating water" deformation
            device=device,
        )
        self._shadow = ShadowEngine(shadow_intensity=0.22)
        self._occlusion = OcclusionEngine(
            feather_radius=8,
            trust_parser_for_body=False,
            trust_parser_for_foreground=True,
            trust_densepose=True,
        )

        # Occlusion masks change slowly with pose; recomputing every frame
        # is wasteful. Cache for N frames matched to the parser cadence.
        self._occlusion_cache: Optional[Dict[str, np.ndarray]] = None
        self._occlusion_cache_frame: int = -100
        # Fit torso mask only changes when parser/densepose update, plus
        # slow pose drift. Cache it on the same cadence.
        self._fit_torso_cache: Optional[np.ndarray] = None
        self._fit_torso_cache_frame: int = -100
        # SAM2 body silhouette cache. Same cadence as parser/densepose
        # since the pose change between cadence ticks is tiny.
        self._sam2_cache: Optional[np.ndarray] = None
        self._sam2_cache_frame: int = -100
        # Phase 1: DensePose body-part clip mask (torso + arms). Used as a
        # hard alpha gate on the warped shirt to kill the sideways sleeve
        # flaps the 2D flow warper produces. SAM2 silhouette is the
        # fallback when DensePose part labels aren't usable.
        self._body_clip_cache: Optional[np.ndarray] = None
        self._body_clip_cache_frame: int = -100

        # Telemetry: log per-stage timings once per second so production
        # users can diagnose lag without a debugger. Aggregates the avg /
        # p95 over the window between log flushes.
        self._telemetry_window: List[Dict[str, float]] = []
        self._telemetry_last_log: float = time.perf_counter()

        self._garments: List[GarmentEntry] = []
        self._current_idx: int = 0

        self._fps_counter = FPSCounter(window=30)
        self._frame_cache = FrameCache(change_threshold=4.0)
        self._last_pose: Optional[PoseKeypoints] = None
        self._last_parsing = None
        self._last_torso = None
        # Parser/densepose cadence. They feed body silhouette refinement
        # which is cached, so 6 frames between updates is visually invisible
        # while saving significant CPU/GPU.
        self._parse_frame_skip = 6
        self._frame_count = 0
        self._debug_overlays: Dict[str, bool] = {
            "holistic": False,
            "parser": False,
            "densepose": False,
            "detectron": False,
        }

        self.stats = PipelineStats()
        self._models_loaded = False
        self._configure_torch_runtime()
        logger.info("RenderPipeline initialized")

    # ── Model Loading ────────────────────────────────────────────────────────

    def load_models(self) -> bool:
        logger.info("Loading AI models...")
        flow = getattr(self._warper, "_flow_warper", None)
        if flow is not None:
            logger.info(
                "Geometric flow warper ready | pyramid_levels=%s | smooth_sigma=%.2f | max_flow_dim=%s",
                getattr(flow, "pyramid_levels", "n/a"),
                float(getattr(flow, "smooth_sigma", 0.0)),
                getattr(flow, "_MAX_FLOW_DIM", "n/a"),
            )
        success = self._pose_engine.load()
        if not success:
            logger.error("MediaPipe Holistic model failed to load!")
            return False
        # Default runtime profile. FP16 on GPU roughly halves pose-detect cost.
        try:
            import torch as _torch
            _gpu = _torch.cuda.is_available()
        except Exception:
            _gpu = False
        self._pose_engine.configure_runtime(
            imgsz=640,
            use_half=bool(_gpu),
        )
        if hasattr(self._parsing, "set_input_size"):
            # 384 instead of 512 cuts parsing inference ~40% with little
            # visible quality loss — and it's only used as a soft refinement
            # of the pose-based body silhouette now.
            self._parsing.set_input_size(384)
        self._async_pose.start()
        self._models_loaded = True
        logger.info("Models loaded successfully (MediaPipe Holistic + geometric flow warper active)")
        self._log_runtime_state()
        return True

    def _log_runtime_state(self) -> None:
        """One-shot startup log so you can see, in the terminal, exactly
        which signals are driving the shirt fit. Useful for verifying that
        DensePose and the cloth preprocessor are actually wired in."""
        densepose_real = bool(getattr(self._densepose, "has_densepose", False))
        densepose_label = (
            "Detectron2 DensePose (real)" if densepose_real
            else "DensePose (parser-mask fallback)"
        )
        sam2_status = self._sam2.get_status() if self._sam2 else {}
        sam2_loaded = bool(sam2_status.get("loaded"))
        sam2_label = (
            f"SAM2 ({sam2_status.get('backend')}/{sam2_status.get('model_id')})"
            if sam2_loaded
            else f"SAM2 NOT LOADED ({sam2_status.get('error') or 'init skipped'})"
        )
        occ = self._occlusion
        logger.info("─" * 60)
        logger.info("Runtime occlusion config:")
        logger.info("  body fit  | %s  trust_sam2=%s",
                    sam2_label, getattr(occ, "trust_sam2", False))
        logger.info("  body fit  | DensePose=%s  trust_densepose=%s",
                    densepose_label, getattr(occ, "trust_densepose", False))
        logger.info("  body fit  | trust_parser_for_body=%s (SCHP refinement, off by default)",
                    getattr(occ, "trust_parser_for_body", False))
        logger.info("  fg occl   | trust_parser_for_foreground=%s (parser arms/head)",
                    getattr(occ, "trust_parser_for_foreground", False))
        logger.info("  preprocessor cache root: %s",
                    getattr(self._cloth_preprocessor, "cache_root", "?"))
        logger.info("  parse_frame_skip=%d  (parser/densepose runs every N frames)",
                    self._parse_frame_skip)
        logger.info("  NOTE: the 'DensePose' debug overlay button is a "
                    "VIEWER ONLY — DensePose is computed every frame regardless.")
        logger.info("─" * 60)

    def unload_models(self):
        self._async_pose.stop()

    # ── Garment Catalog ──────────────────────────────────────────────────────

    def load_garments(self, shirts_dir: str) -> int:
        self._garments.clear()
        shirts_path = Path(shirts_dir)
        shirts_path.mkdir(parents=True, exist_ok=True)

        png_files = sorted(shirts_path.glob("*.png"))
        logger.info(f"Found {len(png_files)} shirt(s) in {shirts_dir}")
        for png_path in png_files:
            self._load_single_garment(str(png_path))

        if not self._garments:
            logger.info("No shirts found - generating placeholder shirts")
            self._generate_placeholder_shirts(shirts_path)

        if self._garments:
            self._preanalyze_garments()

        logger.info(f"Loaded {len(self._garments)} garment(s)")
        return len(self._garments)

    def _load_single_garment(self, path: str) -> bool:
        try:
            name = Path(path).stem

            # Offline preprocessor: dehalo + center + mask + grid points.
            # Cached on disk; first call is ~30-100ms, subsequent runs
            # hit the cache and load in ~5ms.
            try:
                cache_hit = self._cloth_preprocessor.is_cached(path)
                processed = self._cloth_preprocessor.process(path)
                img = processed.image_bgra
                thumb = processed.thumbnail
                logger.info(
                    "garment %-12s | cache=%-3s | shape=%s | bbox=%s",
                    name, "HIT" if cache_hit else "MISS",
                    img.shape, processed.content_bbox,
                )
            except Exception as exc:
                # Fallback to raw load if preprocessor fails — keeps the
                # app usable even with a malformed PNG or write-locked
                # cache directory.
                logger.warning(
                    "Cloth preprocessor failed for %s (%s); using raw load.",
                    path, exc,
                )
                raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                if raw is None:
                    logger.warning("Could not load: %s", path)
                    return False
                img = ensure_bgra(raw)
                thumb = cv2.resize(img, (80, 100))
                processed = None

            # Default landmark ratios — analyser refines from alpha.
            meta = GarmentMeta(
                path=path,
                name=name,
                collar_y_ratio=0.08,
                shoulder_y_ratio=0.18,
                sleeve_end_ratio=0.52,
                hem_y_ratio=0.96,
                neck_x_ratio=0.50,
            )

            entry = GarmentEntry(
                path=path, name=name, image=img, meta=meta, thumbnail=thumb,
            )
            if processed is not None:
                entry.mask = processed.mask
                entry.edges = processed.edges
                entry.control_points = processed.control_points
                entry.content_bbox = processed.content_bbox
            self._garments.append(entry)
            logger.debug(f"Loaded shirt: {name}")
            return True
        except Exception as e:
            logger.error(f"Error loading {path}: {e}")
            return False

    def _generate_placeholder_shirts(self, shirts_path: Path):
        colors = [
            ((30, 80, 180), "Blue_Formal"),
            ((20, 120, 50), "Green_Casual"),
            ((180, 50, 30), "Red_Sport"),
            ((120, 30, 120), "Purple_Fashion"),
            ((30, 120, 150), "Teal_Business"),
        ]
        for color, name in colors:
            shirt_img = create_placeholder_shirt(size=(400, 500), color=color)
            save_path = shirts_path / f"{name}.png"
            cv2.imwrite(str(save_path), shirt_img)
            logger.info(f"Created placeholder: {name}.png")
            self._load_single_garment(str(save_path))

    def _preanalyze_garments(self):
        for entry in self._garments:
            try:
                landmarks = self._garment_analyzer.analyze(
                    entry.image, entry.meta, cache_key=entry.path,
                )
                entry.landmarks = landmarks
            except Exception as e:
                logger.error(f"Landmark analysis failed for {entry.name}: {e}")

    def add_garment(self, path: str) -> bool:
        result = self._load_single_garment(path)
        if result and self._garments:
            entry = self._garments[-1]
            try:
                entry.landmarks = self._garment_analyzer.analyze(
                    entry.image, entry.meta, cache_key=path
                )
            except Exception as e:
                logger.error(f"Failed to analyze new garment: {e}")
        return result

    # ── Shirt Navigation ─────────────────────────────────────────────────────

    @property
    def current_garment(self) -> Optional[GarmentEntry]:
        if not self._garments:
            return None
        return self._garments[self._current_idx]

    @property
    def garment_count(self) -> int:
        return len(self._garments)

    def next_shirt(self):
        if self._garments:
            self._current_idx = (self._current_idx + 1) % len(self._garments)
            self._warper.reset()
            logger.info(f"Shirt: {self.current_garment.name}")

    def previous_shirt(self):
        if self._garments:
            self._current_idx = (self._current_idx - 1) % len(self._garments)
            self._warper.reset()
            logger.info(f"Shirt: {self.current_garment.name}")

    def select_shirt(self, idx: int):
        if 0 <= idx < len(self._garments):
            self._current_idx = idx
            self._warper.reset()

    def get_shirt_names(self) -> List[str]:
        return [g.name for g in self._garments]

    # ── Main Processing ──────────────────────────────────────────────────────

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, PipelineStats]:
        """Run the full try-on pipeline on one camera frame at native resolution."""
        import traceback as _tb
        t_total = time.perf_counter()
        self._frame_count += 1
        try:
            return self._process_frame_inner(frame, frame.copy(), t_total)
        except Exception:
            logger.error("Pipeline error:\n%s", _tb.format_exc())
            self.stats.fps = self._fps_counter.tick()
            return frame, self.stats

    def _process_frame_inner(
        self, frame: np.ndarray, result: np.ndarray, t_total: float
    ) -> Tuple[np.ndarray, PipelineStats]:
        if not self._models_loaded or not self._garments:
            self.stats.fps = self._fps_counter.tick()
            return self._no_detection_overlay(result), self.stats

        garment = self.current_garment
        if garment is None or garment.landmarks is None:
            self.stats.fps = self._fps_counter.tick()
            return result, self.stats

        # ── Pose Detection ────────────────────────────────────────────────────
        t_pose = time.perf_counter()
        self._async_pose.submit_frame(frame)
        pose = self._async_pose.get_latest_pose()
        self.stats.pose_ms = (time.perf_counter() - t_pose) * 1000
        self.stats.pose_detected = pose is not None and pose.is_usable()
        self._last_pose = pose

        if not pose or not pose.is_usable():
            self.stats.fps = self._fps_counter.tick()
            return self._no_detection_overlay(result), self.stats

        # ── Parsing (throttled) ───────────────────────────────────────────────
        h, w = frame.shape[:2]
        cache_invalid = (
            self._last_parsing is None
            or self._last_torso is None
            or getattr(self._last_parsing, "torso",
                       np.zeros((0, 0), dtype=np.uint8)).shape != (h, w)
            or getattr(self._last_torso, "torso_mask",
                       np.zeros((0, 0), dtype=np.uint8)).shape != (h, w)
        )
        if cache_invalid or self._frame_count % self._parse_frame_skip == 0:
            roi = self._compute_pose_roi(frame.shape[:2], pose, pad_ratio=0.12)
            x1, y1, x2, y2 = roi
            roi_frame = frame[y1:y2, x1:x2]
            roi_pose = self._translate_pose(pose, dx=-x1, dy=-y1)

            parsed_roi = self._parsing.parse(roi_frame, roi_pose)
            torso_roi = self._densepose.estimate(
                roi_frame, roi_pose, parsing_mask=parsed_roi,
            )

            self._last_parsing = self._project_parsed_to_frame(
                parsed_roi, frame_shape=frame.shape[:2], roi=roi
            )
            self._last_torso = self._project_torso_to_frame(
                torso_roi, frame_shape=frame.shape[:2], roi=roi
            )

        parsed = self._last_parsing
        torso_map = self._last_torso

        # ── SAM2 body silhouette (cached at parser cadence) ─────────────
        rebuild_sam2 = (
            self._sam2_cache is None
            or self._sam2_cache.shape != frame.shape[:2]
            or (self._frame_count - self._sam2_cache_frame) >= self._parse_frame_skip
        )
        if rebuild_sam2 and self._sam2.available:
            sam2_mask = self._sam2.segment_body(frame, pose)
            if sam2_mask is not None:
                self._sam2_cache = sam2_mask
                self._sam2_cache_frame = self._frame_count
        sam2_body_mask = self._sam2_cache

        rebuild_fit = (
            self._fit_torso_cache is None
            or self._fit_torso_cache.shape != frame.shape[:2]
            or (self._frame_count - self._fit_torso_cache_frame) >= self._parse_frame_skip
        )
        if rebuild_fit:
            self._fit_torso_cache = self._build_fit_torso_mask(
                frame.shape[:2], parsed, torso_map, pose,
                sam2_mask=sam2_body_mask,
            )
            self._fit_torso_cache_frame = self._frame_count
        fit_torso_mask = self._fit_torso_cache

        # ── Warp Shirt ────────────────────────────────────────────────────────
        t_warp = time.perf_counter()
        warp_result = self._warper.warp(
            garment.image,
            garment.landmarks,
            pose,
            frame.shape,           # (h, w, c) — warper uses [:2]
            torso_mask=fit_torso_mask,
            frame=frame,
        )
        self.stats.warp_ms = (time.perf_counter() - t_warp) * 1000

        if warp_result is None:
            self.stats.fps = self._fps_counter.tick()
            return result, self.stats

        # ── Lighting Adaptation ───────────────────────────────────────────────
        t_render = time.perf_counter()
        warped_shirt = warp_result.warped_shirt

        if self.enable_lighting:
            warped_shirt = self._shadow.adapt_shirt_lighting(
                warped_shirt, frame,
                warp_result.placement_x,
                warp_result.placement_y,
            )

        # ── Occlusion Masks (cached for `_parse_frame_skip` frames) ─────────
        rebuild = (
            self._occlusion_cache is None
            or (self._frame_count - self._occlusion_cache_frame) >= self._parse_frame_skip
        )
        if rebuild:
            self._occlusion_cache = self._occlusion.build_occlusion_masks(
                frame, pose, parsed, torso_map, sam2_mask=sam2_body_mask,
            )
            self._occlusion_cache_frame = self._frame_count
        occlusion_masks = self._occlusion_cache

        # ── Phase 1: DensePose body-part hard clip ────────────────────────────
        # Use DensePose torso+arm part labels (with SAM2 fallback) as a
        # ceiling on where the warped shirt is allowed to render. Kills the
        # 2D-warp's sideways sleeve overshoot and hem bleed onto the legs
        # without touching the warp itself.
        rebuild_body_clip = (
            self._body_clip_cache is None
            or self._body_clip_cache[0].shape != frame.shape[:2]
            or (self._frame_count - self._body_clip_cache_frame) >= self._parse_frame_skip
        )
        if rebuild_body_clip:
            self._body_clip_cache = self._build_densepose_body_clip(
                frame.shape[:2], torso_map, sam2_body_mask, pose=pose,
            )
            self._body_clip_cache_frame = self._frame_count
        body_clip = self._body_clip_cache

        if body_clip is not None:
            full_body, torso_only = body_clip
            # Cloth alpha gets the FULL body (torso + arms). The cloth PNG
            # has its own natural sleeve geometry, so allowing it to render
            # over the arm region only matters where the warped sleeve
            # actually has alpha — outside the cloth's own silhouette,
            # alpha is already zero.
            warped_shirt = self._clip_shirt_alpha_to_body(
                warped_shirt, full_body,
                warp_result.placement_x, warp_result.placement_y,
            )
            # Polygon shirt_region (used by the composite's avg-fabric
            # overflow fill) is clipped to TORSO ONLY. Without this, the
            # 0.95 fill_strength painted big rectangular fabric blobs
            # along the arms in T-pose. Shallow-copy the cached dict so
            # we don't mutate it across frames.
            sr = occlusion_masks.get("shirt_region") if occlusion_masks else None
            if sr is not None:
                sr_clipped = cv2.bitwise_and(sr, torso_only)
                occlusion_masks = dict(occlusion_masks)
                occlusion_masks["shirt_region"] = sr_clipped

        # ── Composite (shirt clipped to body, arms/head re-drawn on top) ──────
        result = self._occlusion.composite(
            frame=result,
            warped_shirt=warped_shirt,
            placement_x=warp_result.placement_x,
            placement_y=warp_result.placement_y,
            occlusion_masks=occlusion_masks,
            opacity=float(np.clip(self.opacity, 0.1, 1.0)),
        )

        # ── Shadows (use the body-clipped shirt-region mask, not the raw alpha) ──
        if self.enable_shadows:
            shirt_region_mask = occlusion_masks.get("shirt_region")
            if shirt_region_mask is None:
                shirt_region_mask = self._build_shirt_region_mask(
                    frame_shape=frame.shape[:2],
                    shirt=warped_shirt,
                    x=warp_result.placement_x,
                    y=warp_result.placement_y,
                )
            result = self._shadow.apply_shadows(
                result, shirt_region_mask, pose, warped_shirt
            )

        self.stats.render_ms = (time.perf_counter() - t_render) * 1000
        self.stats.total_ms  = (time.perf_counter() - t_total)  * 1000
        self.stats.fps        = self._fps_counter.tick()
        self.stats.gpu_active = False
        self.stats.active_shirt  = garment.name
        self.stats.engine_method = parsed.method

        self._record_telemetry()
        result = self._apply_debug_overlays(result, pose, parsed, torso_map)
        return result, self.stats

    # ── Debug / Helpers ──────────────────────────────────────────────────────

    def _build_shirt_region_mask(
        self,
        frame_shape: Tuple[int, int],
        shirt: np.ndarray,
        x: int,
        y: int,
    ) -> np.ndarray:
        h, w = frame_shape
        mask = np.zeros((h, w), dtype=np.uint8)
        if shirt is None or shirt.size == 0:
            return mask

        sh, sw = shirt.shape[:2]
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(w, x + sw), min(h, y + sh)
        if x2 <= x1 or y2 <= y1:
            return mask

        sx1 = x1 - x
        sy1 = y1 - y
        sx2 = sx1 + (x2 - x1)
        sy2 = sy1 + (y2 - y1)

        if shirt.shape[2] == 4:
            shirt_alpha = shirt[sy1:sy2, sx1:sx2, 3]
            mask[y1:y2, x1:x2] = np.maximum(mask[y1:y2, x1:x2], shirt_alpha)
        else:
            mask[y1:y2, x1:x2] = 255

        return mask

    def _build_fit_torso_mask(
        self,
        frame_shape: Tuple[int, int],
        parsed: Any,
        torso_map: Any,
        pose: Optional[PoseKeypoints] = None,
        sam2_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        h, w = frame_shape
        fit = np.zeros((h, w), dtype=np.uint8)

        # SAM2 mask (pixel-precise) takes precedence when available.
        if sam2_mask is not None:
            s = np.asarray(sam2_mask)
            if s.ndim == 3:
                s = s[:, :, 0]
            if s.shape != (h, w):
                s = cv2.resize(s, (w, h), interpolation=cv2.INTER_NEAREST)
            fit = np.maximum(fit, s.astype(np.uint8))

        if torso_map is not None:
            tm = getattr(torso_map, "torso_mask", None)
            if tm is not None:
                t = np.asarray(tm)
                if t.ndim == 3:
                    t = t[:, :, 0]
                if t.shape != (h, w):
                    t = cv2.resize(t, (w, h), interpolation=cv2.INTER_NEAREST)
                fit = np.maximum(fit, t.astype(np.uint8))

        if parsed is not None:
            pm = getattr(parsed, "torso", None)
            if pm is not None:
                p = np.asarray(pm)
                if p.ndim == 3:
                    p = p[:, :, 0]
                if p.shape != (h, w):
                    p = cv2.resize(p, (w, h), interpolation=cv2.INTER_NEAREST)
                fit = np.maximum(fit, p.astype(np.uint8))

        # NOTE: previously this took the largest contour and filled its
        # convex hull. For a person with arms hanging at the sides, that
        # hull bridges arm-to-arm and inflates the silhouette to bare-arm
        # width. The warper then samples that silhouette and yanks the
        # shirt's chest/waist/hem control points outward — producing the
        # squeezed, pulled cloth artefact. The pose-ROI polygon clip below
        # already keeps the mask clean without the hull.

        # Constrain fit mask using live pose torso ROI so TPS doesn't overfit to leaked regions.
        if pose is not None:
            ls = pose.left_shoulder
            rs = pose.right_shoulder
            lh = pose.left_hip
            rh = pose.right_hip
            if ls and rs and ls.valid and rs.valid:
                sw = max(20.0, float(pose.shoulder_width))
                top_y = int(max(0, min(ls.y, rs.y) - sw * 0.20))
                if lh and rh and lh.valid and rh.valid:
                    bottom_y = int(min(h - 1, max(lh.y, rh.y) + sw * 0.25))
                    center_x = int((ls.x + rs.x + lh.x + rh.x) / 4.0)
                    half_top = int(sw * 0.75)
                    half_bottom = int(sw * 0.90)
                else:
                    bottom_y = int(min(h - 1, max(ls.y, rs.y) + sw * 1.65))
                    center_x = int((ls.x + rs.x) / 2.0)
                    half_top = int(sw * 0.78)
                    half_bottom = int(sw * 0.85)

                roi_poly = np.array([
                    [center_x - half_top, top_y],
                    [center_x + half_top, top_y],
                    [center_x + half_bottom, bottom_y],
                    [center_x - half_bottom, bottom_y],
                ], dtype=np.int32)
                roi_poly[:, 0] = np.clip(roi_poly[:, 0], 0, w - 1)
                roi_poly[:, 1] = np.clip(roi_poly[:, 1], 0, h - 1)
                roi_mask = np.zeros_like(fit)
                cv2.fillConvexPoly(roi_mask, roi_poly, 255)
                fit = cv2.bitwise_and(fit, roi_mask)

        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        fit = cv2.morphologyEx(fit, cv2.MORPH_CLOSE, kernel_close)
        fit = cv2.morphologyEx(fit, cv2.MORPH_OPEN, kernel_open)
        return fit

    def _build_densepose_body_clip(
        self,
        frame_shape: Tuple[int, int],
        torso_map: Any,
        sam2_mask: Optional[np.ndarray],
        pose: Optional[PoseKeypoints] = None,
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Phase 1: build (full_body, torso_only) clip masks.

        ``full_body`` is the union of DensePose torso + both arm part
        labels (with SAM2 silhouette as fallback). It's the ceiling for
        the warped shirt's alpha — sleeves drawn from the cloth PNG can
        render anywhere this mask is positive.

        ``torso_only`` is the torso part label by itself. It's used to
        clip the polygon shirt_region, so the composite's polygon-fill
        path (avg fabric color where the cloth has no alpha) stays on the
        chest/belly and never blobs out along an outstretched arm.

        Returns None when neither DensePose nor SAM2 produces a usable
        mask, in which case the caller skips clipping entirely.
        """
        h, w = frame_shape

        torso_only = None
        full_body = None

        if torso_map is not None:
            tm = getattr(torso_map, "torso_mask", None)
            arm_masks = getattr(torso_map, "arm_masks", None) or {}
            la = arm_masks.get("left_arm")
            ra = arm_masks.get("right_arm")

            def _norm(m):
                if m is None:
                    return None
                a = np.asarray(m)
                if a.ndim == 3:
                    a = a[:, :, 0]
                if a.shape != (h, w):
                    a = cv2.resize(a, (w, h), interpolation=cv2.INTER_NEAREST)
                return a.astype(np.uint8)

            t_n = _norm(tm)
            la_n = _norm(la)
            ra_n = _norm(ra)

            if t_n is not None and int(np.count_nonzero(t_n)) >= 200:
                torso_only = t_n
                full_body = t_n.copy()
                for arm in (la_n, ra_n):
                    if arm is not None:
                        full_body = np.maximum(full_body, arm)

        # SAM2 fallback: full body only (no torso/arm separation), so we
        # use the same silhouette for both. The polygon-fill blob risk
        # exists with SAM2 too, but is bounded by the pose-polygon ROI
        # already applied upstream in _build_fit_torso_mask.
        if full_body is None and sam2_mask is not None:
            s = np.asarray(sam2_mask)
            if s.ndim == 3:
                s = s[:, :, 0]
            if s.shape != (h, w):
                s = cv2.resize(s, (w, h), interpolation=cv2.INTER_NEAREST)
            full_body = s.astype(np.uint8)
            torso_only = full_body

        if full_body is None or torso_only is None:
            return None

        # Pose-driven shoulder→elbow→wrist corridor. DensePose often fails
        # to label arms on a shirtless user (no clothing edge to anchor
        # the part), and SAM2 silhouettes leave thin gaps along an
        # outstretched arm that the body-clip kernel can't bridge. The
        # corridor is a thick polygon along each arm bone — unioning it
        # into full_body guarantees the cloth's sleeve PNG can render
        # over the arm region.
        if pose is not None:
            corridor = self._build_arm_corridor_mask(frame_shape, pose)
            if corridor is not None:
                full_body = np.maximum(full_body, corridor)

        # Soft dilate the full-body mask so part-label boundaries don't
        # carve thin holes at the seams between torso/arm. The kernel was
        # 21 (10px each side) and not enough to bridge the gap between
        # torso and an outstretched arm. 41 (20px each side) reliably
        # closes the shoulder-seam gap in T-pose.
        k_full = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41))
        full_body = cv2.dilate(full_body, k_full, iterations=1)
        full_body = cv2.morphologyEx(full_body, cv2.MORPH_CLOSE, k_full)

        # Torso-only gets a slightly heavier dilation so the polygon hem
        # can extend ~1 inch past the hipline (natural shirt drape) but
        # NOT laterally onto the arms, since this mask intentionally
        # excludes them.
        k_torso = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
        torso_only = cv2.dilate(torso_only, k_torso, iterations=1)
        torso_only = cv2.morphologyEx(torso_only, cv2.MORPH_CLOSE, k_torso)

        return full_body, torso_only

    def _build_arm_corridor_mask(
        self,
        frame_shape: Tuple[int, int],
        pose: PoseKeypoints,
    ) -> Optional[np.ndarray]:
        """Thick polygon along each arm bone (shoulder→elbow→wrist).
        Used to plug the gap between DensePose / SAM2 arm coverage and
        the shirt's sleeve PNG, especially in T-pose where arm pixels
        are a thin horizontal strip easily lost to silhouette refinement.
        """
        h, w = frame_shape
        body_sw = max(20.0, float(getattr(pose, "shoulder_width", 0.0)))
        thickness = int(max(24.0, body_sw * 0.45))

        mask = np.zeros((h, w), dtype=np.uint8)
        any_drawn = False

        for shoulder, elbow, wrist in (
            (pose.left_shoulder, pose.left_elbow, pose.left_wrist),
            (pose.right_shoulder, pose.right_elbow, pose.right_wrist),
        ):
            if shoulder is None or not shoulder.valid:
                continue
            pts: List[Tuple[int, int]] = [(int(shoulder.x), int(shoulder.y))]
            if elbow is not None and elbow.valid:
                pts.append((int(elbow.x), int(elbow.y)))
            if wrist is not None and wrist.valid:
                pts.append((int(wrist.x), int(wrist.y)))
            if len(pts) < 2:
                continue
            for i in range(len(pts) - 1):
                cv2.line(mask, pts[i], pts[i + 1], 255, thickness=thickness, lineType=cv2.LINE_8)
            any_drawn = True

        if not any_drawn:
            return None
        return mask

    def _clip_shirt_alpha_to_body(
        self,
        warped_shirt: np.ndarray,
        body_clip: np.ndarray,
        placement_x: int,
        placement_y: int,
    ) -> np.ndarray:
        """Multiply the warped shirt's alpha channel by the body clip mask.

        Returns a new BGRA image; the original is not mutated. Pixels of
        the shirt that fall on the frame outside the body region get alpha
        zero, which the composite blends into transparent.
        """
        if warped_shirt is None or warped_shirt.size == 0:
            return warped_shirt
        if warped_shirt.shape[2] != 4:
            return warped_shirt

        h, w = body_clip.shape[:2]
        sh, sw = warped_shirt.shape[:2]

        # Frame-space rect of the shirt; clip to frame bounds.
        x1 = max(0, placement_x)
        y1 = max(0, placement_y)
        x2 = min(w, placement_x + sw)
        y2 = min(h, placement_y + sh)
        if x2 <= x1 or y2 <= y1:
            return warped_shirt

        sx1 = x1 - placement_x
        sy1 = y1 - placement_y
        sx2 = sx1 + (x2 - x1)
        sy2 = sy1 + (y2 - y1)

        # Build a per-pixel alpha-gain in shirt-local coords. Outside the
        # frame-overlap region the gain stays zero (shirt pixels there are
        # off-screen anyway, so the composite's bbox clip already drops
        # them — but keeping it consistent avoids surprises).
        gain = np.zeros((sh, sw), dtype=np.float32)
        body_roi = body_clip[y1:y2, x1:x2].astype(np.float32) / 255.0
        gain[sy1:sy2, sx1:sx2] = body_roi

        out = warped_shirt.copy()
        alpha = out[:, :, 3].astype(np.float32) * gain
        out[:, :, 3] = np.clip(alpha, 0, 255).astype(np.uint8)
        return out

    def _compute_pose_roi(
        self,
        frame_shape: Tuple[int, int],
        pose: Optional[PoseKeypoints],
        pad_ratio: float = 0.10,
    ) -> Tuple[int, int, int, int]:
        h, w = frame_shape
        if pose is None:
            return 0, 0, w, h

        pts = []
        for kp in getattr(pose, "keypoints", []) or []:
            if kp is not None and kp.valid:
                pts.append((float(kp.x), float(kp.y)))
        if len(pts) < 2:
            return 0, 0, w, h

        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)
        bw = max(1.0, x2 - x1)
        bh = max(1.0, y2 - y1)
        pad = max(16.0, max(bw, bh) * float(np.clip(pad_ratio, 0.05, 0.35)))

        rx1 = int(max(0, np.floor(x1 - pad)))
        ry1 = int(max(0, np.floor(y1 - pad)))
        rx2 = int(min(w, np.ceil(x2 + pad)))
        ry2 = int(min(h, np.ceil(y2 + pad)))

        min_side = 192
        if (rx2 - rx1) < min_side:
            cx = (rx1 + rx2) // 2
            half = min_side // 2
            rx1 = max(0, cx - half)
            rx2 = min(w, rx1 + min_side)
            rx1 = max(0, rx2 - min_side)
        if (ry2 - ry1) < min_side:
            cy = (ry1 + ry2) // 2
            half = min_side // 2
            ry1 = max(0, cy - half)
            ry2 = min(h, ry1 + min_side)
            ry1 = max(0, ry2 - min_side)

        return rx1, ry1, rx2, ry2

    def _translate_pose(self, pose: PoseKeypoints, dx: int, dy: int) -> PoseKeypoints:
        kps: List[Keypoint] = []
        for kp in pose.keypoints:
            if kp is None:
                kps.append(Keypoint(0.0, 0.0, 0.0))
            else:
                kps.append(Keypoint(x=kp.x + dx, y=kp.y + dy, confidence=kp.confidence))
        return PoseKeypoints(keypoints=kps, confidence=pose.confidence)

    def _project_mask_to_frame(
        self,
        mask: np.ndarray,
        frame_shape: Tuple[int, int],
        roi: Tuple[int, int, int, int],
    ) -> np.ndarray:
        h, w = frame_shape
        x1, y1, x2, y2 = roi
        out = np.zeros((h, w), dtype=np.uint8)
        if mask is None:
            return out
        m = np.asarray(mask)
        if m.ndim == 3:
            m = m[:, :, 0]
        th = max(1, y2 - y1)
        tw = max(1, x2 - x1)
        if m.shape != (th, tw):
            m = cv2.resize(m, (tw, th), interpolation=cv2.INTER_NEAREST)
        out[y1:y2, x1:x2] = m.astype(np.uint8)
        return out

    def _project_parsed_to_frame(
        self,
        parsed_roi: ParsedRegions,
        frame_shape: Tuple[int, int],
        roi: Tuple[int, int, int, int],
    ) -> ParsedRegions:
        h, w = frame_shape
        projected = ParsedRegions(h, w, method=getattr(parsed_roi, "method", "parsing"))
        projected.torso = self._project_mask_to_frame(parsed_roi.torso, frame_shape, roi)
        projected.left_arm = self._project_mask_to_frame(parsed_roi.left_arm, frame_shape, roi)
        projected.right_arm = self._project_mask_to_frame(parsed_roi.right_arm, frame_shape, roi)
        projected.face = self._project_mask_to_frame(parsed_roi.face, frame_shape, roi)
        projected.hair = self._project_mask_to_frame(parsed_roi.hair, frame_shape, roi)
        projected.legs = self._project_mask_to_frame(parsed_roi.legs, frame_shape, roi)
        return projected

    def _project_torso_to_frame(
        self,
        torso_roi: TorsoMap,
        frame_shape: Tuple[int, int],
        roi: Tuple[int, int, int, int],
    ) -> TorsoMap:
        torso_mask = self._project_mask_to_frame(torso_roi.torso_mask, frame_shape, roi)
        neck_mask = self._project_mask_to_frame(torso_roi.neck_mask, frame_shape, roi)
        arm_masks = {
            "left_arm": self._project_mask_to_frame(
                (torso_roi.arm_masks or {}).get("left_arm"), frame_shape, roi
            ),
            "right_arm": self._project_mask_to_frame(
                (torso_roi.arm_masks or {}).get("right_arm"), frame_shape, roi
            ),
        }
        return TorsoMap(
            torso_mask=torso_mask,
            arm_masks=arm_masks,
            neck_mask=neck_mask,
            uv_map=None,
            method=getattr(torso_roi, "method", "densepose"),
            confidence=float(getattr(torso_roi, "confidence", 1.0)),
        )

    def _overlay_mask(self, frame, mask, color, alpha=0.35):
        if mask is None:
            return frame
        m = np.asarray(mask)
        if m.ndim == 3:
            m = m[:, :, 0]
        if m.shape != frame.shape[:2]:
            m = cv2.resize(m, (frame.shape[1], frame.shape[0]),
                           interpolation=cv2.INTER_NEAREST)
        mask_bool = m > 10
        if not np.any(mask_bool):
            return frame
        out = frame.copy().astype(np.float32)
        color_arr = np.array(color, dtype=np.float32)
        out[mask_bool] = out[mask_bool] * (1.0 - alpha) + color_arr * alpha
        return np.clip(out, 0, 255).astype(np.uint8)

    def _apply_debug_overlays(self, frame, pose, parsed, torso_map):
        out = frame
        if self._debug_overlays.get("parser") and parsed is not None:
            out = self._overlay_mask(out, getattr(parsed, "torso", None),      (50, 170, 255), 0.28)
            out = self._overlay_mask(out, getattr(parsed, "left_arm", None),   (90, 230, 70),  0.32)
            out = self._overlay_mask(out, getattr(parsed, "right_arm", None),  (70, 220, 230), 0.32)
            out = self._overlay_mask(out, getattr(parsed, "face", None),       (40, 170, 255), 0.28)
            out = self._overlay_mask(out, getattr(parsed, "hair", None),       (170, 30, 170), 0.28)
            cv2.putText(out, "Parser", (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 220, 255), 2)

        if (self._debug_overlays.get("densepose") or
                self._debug_overlays.get("detectron")) and torso_map is not None:
            out = self._overlay_mask(out, getattr(torso_map, "torso_mask", None), (255, 120, 0),  0.24)
            arm_masks = getattr(torso_map, "arm_masks", {}) or {}
            out = self._overlay_mask(out, arm_masks.get("left_arm"),  (0, 255, 255), 0.24)
            out = self._overlay_mask(out, arm_masks.get("right_arm"), (0, 255, 0),   0.24)
            out = self._overlay_mask(out, getattr(torso_map, "neck_mask", None), (255, 0, 255), 0.24)
            label = "Detectron2 DensePose" if (
                self._debug_overlays.get("detectron") and self._densepose.has_densepose
            ) else "DensePose"
            cv2.putText(out, label, (10, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 190, 20), 2)

        if self._debug_overlays.get("holistic") and pose is not None:
            out = self._pose_engine.draw_skeleton(out, pose)

        return out

    def _no_detection_overlay(self, frame: np.ndarray) -> np.ndarray:
        result = frame.copy()
        h, w = frame.shape[:2]
        overlay = result.copy()
        cv2.rectangle(overlay, (w // 2 - 200, h - 60), (w // 2 + 200, h - 20), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, result, 0.5, 0, result)
        cv2.putText(result, "Stand in front of camera",
                    (w // 2 - 175, h - 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        return result

    # ── Public API ───────────────────────────────────────────────────────────

    def get_garment_thumbnails(self):
        return [g.thumbnail for g in self._garments]

    def get_model_status(self):
        return {
            "pose":   self._pose_engine.get_status(),
            "parsing": self._parsing.get_status(),
            "densepose": {"available": self._densepose.has_densepose},
            "garments_loaded": len(self._garments),
        }

    def toggle_shadows(self):   self.enable_shadows  = not self.enable_shadows
    def toggle_lighting(self):  self.enable_lighting = not self.enable_lighting
    def set_opacity(self, v):   self.opacity = float(np.clip(v, 0.1, 1.0))
    def set_parse_frame_skip(self, n): self._parse_frame_skip = int(np.clip(n, 1, 12))

    def set_trust_parser(self, enabled: bool):
        """Flip both granular flags together. Useful as a single 'parser
        mode' switch for the UI. For finer control use
        `set_trust_parser_for_foreground` and `set_trust_parser_for_body`."""
        self.set_trust_parser_for_body(enabled)
        self.set_trust_parser_for_foreground(enabled)

    def set_trust_parser_for_foreground(self, enabled: bool):
        """Use parser arms / face / hair to refine the in-front-of-shirt
        layer. Safety-clipped to outside the shirt's inner core, so
        parser-misclassified torso pixels can't be pasted on top of the
        shirt. Recommended ON for clothed and shirtless users alike."""
        if hasattr(self._occlusion, "trust_parser_for_foreground"):
            self._occlusion.trust_parser_for_foreground = bool(enabled)
            self._occlusion_cache = None

    def set_trust_parser_for_body(self, enabled: bool):
        """Allow parser to refine the shirt body region itself. Recommended
        ON only for users wearing a baseline shirt; OFF for shirtless users
        (SCHP would mis-label bare chest as arm/face and carve holes)."""
        if hasattr(self._occlusion, "trust_parser_for_body"):
            self._occlusion.trust_parser_for_body = bool(enabled)
            self._occlusion_cache = None
            self._fit_torso_cache = None

    def set_trust_densepose(self, enabled: bool):
        """Use DensePose torso silhouette as the body region. Recommended
        ON for both shirtless and clothed users — DensePose is reliable
        regardless of clothing. OFF falls back to the pure pose polygon
        (rectangular look)."""
        if hasattr(self._occlusion, "trust_densepose"):
            self._occlusion.trust_densepose = bool(enabled)
            self._occlusion_cache = None
            self._fit_torso_cache = None
            self._body_clip_cache = None

    def set_trust_sam2(self, enabled: bool):
        """Use SAM2's pixel-precise body silhouette as the primary body
        region. Recommended ON when SAM2 is loaded. OFF makes the
        pipeline fall back to DensePose / pose polygon."""
        if hasattr(self._occlusion, "trust_sam2"):
            self._occlusion.trust_sam2 = bool(enabled)
            self._occlusion_cache = None
            self._fit_torso_cache = None
            self._sam2_cache = None
            self._body_clip_cache = None

    def set_debug_overlay(self, name: str, enabled: bool):
        """Toggle a debug overlay on the rendered frame.

        VIEWER ONLY — these flags do not gate computation. Pose, parser,
        and DensePose all run every frame regardless of which overlays
        are visible. The overlays exist so you can verify that each
        subsystem is producing the silhouette / mask you expect.
        """
        key = str(name).strip().lower()
        if key in self._debug_overlays:
            self._debug_overlays[key] = bool(enabled)

    def get_debug_overlays(self):
        return dict(self._debug_overlays)

    def take_screenshot(self, frame: np.ndarray, output_dir: str = "screenshots") -> str:
        Path(output_dir).mkdir(exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        path = str(Path(output_dir) / f"tryon_{ts}.png")
        cv2.imwrite(path, frame)
        logger.info(f"Screenshot saved: {path}")
        return path

    def _record_telemetry(self) -> None:
        """Append per-frame stage timings; flush avg/p95 once per second."""
        self._telemetry_window.append({
            "pose":   float(self.stats.pose_ms),
            "warp":   float(self.stats.warp_ms),
            "render": float(self.stats.render_ms),
            "total":  float(self.stats.total_ms),
        })
        now = time.perf_counter()
        if (now - self._telemetry_last_log) < 1.0 or not self._telemetry_window:
            return
        n = len(self._telemetry_window)
        def stat(key: str) -> Tuple[float, float]:
            vals = sorted(s[key] for s in self._telemetry_window)
            avg = sum(vals) / max(1, n)
            p95 = vals[max(0, int(0.95 * (n - 1)))]
            return avg, p95
        pose_a, pose_p = stat("pose")
        warp_a, warp_p = stat("warp")
        rend_a, rend_p = stat("render")
        tot_a,  tot_p  = stat("total")
        logger.info(
            "perf | n=%d fps=%.1f | pose %.1f/%.1f | warp %.1f/%.1f | "
            "render %.1f/%.1f | total %.1f/%.1f ms (avg/p95) | skip=%d",
            n, float(self.stats.fps),
            pose_a, pose_p, warp_a, warp_p,
            rend_a, rend_p, tot_a, tot_p,
            int(self._parse_frame_skip),
        )
        self._telemetry_window.clear()
        self._telemetry_last_log = now

    def _configure_torch_runtime(self):
        try:
            import torch
            if torch.cuda.is_available():
                torch.backends.cudnn.benchmark = True
                if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
                    torch.backends.cuda.matmul.allow_tf32 = True
                if hasattr(torch.backends.cudnn, "allow_tf32"):
                    torch.backends.cudnn.allow_tf32 = True
                if hasattr(torch, "set_float32_matmul_precision"):
                    torch.set_float32_matmul_precision("high")
        except Exception:
            pass

__all__ = ["RenderPipeline", "PipelineStats", "GarmentEntry"]
