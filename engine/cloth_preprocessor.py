"""
cloth_preprocessor.py - Offline garment preprocessing for virtual try-on.

Each shirt PNG is processed ONCE into a clean cached asset. The runtime
pipeline (`RenderPipeline`, `HybridWarper`, `OcclusionEngine`) is unchanged
and reads the cached asset on garment load — no per-frame preprocessing.

Outputs (per shirt) live under `assets/processed_shirts/<name>/`:
    image.png             - dehaloed BGRA, centered into target_size
    mask.npy              - clean uint8 binary mask (0/255)
    edges.npy             - mask contour map
    control_points.npy    - 5x5 grid of (x, y) points over the cloth
    meta.json             - source path, target_size, content_bbox, mtime
    model_tensors.npz     - optional: RGB in [-1, 1] + mask in [0, 1]

Cache invalidates when the source PNG mtime changes.

Notable processing steps:
  - Alpha-dehalo / edge bleed: replicates edge BGR pixels into the
    transparent region so when the runtime warp interpolates and the
    composite feathers, no dark fringe appears around the shirt. This
    fixes the navy-outline halo visible in many user screenshots.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from engine.coreutils import setup_logger, ensure_bgra

logger = setup_logger("cloth_preprocessor")


@dataclass
class ProcessedCloth:
    name: str
    source_path: str
    image_bgra: np.ndarray
    mask: np.ndarray
    edges: np.ndarray
    control_points: np.ndarray  # (25, 2) for a 5x5 grid
    content_bbox: Tuple[int, int, int, int]
    target_size: Tuple[int, int] = (512, 384)  # (height, width)
    thumbnail: Optional[np.ndarray] = None
    model_image: Optional[np.ndarray] = None   # (H, W, 3) float32 in [-1, 1]
    model_mask: Optional[np.ndarray] = None    # (H, W) float32 in [0, 1]


class ClothPreprocessor:
    """Offline garment processor with disk cache.

    Usage:
        pre = ClothPreprocessor()
        processed = pre.process("assets/shirts/shirt2.png")
        # subsequent calls hit the cache unless the PNG mtime changes
    """

    def __init__(
        self,
        cache_root: str = "assets/processed_shirts",
        target_size: Tuple[int, int] = (512, 384),
        grid_size: Tuple[int, int] = (5, 5),
        alpha_threshold: int = 16,
        morph_open_radius: int = 2,
        morph_close_radius: int = 4,
        dehalo_iterations: int = 8,
        emit_model_tensors: bool = True,
        thumbnail_size: Tuple[int, int] = (80, 100),
    ):
        self.cache_root = Path(cache_root)
        self.target_size = target_size           # (H, W)
        self.grid_size = grid_size               # (rows, cols)
        self.alpha_threshold = int(alpha_threshold)
        self.morph_open_radius = int(morph_open_radius)
        self.morph_close_radius = int(morph_close_radius)
        self.dehalo_iterations = int(dehalo_iterations)
        self.emit_model_tensors = bool(emit_model_tensors)
        self.thumbnail_size = thumbnail_size

    # ── Public API ───────────────────────────────────────────────────────────

    def process(self, source_path: str, force: bool = False) -> ProcessedCloth:
        """Return processed cloth, using cache when valid."""
        src = Path(source_path)
        if not src.exists():
            raise FileNotFoundError(f"Cloth source not found: {src}")

        name = src.stem
        out_dir = self.cache_root / name

        if not force and self._cache_is_valid(out_dir, src):
            try:
                return self._load_from_cache(out_dir, name, source_path=str(src))
            except Exception as exc:
                logger.warning(
                    "Cache for %s unreadable (%s); reprocessing.", name, exc,
                )

        processed = self._process_fresh(src, name)
        self._save_to_cache(out_dir, processed, src)
        return processed

    def is_cached(self, source_path: str) -> bool:
        src = Path(source_path)
        return self._cache_is_valid(self.cache_root / src.stem, src)

    # ── Pipeline ────────────────────────────────────────────────────────────

    def _process_fresh(self, src: Path, name: str) -> ProcessedCloth:
        t0 = time.perf_counter()

        raw = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
        if raw is None:
            raise ValueError(f"Failed to read cloth image: {src}")
        bgra = ensure_bgra(raw)

        alpha = bgra[:, :, 3]
        mask = self._clean_mask(alpha)

        bbox = self._content_bbox(mask)
        if bbox is None:
            raise ValueError(f"Cloth has no opaque content: {src}")

        x1, y1, x2, y2 = bbox
        bgra_crop = bgra[y1:y2, x1:x2].copy()
        mask_crop = mask[y1:y2, x1:x2].copy()
        bgra_crop[:, :, 3] = mask_crop  # apply cleaned mask

        # Pad / center into target frame FIRST. The padding is initially
        # transparent black (alpha=0, BGR=0). We then dehalo the *entire*
        # canvas so any warp sample anywhere in the 512x384 area returns
        # cloth-coloured BGR — not just within the cropped content bbox.
        canvas, place_x, place_y, scale = self._place_into_target(bgra_crop)

        # Alpha-dehalo across the full canvas: cv2.inpaint extrapolates
        # cloth interior color into the entire transparent region.
        canvas = self._dehalo_alpha(canvas)

        # Binary mask for clipping / edges / point-snapping. The image's
        # alpha channel keeps the soft interpolated values from the resize,
        # so anti-aliased edges still render correctly at runtime.
        _, canvas_mask = cv2.threshold(
            canvas[:, :, 3], self.alpha_threshold, 255, cv2.THRESH_BINARY,
        )
        canvas_mask = canvas_mask.astype(np.uint8)

        edges = self._edge_map(canvas_mask)
        cps = self._grid_control_points(canvas_mask)

        thumb = cv2.resize(
            canvas, self.thumbnail_size, interpolation=cv2.INTER_AREA,
        )

        model_image = None
        model_mask = None
        if self.emit_model_tensors:
            rgb = cv2.cvtColor(canvas[:, :, :3], cv2.COLOR_BGR2RGB)
            model_image = (rgb.astype(np.float32) / 127.5) - 1.0
            model_mask = (canvas_mask.astype(np.float32) / 255.0)

        # content_bbox: where the cloth pixels actually live INSIDE the
        # padded canvas. Computed from the post-resize dimensions, not the
        # raw crop — earlier code used `bgra_dehalo.shape` which was the
        # pre-resize size and gave wrong bbox values in the meta log.
        crop_h = max(1, y2 - y1)
        crop_w = max(1, x2 - x1)
        new_w = max(1, int(round(crop_w * scale)))
        new_h = max(1, int(round(crop_h * scale)))
        content_bbox_canvas = (
            place_x, place_y,
            place_x + new_w,
            place_y + new_h,
        )

        dt = (time.perf_counter() - t0) * 1000.0
        logger.info(
            "preprocessed %s | size=%s -> %s | bbox=%s | scale=%.3f | %.1fms",
            name, raw.shape[:2], canvas.shape[:2], content_bbox_canvas, scale, dt,
        )

        return ProcessedCloth(
            name=name,
            source_path=str(src),
            image_bgra=canvas,
            mask=canvas_mask,
            edges=edges,
            control_points=cps,
            content_bbox=content_bbox_canvas,
            target_size=self.target_size,
            thumbnail=thumb,
            model_image=model_image,
            model_mask=model_mask,
        )

    # ── Steps ───────────────────────────────────────────────────────────────

    def _clean_mask(self, alpha: np.ndarray) -> np.ndarray:
        """Threshold + open + close — robust binary cloth mask."""
        _, m = cv2.threshold(alpha, self.alpha_threshold, 255, cv2.THRESH_BINARY)
        if self.morph_open_radius > 0:
            k = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (self.morph_open_radius * 2 + 1, self.morph_open_radius * 2 + 1),
            )
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
        if self.morph_close_radius > 0:
            k = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (self.morph_close_radius * 2 + 1, self.morph_close_radius * 2 + 1),
            )
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
        return m.astype(np.uint8)

    def _content_bbox(
        self, mask: np.ndarray,
    ) -> Optional[Tuple[int, int, int, int]]:
        rows = np.any(mask > 0, axis=1)
        cols = np.any(mask > 0, axis=0)
        if not rows.any():
            return None
        y1, y2 = int(np.where(rows)[0][0]), int(np.where(rows)[0][-1])
        x1, x2 = int(np.where(cols)[0][0]), int(np.where(cols)[0][-1])
        return x1, y1, x2 + 1, y2 + 1

    def _dehalo_alpha(self, bgra: np.ndarray) -> np.ndarray:
        """Fill the entire transparent region with extrapolated cloth color.

        Why: standard PNGs store BGR = 0 outside the alpha. When the warp
        interpolates and the composite feathers, those zeros appear as a
        dark ring around the shirt — and at the shoulder seams, where the
        warp pulls the shirt outward to fit a wider body, the warp samples
        *into* the dark transparent region and produces visible black
        patches in the rendered shirt.

        Strategy: use `cv2.inpaint` (Telea fast-marching) on the
        transparent area. It extrapolates from the cloth's INTERIOR
        pixels — not just the edge pixels — so dark seam shadows at the
        cloth boundary don't propagate the way they would with a simple
        box-filter bleed. The output BGR is smooth and cloth-coloured
        across the whole canvas; the alpha channel is untouched, so the
        cloth shape remains identical.
        """
        bgr = bgra[:, :, :3].copy()
        alpha = bgra[:, :, 3]
        # Inpaint mask: anywhere alpha is below the threshold (transparent
        # / near-transparent) is filled. Telea does fast smooth extrapolation.
        inpaint_mask = (alpha < self.alpha_threshold).astype(np.uint8) * 255
        if not np.any(inpaint_mask) or not np.any(255 - inpaint_mask):
            return bgra

        # Radius=5 gives smooth fill without smearing detail; larger
        # values bleed cloth color further into the canvas.
        try:
            filled_bgr = cv2.inpaint(bgr, inpaint_mask, 5, cv2.INPAINT_TELEA)
        except cv2.error:
            # Fallback: simple repeat-blur fill if inpaint isn't available.
            filled_bgr = self._iterative_bleed(bgr, alpha)

        out = np.dstack([filled_bgr, alpha]).astype(np.uint8)
        return out

    def _iterative_bleed(self, bgr: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        """Fallback dehalo: repeated box-filter bleed (used if inpaint fails)."""
        valid = (alpha > 0).astype(np.uint8) * 255
        out = bgr.copy()
        if not np.any(valid):
            return out
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        for _ in range(max(1, self.dehalo_iterations)):
            grown = cv2.dilate(valid, kernel, iterations=1)
            new_pixels = cv2.bitwise_and(grown, cv2.bitwise_not(valid))
            if not np.any(new_pixels):
                break
            bgr_masked = cv2.bitwise_and(out, out, mask=valid)
            blur_sum = cv2.boxFilter(
                bgr_masked.astype(np.float32), -1, (3, 3),
                normalize=False, borderType=cv2.BORDER_REPLICATE,
            )
            valid_f = valid.astype(np.float32) / 255.0
            count = cv2.boxFilter(
                valid_f, -1, (3, 3),
                normalize=False, borderType=cv2.BORDER_REPLICATE,
            )
            count = np.maximum(count, 1e-3)
            avg = (blur_sum / count[:, :, None]).astype(np.uint8)
            new_mask3 = (new_pixels > 0)[:, :, None]
            out = np.where(new_mask3, avg, out)
            valid = grown
        return out

    def _place_into_target(
        self, bgra: np.ndarray,
    ) -> Tuple[np.ndarray, int, int, float]:
        """Resize-with-aspect into target_size and center on a transparent canvas."""
        th, tw = self.target_size
        sh, sw = bgra.shape[:2]
        scale = min(tw / sw, th / sh)
        new_w = max(1, int(round(sw * scale)))
        new_h = max(1, int(round(sh * scale)))
        resized = cv2.resize(bgra, (new_w, new_h), interpolation=cv2.INTER_AREA)

        canvas = np.zeros((th, tw, 4), dtype=np.uint8)
        place_x = (tw - new_w) // 2
        place_y = (th - new_h) // 2
        canvas[place_y:place_y + new_h, place_x:place_x + new_w] = resized
        return canvas, place_x, place_y, scale

    def _edge_map(self, mask: np.ndarray) -> np.ndarray:
        """Single-pixel-wide cloth contour."""
        edges = cv2.Canny(mask, 50, 150)
        return edges.astype(np.uint8)

    def _grid_control_points(self, mask: np.ndarray) -> np.ndarray:
        """N x M evenly-spaced points over the cloth's bounding box.

        Points are guaranteed to land on opaque mask pixels by snapping
        each (x, y) to the nearest opaque pixel within a small radius;
        if no opaque pixel is nearby the original grid coordinate is kept
        (caller can ignore via the mask if needed).
        """
        rows, cols = self.grid_size
        bbox = self._content_bbox(mask)
        if bbox is None:
            return np.zeros((rows * cols, 2), dtype=np.float32)
        x1, y1, x2, y2 = bbox

        xs = np.linspace(x1, x2 - 1, cols, dtype=np.float32)
        ys = np.linspace(y1, y2 - 1, rows, dtype=np.float32)

        pts = []
        for y in ys:
            for x in xs:
                snapped = self._snap_to_mask(mask, int(round(x)), int(round(y)), radius=12)
                pts.append(snapped if snapped is not None else (float(x), float(y)))
        return np.array(pts, dtype=np.float32)

    def _snap_to_mask(
        self, mask: np.ndarray, x: int, y: int, radius: int = 12,
    ) -> Optional[Tuple[float, float]]:
        h, w = mask.shape[:2]
        if not (0 <= x < w and 0 <= y < h):
            return None
        if mask[y, x] > 0:
            return float(x), float(y)
        # Search a small neighbourhood for the nearest opaque pixel.
        x1 = max(0, x - radius); x2 = min(w, x + radius + 1)
        y1 = max(0, y - radius); y2 = min(h, y + radius + 1)
        sub = mask[y1:y2, x1:x2]
        ys, xs = np.where(sub > 0)
        if xs.size == 0:
            return None
        dx = xs - (x - x1)
        dy = ys - (y - y1)
        dist2 = dx * dx + dy * dy
        idx = int(np.argmin(dist2))
        return float(x1 + xs[idx]), float(y1 + ys[idx])

    # ── Cache I/O ───────────────────────────────────────────────────────────

    def _cache_is_valid(self, out_dir: Path, src: Path) -> bool:
        meta_path = out_dir / "meta.json"
        if not meta_path.exists():
            return False
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            return False
        cached_mtime = float(meta.get("source_mtime", 0))
        try:
            current_mtime = float(src.stat().st_mtime)
        except OSError:
            return False
        if abs(cached_mtime - current_mtime) > 1e-3:
            return False
        # Verify all expected files exist.
        for fn in ("image.png", "mask.npy", "edges.npy", "control_points.npy"):
            if not (out_dir / fn).exists():
                return False
        # target_size sanity
        cached_size = tuple(meta.get("target_size", []))
        if tuple(self.target_size) != cached_size:
            return False
        return True

    def _save_to_cache(
        self, out_dir: Path, p: ProcessedCloth, src: Path,
    ) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_dir / "image.png"), p.image_bgra)
        np.save(out_dir / "mask.npy", p.mask)
        np.save(out_dir / "edges.npy", p.edges)
        np.save(out_dir / "control_points.npy", p.control_points)
        if p.thumbnail is not None:
            cv2.imwrite(str(out_dir / "thumbnail.png"), p.thumbnail)
        if p.model_image is not None and p.model_mask is not None:
            np.savez_compressed(
                out_dir / "model_tensors.npz",
                image=p.model_image,
                mask=p.model_mask,
            )
        meta = {
            "name": p.name,
            "source_path": str(src),
            "target_size": list(self.target_size),
            "content_bbox": list(p.content_bbox),
            "source_mtime": float(src.stat().st_mtime),
            "version": 1,
        }
        (out_dir / "meta.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8",
        )

    def _load_from_cache(
        self, out_dir: Path, name: str, source_path: str,
    ) -> ProcessedCloth:
        image = cv2.imread(str(out_dir / "image.png"), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise FileNotFoundError(out_dir / "image.png")
        image = ensure_bgra(image)
        mask = np.load(out_dir / "mask.npy")
        edges = np.load(out_dir / "edges.npy")
        cps = np.load(out_dir / "control_points.npy")
        thumb = None
        thumb_path = out_dir / "thumbnail.png"
        if thumb_path.exists():
            thumb = cv2.imread(str(thumb_path), cv2.IMREAD_UNCHANGED)
            if thumb is not None:
                thumb = ensure_bgra(thumb)

        model_image = None
        model_mask = None
        tensors_path = out_dir / "model_tensors.npz"
        if tensors_path.exists() and self.emit_model_tensors:
            with np.load(tensors_path) as data:
                model_image = data["image"]
                model_mask = data["mask"]

        meta = json.loads((out_dir / "meta.json").read_text(encoding="utf-8"))
        bbox = tuple(meta.get("content_bbox", (0, 0, image.shape[1], image.shape[0])))

        return ProcessedCloth(
            name=name,
            source_path=source_path,
            image_bgra=image,
            mask=mask,
            edges=edges,
            control_points=cps,
            content_bbox=bbox,
            target_size=self.target_size,
            thumbnail=thumb,
            model_image=model_image,
            model_mask=model_mask,
        )


__all__ = ["ProcessedCloth", "ClothPreprocessor"]
