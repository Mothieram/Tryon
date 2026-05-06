"""
parsing_engine.py - SCHP human parsing adapter used by render pipeline.
"""

from __future__ import annotations

import sys
import importlib
import types
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np
import torchvision.transforms as transforms

# Ensure project root is importable when run as:
#   python engine/parsing_engine.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from engine.coreutils import setup_logger

try:
    import torch
except Exception:  # pragma: no cover - optional runtime dependency
    torch = None
try:
    import torch.nn as nn
except Exception:
    nn = None
try:
    import torch.nn.functional as F
except Exception:
    F = None


def _resolve_schp_root() -> Path:
    schp_env = None
    try:
        import os
        schp_env = os.environ.get("SCHP_ROOT")
    except Exception:
        schp_env = None

    candidates = []
    if schp_env:
        candidates.append(Path(schp_env))
    candidates.extend(
        [
            PROJECT_ROOT / "Self-Correction-Human-Parsing",
            PROJECT_ROOT / "self-correction-human-parsing",
            PROJECT_ROOT / "engine",  # engine-local fallback bundle: networks/utils/modules
        ]
    )
    for p in candidates:
        if (p / "networks").exists() and (p / "modules").exists() and (p / "utils").exists():
            return p
    return PROJECT_ROOT / "Self-Correction-Human-Parsing"


SCHP_ROOT = _resolve_schp_root()
if str(SCHP_ROOT) not in sys.path:
    sys.path.insert(0, str(SCHP_ROOT))

_SCHP_IMPORT_ERROR: Optional[str] = None


def _install_abn_fallback_module() -> bool:
    """
    Provide a pure-PyTorch fallback for SCHP `modules.InPlaceABN*` classes.
    This avoids building inplace_abn C++/CUDA extension at runtime.
    """
    if nn is None or F is None:
        return False

    mod = types.ModuleType("modules")

    class _ABNBase(nn.Module):
        def __init__(
            self,
            num_features: int,
            eps: float = 1e-5,
            momentum: float = 0.1,
            affine: bool = True,
            activation: str = "leaky_relu",
            slope: float = 0.01,
        ):
            super().__init__()
            self.num_features = num_features
            self.eps = eps
            self.momentum = momentum
            self.affine = affine
            self.activation = activation
            self.slope = slope
            if self.affine:
                self.weight = nn.Parameter(torch.ones(num_features))
                self.bias = nn.Parameter(torch.zeros(num_features))
            else:
                self.register_parameter("weight", None)
                self.register_parameter("bias", None)
            self.register_buffer("running_mean", torch.zeros(num_features))
            self.register_buffer("running_var", torch.ones(num_features))

        def forward(self, x):
            x = F.batch_norm(
                x,
                self.running_mean,
                self.running_var,
                self.weight,
                self.bias,
                self.training,
                self.momentum,
                self.eps,
            )
            if self.activation == "none":
                return x
            if self.activation == "relu":
                return F.relu(x, inplace=True)
            if self.activation == "elu":
                return F.elu(x, inplace=True)
            # Default used by SCHP: leaky_relu
            return F.leaky_relu(x, negative_slope=self.slope, inplace=True)

    class InPlaceABN(_ABNBase):
        pass

    class InPlaceABNSync(_ABNBase):
        pass

    # Export names expected by SCHP networks.
    mod.InPlaceABN = InPlaceABN
    mod.InPlaceABNSync = InPlaceABNSync
    mod.ABN = _ABNBase
    mod.ACT_RELU = "relu"
    mod.ACT_LEAKY_RELU = "leaky_relu"
    mod.ACT_ELU = "elu"
    mod.ACT_NONE = "none"

    sys.modules["modules"] = mod
    return True


try:
    networks = importlib.import_module("networks")
    schp_transforms = importlib.import_module("utils.transforms")

    # Guard against importing wrong "utils" package from project code.
    networks_file = Path(getattr(networks, "__file__", "")).resolve()
    transforms_file = Path(getattr(schp_transforms, "__file__", "")).resolve()
    if not str(networks_file).startswith(str(SCHP_ROOT.resolve())):
        raise ImportError(f"Imported wrong networks module: {networks_file}")
    if not str(transforms_file).startswith(str(SCHP_ROOT.resolve())):
        raise ImportError(f"Imported wrong utils.transforms module: {transforms_file}")

    get_affine_transform = schp_transforms.get_affine_transform
    transform_logits = schp_transforms.transform_logits
except Exception as exc:
    # Retry once with pure-PyTorch ABN fallback when C++ extension build fails.
    first_error = str(exc)
    networks = None
    get_affine_transform = None
    transform_logits = None
    retried = False
    if any(k in first_error.lower() for k in ["ninja", "c++ extensions", "inplace_abn", "no module named 'modules'"]):
        retried = _install_abn_fallback_module()
        if retried:
            try:
                networks = importlib.import_module("networks")
                schp_transforms = importlib.import_module("utils.transforms")
                get_affine_transform = schp_transforms.get_affine_transform
                transform_logits = schp_transforms.transform_logits
            except Exception as exc2:
                _SCHP_IMPORT_ERROR = f"{first_error} | fallback retry failed: {exc2}"
        else:
            _SCHP_IMPORT_ERROR = f"{first_error} | fallback module unavailable"
    if not retried and _SCHP_IMPORT_ERROR is None:
        _SCHP_IMPORT_ERROR = first_error

logger = setup_logger("parsing")

class ParsedRegions:
    """Binary region masks returned by the parser."""

    def __init__(self, h: int, w: int, method: str = "parsing"):
        self.torso = np.zeros((h, w), np.uint8)
        self.left_arm = np.zeros((h, w), np.uint8)
        self.right_arm = np.zeros((h, w), np.uint8)
        self.face = np.zeros((h, w), np.uint8)
        self.hair = np.zeros((h, w), np.uint8)
        self.legs = np.zeros((h, w), np.uint8)
        self.method = method

    @property
    def arms_combined(self) -> np.ndarray:
        return cv2.bitwise_or(self.left_arm, self.right_arm)

    @property
    def head_region(self) -> np.ndarray:
        return cv2.bitwise_or(self.face, self.hair)


class ParsingEngine:
    """SCHP parser with safe fallback when model/deps are unavailable."""

    def __init__(self, model_path: Optional[str] = None, device: str = "auto"):
        self.input_size = [512, 512]
        self.num_classes = 18
        self.aspect_ratio = self.input_size[1] * 1.0 / self.input_size[0]

        self._is_loaded = False
        self._load_error: Optional[str] = None
        self.model = None

        self.device = self._resolve_device(device)
        self._use_half = self.device.startswith("cuda")
        self.model_path = model_path or str(
            Path(__file__).resolve().parents[1] / "models" / "exp-schp-201908301523-atr.pth"
        )

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.406, 0.456, 0.485], std=[0.225, 0.224, 0.229]),
            ]
        )
        self._try_load_model()

    def _resolve_device(self, device: str) -> str:
        if device != "auto":
            return "cuda" if str(device).startswith("cuda") else str(device)
        if torch is not None and torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def _try_load_model(self) -> None:
        if torch is None:
            self._load_error = "torch is not installed"
            return
        if networks is None or get_affine_transform is None or transform_logits is None:
            msg = "SCHP imports failed (networks/utils.transforms)"
            if _SCHP_IMPORT_ERROR:
                msg = f"{msg}: {_SCHP_IMPORT_ERROR}"
            if not SCHP_ROOT.exists():
                msg = f"{msg} | SCHP root not found: {SCHP_ROOT}"
            self._load_error = msg
            return
        if not Path(self.model_path).exists():
            self._load_error = f"SCHP checkpoint not found: {self.model_path}"
            return

        try:
            self.model = networks.init_model(
                "resnet101",
                num_classes=self.num_classes,
                pretrained=None,
            )
            ckpt = torch.load(self.model_path, map_location=self.device, weights_only=False)
            state_dict = ckpt.get("state_dict", ckpt)
            new_state = OrderedDict((k.replace("module.", ""), v) for k, v in state_dict.items())
            self.model.load_state_dict(new_state, strict=True)
            self.model.to(self.device)
            if self._use_half:
                self.model.half()
            self.model.eval()
            self._is_loaded = True
        except Exception as exc:
            self._load_error = str(exc)
            self.model = None
            self._is_loaded = False

    def _xywh2cs(self, x: float, y: float, w: float, h: float):
        center = np.zeros((2), dtype=np.float32)
        center[0] = x + w * 0.5
        center[1] = y + h * 0.5

        if w > self.aspect_ratio * h:
            h = w / self.aspect_ratio
        elif w < self.aspect_ratio * h:
            w = h * self.aspect_ratio

        scale = np.array([w, h], dtype=np.float32)
        return center, scale

    def _preprocess(self, frame: np.ndarray):
        h, w = frame.shape[:2]
        center, scale = self._xywh2cs(0, 0, w - 1, h - 1)
        trans = get_affine_transform(center, scale, 0, np.asarray(self.input_size))
        warped = cv2.warpAffine(
            frame,
            trans,
            (int(self.input_size[1]), int(self.input_size[0])),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        return warped, center, scale, w, h

    def parse(self, frame: np.ndarray, pose: Optional[Any] = None) -> ParsedRegions:
        h, w = frame.shape[:2]
        if not self._is_loaded or self.model is None:
            return self._fallback_regions(frame, pose, method="fallback")

        img, center, scale, w, h = self._preprocess(frame)
        # Re-read actual frame dimensions (preprocess may change h/w names)
        orig_h, orig_w = frame.shape[:2]
        tensor = self.transform(img).unsqueeze(0).to(self.device)
        if self._use_half:
            tensor = tensor.half()

        try:
            with torch.inference_mode():
                output = self.model(tensor)
                logits_chw = self._extract_model_logits_chw(output)
                if logits_chw is None:
                    raise RuntimeError("Unable to extract logits from SCHP model output.")
                upsample = torch.nn.Upsample(size=self.input_size, mode="bilinear", align_corners=True)
                pred = upsample(logits_chw.unsqueeze(0))
                # OpenCV remap path inside transform_logits does not support float16 input.
                # Keep model/inference in half on GPU, then cast logits to float32 before numpy/OpenCV.
                pred = pred.float().squeeze().permute(1, 2, 0).cpu().numpy()

            logits = transform_logits(
                pred,
                center,
                scale,
                orig_w,
                orig_h,
                input_size=np.asarray(self.input_size),
            )
            parsing = np.argmax(logits, axis=2).astype(np.uint8)
            regions = self.extract_regions(parsing, orig_h, orig_w, method="parsing")
            # Guarantee all masks are exactly (orig_h, orig_w)
            regions = self._normalize_regions(regions, orig_h, orig_w)
            return regions
        except Exception as exc:
            logger.warning(f"Parsing inference failed, using fallback: {exc}")
            return self._fallback_regions(frame, pose, method="fallback")

    def _extract_model_logits_chw(self, output: Any) -> Optional[Any]:
        """
        Normalize SCHP model output to a tensor shaped (C, H, W).
        Prefer exact CE2P path used in `simple_extractor.py`:
            output[0][-1][0]
        Then fall back to generic unwrapping.
        """
        # 1) Exact CE2P/simple_extractor path
        try:
            x = output[0][-1][0]
            if hasattr(x, "dim"):
                if x.dim() == 4:
                    return x[0]
                if x.dim() == 3:
                    return x
        except Exception:
            pass

        # 2) Secondary CE2P candidate: first parsing head output
        try:
            x = output[0][0][0]
            if hasattr(x, "dim"):
                if x.dim() == 4:
                    return x[0]
                if x.dim() == 3:
                    return x
        except Exception:
            pass

        # 3) Generic unwrapping fallback
        x = output
        while isinstance(x, (list, tuple)) and len(x) > 0:
            x = x[0] if len(x) == 1 else x[-1]
        if isinstance(x, (list, tuple)) or not hasattr(x, "dim"):
            return None
        if x.dim() == 4:
            return x[0]
        if x.dim() == 3:
            return x
        return None

    def extract_regions(self, mask: np.ndarray, h: int, w: int, method: str = "parsing") -> ParsedRegions:
        r = ParsedRegions(h, w, method=method)

        def mk(ids):
            out = np.zeros((h, w), np.uint8)
            for i in ids:
                out[mask == i] = 255
            return out

        torso = mk([4, 7])
        left_arm = mk([14])
        right_arm = mk([15])
        face = mk([11])
        hair = mk([1, 2])
        legs = mk([12, 13])

        k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        k9 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

        torso = cv2.morphologyEx(torso, cv2.MORPH_CLOSE, k9)
        torso = cv2.morphologyEx(torso, cv2.MORPH_OPEN, k5)

        head = cv2.bitwise_or(face, hair)
        head = cv2.dilate(head, k5, iterations=2)
        torso = cv2.bitwise_and(torso, cv2.bitwise_not(head))

        left_arm = cv2.morphologyEx(left_arm, cv2.MORPH_CLOSE, k5)
        right_arm = cv2.morphologyEx(right_arm, cv2.MORPH_CLOSE, k5)
        torso = cv2.bitwise_and(torso, cv2.bitwise_not(left_arm))
        torso = cv2.bitwise_and(torso, cv2.bitwise_not(right_arm))

        face = cv2.morphologyEx(face, cv2.MORPH_CLOSE, k3)
        hair = cv2.morphologyEx(hair, cv2.MORPH_CLOSE, k3)
        legs = cv2.morphologyEx(legs, cv2.MORPH_CLOSE, k5)

        r.torso = torso
        r.left_arm = left_arm
        r.right_arm = right_arm
        r.face = face
        r.hair = hair
        r.legs = legs
        return r

    def _fallback_regions(self, frame: np.ndarray, pose: Optional[Any], method: str) -> ParsedRegions:
        """Approximate segmentation from keypoints when SCHP is unavailable."""
        h, w = frame.shape[:2]
        r = ParsedRegions(h, w, method=method)
        if pose is None:
            return r

        ls = getattr(pose, "left_shoulder", None)
        rs = getattr(pose, "right_shoulder", None)
        lh = getattr(pose, "left_hip", None)
        rh = getattr(pose, "right_hip", None)
        nose = getattr(pose, "nose", None)

        if ls and rs and ls.valid and rs.valid:
            sw = max(20.0, getattr(pose, "shoulder_width", 0.0))
            expand = sw * 0.15
            pts = [[ls.x - expand, ls.y], [rs.x + expand, rs.y]]
            if rh and rh.valid:
                pts.append([rh.x + expand * 0.4, rh.y + sw * 0.1])
            if lh and lh.valid:
                pts.append([lh.x - expand * 0.4, lh.y + sw * 0.1])
            if len(pts) >= 3:
                cv2.fillConvexPoly(r.torso, np.array(pts, dtype=np.int32), 255)
                r.torso = cv2.GaussianBlur(r.torso, (11, 11), 0)

            def draw_arm(mask: np.ndarray, p1, p2, p3=None):
                if p1 and p2 and p1.valid and p2.valid:
                    t = max(10, int(sw * 0.13))
                    cv2.line(mask, p1.to_tuple(), p2.to_tuple(), 255, t)
                    if p3 and p3.valid:
                        cv2.line(mask, p2.to_tuple(), p3.to_tuple(), 255, int(t * 0.9))

            draw_arm(r.left_arm, ls, getattr(pose, "left_elbow", None), getattr(pose, "left_wrist", None))
            draw_arm(r.right_arm, rs, getattr(pose, "right_elbow", None), getattr(pose, "right_wrist", None))
            r.left_arm = cv2.GaussianBlur(r.left_arm, (9, 9), 0)
            r.right_arm = cv2.GaussianBlur(r.right_arm, (9, 9), 0)

            if nose and nose.valid:
                center = (int(nose.x), int(nose.y))
                cv2.ellipse(r.face, center, (int(sw * 0.20), int(sw * 0.24)), 0, 0, 360, 255, -1)
                cv2.ellipse(
                    r.hair,
                    (center[0], center[1] - int(sw * 0.10)),
                    (int(sw * 0.24), int(sw * 0.18)),
                    0,
                    0,
                    360,
                    255,
                    -1,
                )
        return self._normalize_regions(r, h, w)

    def _normalize_regions(self, r: ParsedRegions, h: int, w: int) -> ParsedRegions:
        """Ensure every mask in a ParsedRegions object is exactly (h, w) uint8."""
        for attr in ("torso", "left_arm", "right_arm", "face", "hair", "legs"):
            mask = getattr(r, attr)
            if mask.shape != (h, w):
                resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                setattr(r, attr, resized.astype(np.uint8))
            elif mask.dtype != np.uint8:
                setattr(r, attr, mask.astype(np.uint8))
        return r

    def get_status(self) -> Dict[str, Any]:
        return {
            "loaded": self._is_loaded,
            "device": self.device,
            "half": self._use_half,
            "input_size": tuple(self.input_size),
            "model_path": self.model_path,
            "error": self._load_error,
        }

    def set_input_size(self, size: int):
        sz = int(np.clip(int(size), 256, 640))
        self.input_size = [sz, sz]
        self.aspect_ratio = self.input_size[1] * 1.0 / self.input_size[0]


__all__ = ["ParsingEngine", "ParsedRegions"]


if __name__ == "__main__":
    import argparse
    from engine.coreutils import Keypoint, PoseKeypoints

    parser = argparse.ArgumentParser(description="Quick parsing engine smoke test.")
    parser.add_argument(
        "--image",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "person.jpg"),
        help="Path to input image.",
    )
    args = parser.parse_args()

    img = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"Could not read image: {args.image}")

    engine = ParsingEngine()
    pose = None
    # If SCHP is unavailable, build a tiny synthetic pose so fallback test is visible.
    if not engine.get_status().get("loaded", False):
        h, w = img.shape[:2]
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

    regions = engine.parse(img, pose=pose)
    status = engine.get_status()
    print(f"Parser loaded: {status['loaded']} | device: {status['device']}")
    if status.get("error"):
        print(f"Parser load error: {status['error']}")

    counts = {
        "torso": int(np.count_nonzero(regions.torso)),
        "left_arm": int(np.count_nonzero(regions.left_arm)),
        "right_arm": int(np.count_nonzero(regions.right_arm)),
        "face": int(np.count_nonzero(regions.face)),
        "hair": int(np.count_nonzero(regions.hair)),
        "legs": int(np.count_nonzero(regions.legs)),
    }
    print(f"Method: {regions.method} | mask pixels: {counts}")

    # Color style close to your reference image (BGR):
    # torso: blue, hair: purple, face: orange, left arm: green, right arm: yellow.
    vis = np.zeros_like(img)
    vis[regions.torso > 0] = (180, 40, 20)      # deep blue
    vis[regions.left_arm > 0] = (90, 230, 70)   # green
    vis[regions.right_arm > 0] = (90, 220, 250) # yellow
    vis[regions.face > 0] = (40, 170, 255)      # orange
    vis[regions.hair > 0] = (170, 30, 170)      # purple
    vis[regions.legs > 0] = (150, 0, 120)       # violet

    # Class-aware alpha blend for vivid parsing look.
    alpha = np.zeros(img.shape[:2], dtype=np.float32)
    alpha[regions.torso > 0] = 0.70
    alpha[regions.left_arm > 0] = 0.70
    alpha[regions.right_arm > 0] = 0.70
    alpha[regions.face > 0] = 0.55
    alpha[regions.hair > 0] = 0.75
    alpha[regions.legs > 0] = 0.65
    alpha = cv2.GaussianBlur(alpha, (5, 5), 0)
    alpha3 = np.stack([alpha] * 3, axis=2)
    overlay = (img.astype(np.float32) * (1 - alpha3) + vis.astype(np.float32) * alpha3).astype(np.uint8)
    out_dir = Path(__file__).resolve().parents[1]
    color_out = out_dir / "parsing_test_colored.png"
    overlay_out = out_dir / "parsing_overlay.png"
    cv2.imwrite(str(color_out), vis)
    cv2.imwrite(str(overlay_out), overlay)
    print(f"Saved: {color_out}")
    print(f"Saved: {overlay_out}")
