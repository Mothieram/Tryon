"""
mediapipe_holistic_pose.py - MediaPipe Holistic Pose Engine
Pose tracking via MediaPipe Holistic landmarks.
"""

import time
import threading
from typing import Optional, Dict, Any, Tuple, List

import cv2
import numpy as np

from engine.coreutils import (
    setup_logger,
    Keypoint,
    PoseKeypoints,
    smooth_array,
    FPSCounter,
)

logger = setup_logger("mediapipe_holistic_pose")


class MediaPipeHolisticPoseEngine:
    """
    MediaPipe Holistic pose engine with COCO-17 compatibility output.
    """

    COCO_FROM_MEDIAPIPE = {
        0: 0,    # nose
        1: 2,    # left_eye
        2: 5,    # right_eye
        3: 7,    # left_ear
        4: 8,    # right_ear
        5: 11,   # left_shoulder
        6: 12,   # right_shoulder
        7: 13,   # left_elbow
        8: 14,   # right_elbow
        9: 15,   # left_wrist
        10: 16,  # right_wrist
        11: 23,  # left_hip
        12: 24,  # right_hip
        13: 25,  # left_knee
        14: 26,  # right_knee
        15: 27,  # left_ankle
        16: 28,  # right_ankle
    }

    def __init__(
        self,
        keypoint_conf: float = 0.3,
        smooth_alpha: float = 0.4,
        model_complexity: int = 1,
    ):
        self.keypoint_conf = float(np.clip(keypoint_conf, 0.05, 0.9))
        self.smooth_alpha = float(np.clip(smooth_alpha, 0.05, 1.0))
        self.model_complexity = int(np.clip(model_complexity, 0, 2))

        self._is_loaded = False
        self._load_error: Optional[str] = None
        self._mp_holistic = None
        self._holistic = None
        self._max_input_size = 960
        self._prev_keypoints: Optional[np.ndarray] = None

        self.inference_ms: float = 0.0
        self._fps = FPSCounter(window=20)

        logger.info(
            "MediaPipeHolisticPoseEngine initialized | complexity=%d",
            self.model_complexity,
        )

    def load(self) -> bool:
        try:
            import mediapipe as mp
        except Exception as e:
            self._load_error = f"mediapipe import failed: {e}"
            logger.error(self._load_error)
            return False

        try:
            self._mp_holistic = mp.solutions.holistic
            self._holistic = self._mp_holistic.Holistic(
                static_image_mode=False,
                model_complexity=self.model_complexity,
                smooth_landmarks=True,
                refine_face_landmarks=False,
                min_detection_confidence=max(0.2, self.keypoint_conf),
                min_tracking_confidence=max(0.2, self.keypoint_conf),
            )

            dummy = np.zeros((320, 320, 3), dtype=np.uint8)
            _ = self.detect(dummy)

            self._is_loaded = True
            logger.info("MediaPipe Holistic loaded successfully")
            return True
        except Exception as e:
            self._load_error = str(e)
            logger.error("Failed to load MediaPipe Holistic: %s", e)
            return False

    def close(self):
        try:
            if self._holistic is not None:
                self._holistic.close()
        except Exception:
            pass
        self._holistic = None
        self._is_loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    @property
    def load_error(self) -> Optional[str]:
        return self._load_error

    def _prepare_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, float]:
        h, w = frame.shape[:2]
        max_side = max(h, w)
        if max_side <= self._max_input_size:
            return frame, 1.0
        scale = self._max_input_size / float(max_side)
        resized = cv2.resize(
            frame,
            (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_AREA,
        )
        return resized, scale

    def detect(self, frame: np.ndarray) -> Optional[PoseKeypoints]:
        if not self._is_loaded or self._holistic is None:
            return None

        t0 = time.perf_counter()
        try:
            proc_frame, scale = self._prepare_frame(frame)
            rgb = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2RGB)
            results = self._holistic.process(rgb)

            self.inference_ms = (time.perf_counter() - t0) * 1000.0
            self._fps.tick()

            pose_landmarks = getattr(results, "pose_landmarks", None)
            if pose_landmarks is None:
                return None

            return self._to_coco_pose(
                landmarks=pose_landmarks.landmark,
                frame_shape=frame.shape[:2],
                scale=scale,
            )
        except Exception as e:
            logger.error("Holistic inference error: %s", e)
            return None

    def _to_coco_pose(
        self,
        landmarks: List[Any],
        frame_shape: Tuple[int, int],
        scale: float,
    ) -> PoseKeypoints:
        h, w = frame_shape
        inv_scale = 1.0 / max(scale, 1e-6)

        keypoints: List[Keypoint] = []
        raw_xy = np.zeros((17, 2), dtype=np.float32)
        confs = np.zeros((17,), dtype=np.float32)

        for coco_idx in range(17):
            mp_idx = self.COCO_FROM_MEDIAPIPE[coco_idx]
            lm = landmarks[mp_idx]
            conf = float(getattr(lm, "visibility", 1.0))

            x = float(lm.x) * (w * scale) * inv_scale
            y = float(lm.y) * (h * scale) * inv_scale
            x = float(np.clip(x, 0.0, float(w - 1)))
            y = float(np.clip(y, 0.0, float(h - 1)))

            raw_xy[coco_idx] = [x, y]
            confs[coco_idx] = conf
            keypoints.append(Keypoint(x=x, y=y, confidence=conf))

        smoothed_xy = self._temporal_smooth(raw_xy, confs)
        smoothed_kps = [
            Keypoint(
                x=float(smoothed_xy[i, 0]),
                y=float(smoothed_xy[i, 1]),
                confidence=float(confs[i]),
            )
            for i in range(17)
        ]
        overall_conf = float(np.mean(confs))
        return PoseKeypoints(keypoints=smoothed_kps, confidence=overall_conf)

    def _temporal_smooth(self, raw_xy: np.ndarray, confs: np.ndarray) -> np.ndarray:
        if self._prev_keypoints is None:
            self._prev_keypoints = raw_xy.copy()
            return raw_xy

        smoothed = self._prev_keypoints.copy()
        visible = confs >= self.keypoint_conf
        for i in range(17):
            if visible[i]:
                smoothed[i] = smooth_array(
                    self._prev_keypoints[i],
                    raw_xy[i],
                    alpha=self.smooth_alpha,
                )
        self._prev_keypoints = smoothed
        return smoothed

    def reset_smoothing(self):
        self._prev_keypoints = None

    def draw_skeleton(self, frame: np.ndarray, pose: PoseKeypoints) -> np.ndarray:
        out = frame.copy()
        if pose is None:
            return out

        connections = [
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
            (5, 11), (6, 12), (11, 12),
            (11, 13), (13, 15), (12, 14), (14, 16),
            (0, 5), (0, 6),
        ]
        for i, j in connections:
            kp1 = pose.keypoints[i] if i < len(pose.keypoints) else None
            kp2 = pose.keypoints[j] if j < len(pose.keypoints) else None
            if kp1 and kp2 and kp1.valid and kp2.valid:
                cv2.line(out, kp1.to_tuple(), kp2.to_tuple(), (0, 255, 0), 2)

        for kp in pose.keypoints:
            if kp.valid:
                cv2.circle(out, kp.to_tuple(), 4, (255, 255, 0), -1)

        cv2.putText(
            out,
            f"Holistic FPS: {self._fps.fps:.0f} | Infer: {self.inference_ms:.0f}ms",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )
        return out

    @property
    def current_fps(self) -> float:
        return self._fps.fps

    def get_status(self) -> Dict[str, Any]:
        return {
            "loaded": self._is_loaded,
            "device": "cpu",
            "model": "mediapipe_holistic",
            "model_complexity": self.model_complexity,
            "input_size": self._max_input_size,
            "fps": self.current_fps,
            "inference_ms": self.inference_ms,
            "error": self._load_error,
        }

    def configure_runtime(self, imgsz: Optional[int] = None, use_half: Optional[bool] = None):
        if imgsz is not None:
            self._max_input_size = int(np.clip(int(imgsz), 256, 1280))
        _ = use_half


class AsyncPoseEngine:
    """Async wrapper for pose inference."""

    def __init__(self, engine: MediaPipeHolisticPoseEngine):
        self.engine = engine
        self._last_pose: Optional[PoseKeypoints] = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._latest_frame: Optional[np.ndarray] = None
        self._frame_lock = threading.Lock()

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._inference_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        self.engine.close()

    def submit_frame(self, frame: np.ndarray):
        with self._frame_lock:
            self._latest_frame = frame

    def get_latest_pose(self) -> Optional[PoseKeypoints]:
        with self._lock:
            return self._last_pose

    def _inference_loop(self):
        while self._running:
            with self._frame_lock:
                frame = self._latest_frame
            if frame is not None:
                pose = self.engine.detect(frame)
                with self._lock:
                    self._last_pose = pose
            time.sleep(0.001)


__all__ = ["MediaPipeHolisticPoseEngine", "AsyncPoseEngine"]
