"""
main.py - Virtual Try-On System Entry Point
Run this file to start the application.
"""

import sys
import os
import logging
from pathlib import Path

# Ensure project root is on path
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from engine.coreutils import setup_logger
from engine.render_pipeline import RenderPipeline

logger = setup_logger("main", level=logging.INFO)


def main():
    """Main entry point - initialize pipeline and launch GUI."""
    logger.info("=" * 60)
    logger.info("  Virtual Try-On Studio  v1.0")
    logger.info("  AI-Powered Shirt Overlay System")
    logger.info("=" * 60)

    # ── Initialize Pipeline ─────────────────────────────────
    pipeline = RenderPipeline(
        device="auto",                        # CUDA if available, else CPU
        target_fps=30,
        enable_shadows=True,
        enable_lighting=True,
        opacity=0.95,
    )

    # ── Load AI Models ──────────────────────────────────────
    logger.info("Loading AI models...")
    if not pipeline.load_models():
        logger.error("Model loading failed. Check internet connection.")
        sys.exit(1)

    # ── Load Garments ───────────────────────────────────────
    shirts_dir = str(ROOT / "assets" / "shirts")
    count = pipeline.load_garments(shirts_dir)
    logger.info(f"Loaded {count} garment(s)")

    # ── Launch GUI ──────────────────────────────────────────
    try:
        from ui.app import run_app
        logger.info("Launching GUI...")
        run_app(pipeline)
    except ImportError as e:
        logger.error(f"GUI import failed: {e}")
        logger.info("Running headless demo mode instead...")
        _headless_demo(pipeline)
    except Exception as e:
        logger.error(f"Application error: {e}")
        raise


def _headless_demo(pipeline: RenderPipeline):
    """Headless OpenCV window demo (no GUI library needed)."""
    import cv2

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("No camera available for demo")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    logger.info("OpenCV demo mode - Press Q to quit, N for next shirt, P for previous")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        processed, stats = pipeline.process_frame(frame)

        # Overlay stats
        cv2.putText(processed, f"FPS: {stats.fps:.0f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(processed, f"Shirt: {stats.active_shirt}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(processed, f"Pose: {'✓' if stats.pose_detected else '✗'}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 0) if stats.pose_detected else (0, 0, 255), 2)
        cv2.putText(processed, "N=Next  P=Prev  S=Screenshot  Q=Quit",
                    (10, processed.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow("Virtual Try-On Studio", processed)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('n'):
            pipeline.next_shirt()
        elif key == ord('p'):
            pipeline.previous_shirt()
        elif key == ord('s'):
            path = pipeline.take_screenshot(processed)
            logger.info(f"Saved: {path}")

    cap.release()
    cv2.destroyAllWindows()
    pipeline.unload_models()


if __name__ == "__main__":
    main()
