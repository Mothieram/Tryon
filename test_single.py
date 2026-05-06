import cv2
import time
import logging

from engine.render_pipeline import RenderPipeline


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("test_single")


pipeline = RenderPipeline(device="auto", enable_shadows=True)
logger.info("Loading models...")
if not pipeline.load_models():
    raise RuntimeError("Pipeline model loading failed. Check MediaPipe dependencies.")

flow = getattr(getattr(pipeline, "_warper", None), "_flow_warper", None)
if flow is not None:
    logger.info(
        "Geometric flow warper loaded | pyramid_levels=%s smooth_sigma=%.2f max_flow_dim=%s",
        getattr(flow, "pyramid_levels", "n/a"),
        float(getattr(flow, "smooth_sigma", 0.0)),
        getattr(flow, "_MAX_FLOW_DIM", "n/a"),
    )
else:
    logger.warning("Geometric flow warper instance not found on pipeline.")

pipeline.load_garments("assets/shirts")

frame = cv2.imread("person.jpg")
if frame is None:
    raise FileNotFoundError("Could not load person.jpg")

for i in range(10):
    out, stats = pipeline.process_frame(frame)
    time.sleep(0.05)
    if stats.pose_detected:
        logger.info("Pose detected on iteration %d", i + 1)
        break

out, stats = pipeline.process_frame(frame)
cv2.imwrite("test_output.jpg", out)
print(
    f"Warp: {stats.warp_ms:.1f}ms | FPS: {stats.fps:.1f} | "
    f"Pose: {stats.pose_detected}"
)
