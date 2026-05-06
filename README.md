# Virtual Try-On Studio (MediaPipe Holistic)

A real-time virtual try-on pipeline using **MediaPipe Holistic** for body tracking, with garment warping, parsing, DensePose-assisted torso masks, and layered rendering.

## What This Uses

- Pose tracking: `MediaPipe Holistic` (COCO-17 mapped internally)
- Parsing: SCHP parsing engine + geometric fallback
- Torso guidance: DensePose engine + keypoint fallback
- Garment fitting: hybrid warper + flow refinement
- Rendering: alpha blend + lighting + shadow adaptation
- UI: `CustomTkinter` desktop app

## Project Layout

```text
engine/
  mediapipe_holistic_pose.py   # MediaPipe Holistic pose engine
  render_pipeline.py           # Main orchestration pipeline
  parsing_engine.py            # Human parsing
  densepose_engine.py          # DensePose torso mapping
  hybrid_warper.py             # Garment warp engine
  shadow_engine.py             # Lighting/shadow adaptation
ui/
  app.py                       # Desktop UI
main.py                        # App entrypoint
test_single.py                 # Single-image smoke test
```

## Installation

1. Create/activate a Python environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Install PyTorch separately if you want GPU acceleration for supported modules:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## Run

```bash
python main.py
```

The app starts with MediaPipe Holistic as the pose tracker.

## Optional Smoke Test

```bash
python test_single.py
```

This writes `test_output.jpg` after processing `person.jpg`.

## Notes

- This repository is now **MediaPipe-only** for body tracking.
- YOLO pose model/configuration is removed from the active pipeline.
- If pose fails to initialize, verify `mediapipe` installed correctly in the same environment used to run the app.
