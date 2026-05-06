# SMPL-X live overlay on webcam, driven by PIXIE
# Per frame: detect body bbox -> PIXIE.encode/decode -> render posed SMPL-X mesh -> overlay on camera feed

import os
import sys
import cv2
import torch
import numpy as np
import trimesh
import pyrender
from skimage.transform import estimate_transform, warp

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PIXIE_DIR = os.path.join(THIS_DIR, "..", "third_party", "PIXIE")
sys.path.insert(0, PIXIE_DIR)

from pixielib.pixie import PIXIE
from pixielib.utils.config import cfg as pixie_cfg
from pixielib.datasets import detectors

# -------- CONFIG --------
CAMERA_INDEX = 0          # default webcam; change if you have multiple cams
CAM_WIDTH = 1280          # request capture resolution from webcam
CAM_HEIGHT = 720
DISPLAY_WIDTH = 1280      # window display size (0 = use captured size as-is)
DISPLAY_HEIGHT = 720
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CROP_SIZE = 224
HD_SIZE = 1024
SCALE = 1.1
EMA_ALPHA = 0.5           # 0.3 = smoother/laggy, 0.8 = snappier/jittery
MESH_ALPHA = 0.85         # opacity of mesh on the overlay (0..1)
DETECT_EVERY = 1          # run pose estimation every N frames (raise for higher FPS)

# -------- LOAD PIXIE --------
pixie_cfg.model.use_tex = False
pixie = PIXIE(config=pixie_cfg, device=DEVICE)
faces = pixie.smplx.faces_tensor.cpu().numpy().astype(np.int32)
detector = detectors.FasterRCNN(device=DEVICE)

# -------- PYRENDER SETUP --------
scene = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=[0.5, 0.5, 0.5])
camera = pyrender.OrthographicCamera(xmag=1.0, ymag=1.0, znear=0.01, zfar=100.0)
cam_pose = np.eye(4)
cam_pose[2, 3] = 10.0
scene.add(camera, pose=cam_pose)

light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
scene.add(light, pose=cam_pose)

renderer = pyrender.OffscreenRenderer(viewport_width=CROP_SIZE, viewport_height=CROP_SIZE)

# -------- WEBCAM --------
print(f"Opening camera index {CAMERA_INDEX}...")
cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
if not cap.isOpened():
    raise RuntimeError(f"Could not open camera index {CAMERA_INDEX}")
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
print(f"Camera opened. Resolution: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
cv2.namedWindow("PIXIE Live - SMPL-X Overlay", cv2.WINDOW_NORMAL)
cv2.resizeWindow("PIXIE Live - SMPL-X Overlay", DISPLAY_WIDTH, DISPLAY_HEIGHT)
print(f"Device: {DEVICE} | Detect every {DETECT_EVERY} frames")

prev_verts_render = None
last_rendered_full = None
last_mask = None
frame_idx = 0

print("Press ESC or 'q' to quit.")
while True:
    ret, frame = cap.read()
    if not ret:
        break
    H, W = frame.shape[:2]
    frame_idx += 1

    # Run PIXIE every Nth frame; reuse the last mesh between predictions
    if frame_idx % DETECT_EVERY == 0:
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        image_tensor = torch.tensor(image_rgb.transpose(2, 0, 1)).float()[None]

        bbox = detector.run(image_tensor)
        if bbox is None:
            print(f"frame {frame_idx}: no person detected")
        if bbox is not None:
            left, top, right, bottom = bbox
            old_size = max(right - left, bottom - top)
            center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0])
            size = int(old_size * SCALE)
            src_pts = np.array([
                [center[0] - size / 2, center[1] - size / 2],
                [center[0] - size / 2, center[1] + size / 2],
                [center[0] + size / 2, center[1] - size / 2],
            ])

            dst_pts = np.array([[0, 0], [0, CROP_SIZE - 1], [CROP_SIZE - 1, 0]])
            tform = estimate_transform("similarity", src_pts, dst_pts)
            crop_img = warp(image_rgb, tform.inverse, output_shape=(CROP_SIZE, CROP_SIZE))

            dst_pts_hd = np.array([[0, 0], [0, HD_SIZE - 1], [HD_SIZE - 1, 0]])
            tform_hd = estimate_transform("similarity", src_pts, dst_pts_hd)
            hd_img = warp(image_rgb, tform_hd.inverse, output_shape=(HD_SIZE, HD_SIZE))

            crop_tensor = torch.tensor(crop_img.transpose(2, 0, 1)).float()[None].to(DEVICE)
            hd_tensor = torch.tensor(hd_img.transpose(2, 0, 1)).float()[None].to(DEVICE)

            with torch.no_grad():
                data = {"body": {"image": crop_tensor, "image_hd": hd_tensor}}
                param_dict = pixie.encode(data, threthold=True, keep_local=True, copy_and_paste=False)
                opdict = pixie.decode(param_dict["body"], param_type="body")

            verts = opdict["transformed_vertices"][0].cpu().numpy()
            verts_render = verts.copy()
            verts_render[:, 1] *= -1.0
            verts_render[:, 2] *= -1.0

            if prev_verts_render is not None and prev_verts_render.shape == verts_render.shape:
                verts_render = EMA_ALPHA * verts_render + (1.0 - EMA_ALPHA) * prev_verts_render
            prev_verts_render = verts_render.copy()

            mesh = trimesh.Trimesh(vertices=verts_render, faces=faces, process=False)
            render_mesh = pyrender.Mesh.from_trimesh(mesh)
            for node in list(scene.mesh_nodes):
                scene.remove_node(node)
            scene.add(render_mesh)

            color, _ = renderer.render(scene)
            color = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)

            M_inv = tform.inverse.params[:2]
            last_rendered_full = cv2.warpAffine(color, M_inv, (W, H))
            last_mask = (last_rendered_full.sum(axis=2) > 0).astype(np.float32)[..., None]

    # Composite: mesh only where it rendered, original frame elsewhere
    if last_rendered_full is not None and last_mask is not None:
        overlay = (
            frame * (1 - last_mask * MESH_ALPHA)
            + last_rendered_full * (last_mask * MESH_ALPHA)
        ).astype(np.uint8)
    else:
        overlay = frame

    if DISPLAY_WIDTH and DISPLAY_HEIGHT and (overlay.shape[1], overlay.shape[0]) != (DISPLAY_WIDTH, DISPLAY_HEIGHT):
        overlay = cv2.resize(overlay, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
    cv2.imshow("PIXIE Live - SMPL-X Overlay", overlay)
    key = cv2.waitKey(1) & 0xFF
    if key == 27 or key == ord("q"):
        break

cap.release()
renderer.delete()
cv2.destroyAllWindows()
