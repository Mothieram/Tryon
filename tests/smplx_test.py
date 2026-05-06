# SMPL-X pose-tracked mesh overlay on video, driven by PIXIE
# Per frame: detect body bbox -> PIXIE.encode/decode -> render posed SMPL-X mesh -> warp into video

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
VIDEO_PATH = os.path.join(THIS_DIR, "input2.mp4")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CROP_SIZE = 224
HD_SIZE = 1024
SCALE = 1.1

# -------- LOAD PIXIE --------
pixie_cfg.model.use_tex = False
pixie = PIXIE(config=pixie_cfg, device=DEVICE)
faces = pixie.smplx.faces_tensor.cpu().numpy().astype(np.int32)
detector = detectors.FasterRCNN(device=DEVICE)

# -------- PYRENDER SETUP (renders into the 224 crop's NDC) --------
scene = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=[0.5, 0.5, 0.5])
camera = pyrender.OrthographicCamera(xmag=1.0, ymag=1.0, znear=0.01, zfar=100.0)
cam_pose = np.eye(4)
cam_pose[2, 3] = 10.0  # camera in front of origin, looking down -Z
scene.add(camera, pose=cam_pose)

light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
scene.add(light, pose=cam_pose)

renderer = pyrender.OffscreenRenderer(viewport_width=CROP_SIZE, viewport_height=CROP_SIZE)

# -------- VIDEO --------
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
out_W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
out_H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
OUT_PATH = os.path.join(THIS_DIR, "mesh_only.avi")
writer = cv2.VideoWriter(OUT_PATH, cv2.VideoWriter_fourcc(*"MJPG"), fps, (out_W, out_H))

# Temporal smoothing across frames to reduce per-frame jitter
prev_verts_render = None
EMA_ALPHA = 0.5  # 0 = freeze on first frame, 1 = no smoothing

while True:
    ret, frame = cap.read()
    if not ret:
        break
    H, W = frame.shape[:2]

    # PIXIE expects RGB float in [0, 1]
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    image_tensor = torch.tensor(image_rgb.transpose(2, 0, 1)).float()[None]

    # Detect body bbox
    bbox = detector.run(image_tensor)
    if bbox is None:
        blank = np.zeros((H, W, 3), dtype=np.uint8)
        writer.write(blank)
        cv2.imshow("PIXIE Mesh Only", blank)
        if cv2.waitKey(1) & 0xFF == 27:
            break
        continue

    left, top, right, bottom = bbox
    old_size = max(right - left, bottom - top)
    center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0])
    size = int(old_size * SCALE)
    src_pts = np.array([
        [center[0] - size / 2, center[1] - size / 2],
        [center[0] - size / 2, center[1] + size / 2],
        [center[0] + size / 2, center[1] - size / 2],
    ])

    # Crop to 224 (model input) and 1024 (PIXIE wants both)
    dst_pts = np.array([[0, 0], [0, CROP_SIZE - 1], [CROP_SIZE - 1, 0]])
    tform = estimate_transform("similarity", src_pts, dst_pts)
    crop_img = warp(image_rgb, tform.inverse, output_shape=(CROP_SIZE, CROP_SIZE))

    dst_pts_hd = np.array([[0, 0], [0, HD_SIZE - 1], [HD_SIZE - 1, 0]])
    tform_hd = estimate_transform("similarity", src_pts, dst_pts_hd)
    hd_img = warp(image_rgb, tform_hd.inverse, output_shape=(HD_SIZE, HD_SIZE))

    crop_tensor = torch.tensor(crop_img.transpose(2, 0, 1)).float()[None].to(DEVICE)
    hd_tensor = torch.tensor(hd_img.transpose(2, 0, 1)).float()[None].to(DEVICE)

    # PIXIE inference
    with torch.no_grad():
        data = {"body": {"image": crop_tensor, "image_hd": hd_tensor}}
        param_dict = pixie.encode(data, threthold=True, keep_local=True, copy_and_paste=False)
        opdict = pixie.decode(param_dict["body"], param_type="body")

    # transformed_vertices: (N, 3), x/y in [-1, 1] NDC of the 224 crop with image-Y-down
    verts = opdict["transformed_vertices"][0].cpu().numpy()
    verts_render = verts.copy()
    verts_render[:, 1] *= -1.0  # flip Y for OpenGL convention
    verts_render[:, 2] *= -1.0  # depth: closer = larger -> negative Z in pyrender

    # Temporal EMA smoothing on vertices
    if prev_verts_render is not None and prev_verts_render.shape == verts_render.shape:
        verts_render = EMA_ALPHA * verts_render + (1.0 - EMA_ALPHA) * prev_verts_render
    prev_verts_render = verts_render.copy()

    mesh = trimesh.Trimesh(vertices=verts_render, faces=faces, process=False)
    render_mesh = pyrender.Mesh.from_trimesh(mesh)

    for node in list(scene.mesh_nodes):
        scene.remove_node(node)
    scene.add(render_mesh)

    color, _ = renderer.render(scene)  # CROP_SIZE x CROP_SIZE RGB
    color = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)

    # Warp the crop-rendered mesh back into the original video frame
    M_inv = tform.inverse.params[:2]  # 2x3 affine: crop -> original
    rendered_full = cv2.warpAffine(color, M_inv, (W, H))

    writer.write(rendered_full)

    cv2.imshow("PIXIE Mesh Only", rendered_full)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
writer.release()
renderer.delete()
cv2.destroyAllWindows()
print(f"Saved mesh-only video to: {OUT_PATH}")
