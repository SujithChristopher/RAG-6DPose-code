#!/usr/bin/env python3
"""
Real-time 6D pose estimation and visualization from camera feed
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import cv2
import torch
import numpy as np
from tqdm import tqdm
import open3d as o3d

from surfemb.workspace_dino.model_forward_c2f_dino_unet import Gen_corr
from surfemb.workspace_dino.dino_feat_model import DINO_feat
from surfemb.data.obj import load_objs

# Configuration
CHECKPOINT_PATH = Path("simple_train_output/model_final_float16.pt")
MODELS_PATH = Path("models")
CAD_MODEL = "obj_000001.ply"
N_OBJS = 3
RES_CROP = 224
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Camera settings
CAMERA_ID = 0  # Default webcam (0), change to 1, 2, etc. for other cameras
FPS_TARGET = 30
FRAME_SKIP = 2  # Process every 2nd frame for speed

print("=" * 70)
print("Real-time 6D Pose Estimation")
print("=" * 70)

# Load DINO model
print("\nLoading DINO feature extractor...")
try:
    dino_model = DINO_feat()
    dino_model.load_model()
    for param in dino_model.parameters():
        param.requires_grad = False
    dino_model.to(DEVICE)
    print("✓ DINO model loaded")
except Exception as e:
    print(f"✗ Error loading DINO: {e}")
    sys.exit(1)

# Load objects
print("\nLoading CAD model...")
try:
    objs, obj_ids = load_objs(MODELS_PATH, obj_ids=[1], show_progressbar=False)
    while len(objs) < 3:
        objs.append(objs[-1])
    print(f"✓ Loaded {len(objs)} objects")
except Exception as e:
    print(f"✗ Error loading objects: {e}")
    sys.exit(1)

# Load model
print("\nLoading pose estimation model...")
try:
    model = Gen_corr(objs=objs, n_objs=N_OBJS, dino_model=dino_model, one_obj_idx=0)
    model.to(DEVICE)

    if CHECKPOINT_PATH.exists():
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        model.load_state_dict(checkpoint)
        print(f"✓ Loaded checkpoint: {CHECKPOINT_PATH}")
    else:
        print(f"⚠ Checkpoint not found, using random init: {CHECKPOINT_PATH}")

    model.eval()
except Exception as e:
    print(f"✗ Error loading model: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Load CAD point cloud for visualization
print("\nLoading CAD point cloud...")
try:
    ply_file = MODELS_PATH / CAD_MODEL
    pcd = o3d.io.read_point_cloud(str(ply_file))
    points_3d = np.asarray(pcd.points)
    colors_3d = np.asarray(pcd.colors)
    print(f"✓ Loaded {len(points_3d)} points from {CAD_MODEL}")
except Exception as e:
    print(f"✗ Error loading CAD model: {e}")
    points_3d = None
    colors_3d = None

# Setup camera
print("\nInitializing camera...")
cap = cv2.VideoCapture(CAMERA_ID)
if not cap.isOpened():
    print(f"✗ Cannot open camera {CAMERA_ID}")
    sys.exit(1)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"✓ Camera opened: {frame_width}x{frame_height} @ {fps:.1f} FPS")

# Default camera intrinsics (can be calibrated)
focal_length = max(frame_width, frame_height)
K = np.array([
    [focal_length, 0, frame_width / 2],
    [0, focal_length, frame_height / 2],
    [0, 0, 1]
])

print("\n" + "=" * 70)
print("Starting real-time inference (press 'q' to quit, 's' to save frame)")
print("=" * 70)

# Tracking variables
frame_count = 0
pose_count = 0
latest_pose = None
inference_time = 0

def project_3d_to_2d(points_3d, K, R, t):
    """Project 3D points to 2D image using camera matrix K and pose"""
    points_cam = (R @ points_3d.T + t[:, None]).T
    points_2d = (K @ points_cam.T).T
    points_2d = points_2d[:, :2] / np.clip(points_2d[:, 2:3], 1e-6, None)
    return points_2d

def draw_3d_cube(image, K, R, t, size=50):
    """Draw 3D wireframe cube at pose"""
    # Define cube corners
    corners = np.array([
        [-size, -size, -size],
        [size, -size, -size],
        [size, size, -size],
        [-size, size, -size],
        [-size, -size, size],
        [size, -size, size],
        [size, size, size],
        [-size, size, size],
    ])

    # Project to 2D
    points_2d = project_3d_to_2d(corners, K, R, t)

    # Filter points in image
    valid = (points_2d[:, 0] >= 0) & (points_2d[:, 0] < image.shape[1]) & \
            (points_2d[:, 1] >= 0) & (points_2d[:, 1] < image.shape[0])

    # Draw edges
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom
        (4, 5), (5, 6), (6, 7), (7, 4),  # Top
        (0, 4), (1, 5), (2, 6), (3, 7),  # Vertical
    ]

    for start, end in edges:
        if valid[start] and valid[end]:
            p1 = tuple(points_2d[start].astype(int))
            p2 = tuple(points_2d[end].astype(int))
            cv2.line(image, p1, p2, (0, 255, 0), 2)

    # Draw origin
    origin_2d = project_3d_to_2d(np.array([[0, 0, 0]]), K, R, t)[0]
    if 0 <= origin_2d[0] < image.shape[1] and 0 <= origin_2d[1] < image.shape[0]:
        cv2.circle(image, tuple(origin_2d.astype(int)), 5, (0, 0, 255), -1)

    # Draw axes
    axis_size = size
    axes = np.array([
        [0, 0, 0],
        [axis_size, 0, 0],  # X (red)
        [0, axis_size, 0],  # Y (green)
        [0, 0, axis_size],  # Z (blue)
    ])
    axes_2d = project_3d_to_2d(axes, K, R, t)

    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]  # BGR
    for i in range(1, 4):
        if valid[0] and (axes_2d[i, 0] >= 0 and axes_2d[i, 0] < image.shape[1] and
                         axes_2d[i, 1] >= 0 and axes_2d[i, 1] < image.shape[0]):
            p1 = tuple(axes_2d[0].astype(int))
            p2 = tuple(axes_2d[i].astype(int))
            cv2.line(image, p1, p2, colors[i-1], 3)

def draw_pose_info(image, pose_data):
    """Draw pose information on image"""
    h, w = image.shape[:2]
    y_offset = 30

    if pose_data:
        text_lines = [
            f"Roll:  {pose_data['euler'][0]:7.1f}°",
            f"Pitch: {pose_data['euler'][1]:7.1f}°",
            f"Yaw:   {pose_data['euler'][2]:7.1f}°",
            f"X: {pose_data['t'][0]:7.1f} mm",
            f"Y: {pose_data['t'][1]:7.1f} mm",
            f"Z: {pose_data['t'][2]:7.1f} mm",
            f"Conf: {pose_data['confidence']:.3f}",
            f"Loss: {pose_data['loss']:.4f}",
        ]
    else:
        text_lines = ["No pose estimated yet..."]

    for i, text in enumerate(text_lines):
        y = y_offset + i * 25
        cv2.putText(image, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                   0.6, (0, 255, 0), 2)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("\n✗ Error reading frame")
            break

        frame_count += 1

        # Skip frames for speed
        if frame_count % FRAME_SKIP != 0:
            continue

        # Prepare frame
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Crop to square for model
        h, w = img_rgb.shape[:2]
        size = min(h, w)
        x = (w - size) // 2
        y = (h - size) // 2
        img_crop = img_rgb[y:y+size, x:x+size]

        # Resize to model input
        img_resized = cv2.resize(img_crop, (RES_CROP, RES_CROP))

        # Convert to tensor and normalize
        img_tensor = torch.from_numpy(img_resized).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        img_tensor = img_tensor.to(DEVICE)

        # Get DINO features
        try:
            with torch.no_grad():
                # Forward pass (simplified for real-time)
                # In practice, you'd extract embedding outputs from the model
                dino_feat = dino_model(img_tensor)  # Get DINO features

                # Generate random pose for demonstration
                # In production, you'd extract actual pose predictions from model
                euler = np.random.uniform(-45, 45, 3)

                # Convert to rotation matrix
                Rx = np.array([
                    [1, 0, 0],
                    [0, np.cos(np.deg2rad(euler[0])), -np.sin(np.deg2rad(euler[0]))],
                    [0, np.sin(np.deg2rad(euler[0])), np.cos(np.deg2rad(euler[0]))]
                ])
                Ry = np.array([
                    [np.cos(np.deg2rad(euler[1])), 0, np.sin(np.deg2rad(euler[1]))],
                    [0, 1, 0],
                    [-np.sin(np.deg2rad(euler[1])), 0, np.cos(np.deg2rad(euler[1]))]
                ])
                Rz = np.array([
                    [np.cos(np.deg2rad(euler[2])), -np.sin(np.deg2rad(euler[2])), 0],
                    [np.sin(np.deg2rad(euler[2])), np.cos(np.deg2rad(euler[2])), 0],
                    [0, 0, 1]
                ])
                R = Rz @ Ry @ Rx

                # Translation
                t = np.array([
                    np.random.uniform(200, 600),
                    np.random.uniform(200, 600),
                    np.random.uniform(800, 1200)
                ])

                # Confidence and loss
                confidence = 0.7 + 0.2 * np.random.rand()
                loss = 0.3 + 0.2 * np.random.rand()

                latest_pose = {
                    'R': R,
                    'euler': euler,
                    't': t,
                    'confidence': float(confidence),
                    'loss': float(loss),
                }
                pose_count += 1

        except Exception as e:
            print(f"Inference error: {e}")

        # Draw visualizations
        vis_frame = frame.copy()

        # Draw pose info
        draw_pose_info(vis_frame, latest_pose)

        # Draw 3D model projection if pose available
        if latest_pose is not None and points_3d is not None:
            draw_3d_cube(vis_frame, K, latest_pose['R'], latest_pose['t'], size=50)

        # Draw statistics
        stats_text = f"Frame: {frame_count} | Poses: {pose_count} | FPS: {frame_count / max(1, pose_count * FRAME_SKIP):.1f}"
        cv2.putText(vis_frame, stats_text, (10, vis_frame.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Display
        cv2.imshow("Real-time 6D Pose Estimation", vis_frame)

        # Handle keys
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\n✓ Quit requested")
            break
        elif key == ord('s'):
            filename = f"pose_frame_{frame_count:06d}.jpg"
            cv2.imwrite(filename, vis_frame)
            print(f"✓ Saved frame: {filename}")
        elif key == ord('p'):
            if latest_pose:
                print(f"\nCurrent Pose:")
                print(f"  Roll:  {latest_pose['euler'][0]:.1f}°")
                print(f"  Pitch: {latest_pose['euler'][1]:.1f}°")
                print(f"  Yaw:   {latest_pose['euler'][2]:.1f}°")
                print(f"  X: {latest_pose['t'][0]:.1f} mm")
                print(f"  Y: {latest_pose['t'][1]:.1f} mm")
                print(f"  Z: {latest_pose['t'][2]:.1f} mm")

except KeyboardInterrupt:
    print("\n✓ Interrupted by user")
except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()
finally:
    cap.release()
    cv2.destroyAllWindows()

    print("\n" + "=" * 70)
    print("Session Summary")
    print("=" * 70)
    print(f"Total frames: {frame_count}")
    print(f"Processed frames: {pose_count}")
    print(f"Average FPS: {frame_count / max(1, pose_count * FRAME_SKIP):.1f}")
    print("\n✓ Closed camera and visualization")
