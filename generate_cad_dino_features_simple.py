"""
Simplified DINO feature generation for CAD model using direct mesh rendering

This version uses Trimesh + pyrender for more reliable rendering
"""

import numpy as np
import torch
import open3d as o3d
import trimesh
from pathlib import Path
import argparse
from tqdm import tqdm
from collections import defaultdict
import json
import warnings
warnings.filterwarnings('ignore')

# For image processing
from PIL import Image
import torchvision.transforms as transforms

# Try to import pyrender
try:
    import pyrender
    import os
    os.environ['PYOPENGL_PLATFORM'] = 'egl'  # For headless rendering
    HAS_PYRENDER = True
except ImportError:
    HAS_PYRENDER = False
    print("Warning: pyrender not available, will use simplified approach")


def load_dino_model(device='cuda'):
    """Load DINOv2 ViT-B/14 model (768-dim features)"""
    print(f"Loading DINO model (dinov2_vitb14)...")
    print(f"Device: {device}")

    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
    model = model.to(device)
    model.eval()

    print(f"DINO model loaded successfully (feature dim: {model.embed_dim})")
    return model


def create_sphere_viewpoints(n_views=100, distance=2.0):
    """Generate camera viewpoints on a sphere"""
    poses = []
    phi = np.pi * (3. - np.sqrt(5.))  # Golden angle

    for i in range(n_views):
        y = 1 - (i / float(n_views - 1)) * 2
        radius = np.sqrt(1 - y * y)
        theta = phi * i

        x = np.cos(theta) * radius
        z = np.sin(theta) * radius

        eye = np.array([x, y, z]) * distance
        forward = -eye / np.linalg.norm(eye)
        right = np.cross(forward, np.array([0, 0, 1]))
        if np.linalg.norm(right) < 1e-6:
            right = np.cross(forward, np.array([0, 1, 0]))
        right = right / np.linalg.norm(right)
        up = np.cross(right, forward)

        pose = np.eye(4)
        pose[:3, 0] = right
        pose[:3, 1] = up
        pose[:3, 2] = -forward
        pose[:3, 3] = eye

        poses.append(pose)

    return poses


def render_view_pyrender(mesh, camera_pose, width=224, height=224, fov=60):
    """Render using pyrender"""
    scene = pyrender.Scene()

    # Add mesh
    mesh_pr = pyrender.Mesh.from_trimesh(mesh)
    scene.add(mesh_pr)

    # Add camera
    camera = pyrender.PerspectiveCamera(yfov=np.deg2rad(fov))
    scene.add(camera, pose=camera_pose)

    # Add light
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
    scene.add(light, pose=camera_pose)

    # Render
    renderer = pyrender.OffscreenRenderer(width, height)
    rgb, depth = renderer.render(scene)
    renderer.delete()

    return rgb, depth


def extract_dino_features(dino_model, rgb_image, device='cuda'):
    """Extract DINO features from RGB image"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    pil_image = Image.fromarray(rgb_image)
    image_tensor = transform(pil_image).unsqueeze(0).to(device)

    with torch.no_grad():
        features = dino_model.forward_features(image_tensor)
        features = features['x_norm_patchtokens']

        patch_h = patch_w = int(np.sqrt(features.shape[1]))
        features = features.reshape(1, patch_h, patch_w, -1)
        features = features.permute(0, 3, 1, 2)

    return features, (patch_h, patch_w)


def project_and_interpolate(points, camera_pose, features, feat_res, img_size, depth_map):
    """
    Project 3D points to image, check visibility, and interpolate features

    Returns:
        point_features: dict mapping point_idx -> list of features
    """
    # Camera intrinsics (FOV=60deg)
    fov = 60
    f = img_size / (2 * np.tan(np.deg2rad(fov) / 2))
    cx, cy = img_size / 2, img_size / 2

    # Transform to camera frame
    R = camera_pose[:3, :3].T
    t = camera_pose[:3, 3]
    points_cam = (points - t) @ R

    # Project to image
    u = f * points_cam[:, 0] / (points_cam[:, 2] + 1e-8) + cx
    v = f * points_cam[:, 1] / (points_cam[:, 2] + 1e-8) + cy

    # Check bounds and depth
    valid = (points_cam[:, 2] > 0) & \
            (u >= 0) & (u < img_size) & \
            (v >= 0) & (v < img_size)

    point_features = {}

    if not valid.any():
        return point_features

    valid_indices = np.where(valid)[0]
    u_valid = u[valid]
    v_valid = v[valid]
    depth_valid = points_cam[valid, 2]

    # Check depth visibility
    u_int = np.clip(u_valid.astype(int), 0, img_size - 1)
    v_int = np.clip(v_valid.astype(int), 0, img_size - 1)
    rendered_depth = depth_map[v_int, u_int]

    # PyRender depth is in same units as mesh (mm in our case)
    # Use relative tolerance to handle scale differences
    depth_diff = np.abs(depth_valid - rendered_depth)
    depth_threshold = np.maximum(depth_valid * 0.05, 10.0)  # 5% or 10mm tolerance
    visible = (rendered_depth > 0) & (depth_diff < depth_threshold)

    if not visible.any():
        return point_features

    visible_indices = valid_indices[visible]
    u_vis = u_valid[visible]
    v_vis = v_valid[visible]

    # Scale to feature map coordinates
    feat_h, feat_w = feat_res
    u_feat = u_vis * (feat_w / img_size)
    v_feat = v_vis * (feat_h / img_size)

    # Interpolate features using grid_sample
    device = features.device
    u_norm = (u_feat / feat_w) * 2 - 1
    v_norm = (v_feat / feat_h) * 2 - 1

    grid = torch.from_numpy(np.stack([u_norm, v_norm], axis=1)).float()
    grid = grid.unsqueeze(0).unsqueeze(0).to(device)

    interpolated = torch.nn.functional.grid_sample(
        features, grid, mode='bilinear', padding_mode='zeros', align_corners=True
    )

    interpolated = interpolated.squeeze().permute(1, 0).cpu().numpy()

    # Store features
    for i, pt_idx in enumerate(visible_indices):
        point_features[int(pt_idx)] = interpolated[i]

    return point_features


def generate_features_simple(cad_path, output_path, n_views=100, device='cuda'):
    """
    Generate DINO features with simplified rendering
    """
    print("="*60)
    print("CAD DINO Feature Generation (Simplified)")
    print("="*60)

    # Load mesh and point cloud
    print(f"\n1. Loading CAD model: {cad_path}")
    mesh = trimesh.load_mesh(str(cad_path))
    pcd = o3d.io.read_point_cloud(str(cad_path))
    points = np.asarray(pcd.points)
    n_points = len(points)

    print(f"   Points: {n_points}")
    print(f"   Triangles: {len(mesh.faces)}")

    # Center and scale mesh
    bbox_size = np.linalg.norm(mesh.bounds[1] - mesh.bounds[0])
    camera_distance = bbox_size * 1.5

    print(f"   Bounding box size: {bbox_size:.2f}")
    print(f"   Camera distance: {camera_distance:.2f}")

    # Load DINO
    print(f"\n2. Loading DINO model...")
    dino_model = load_dino_model(device)

    # Generate viewpoints
    print(f"\n3. Generating {n_views} viewpoints...")
    camera_poses = create_sphere_viewpoints(n_views, camera_distance)

    # Render and extract
    print(f"\n4. Rendering views and extracting features...")
    img_size = 224

    if not HAS_PYRENDER:
        print("ERROR: pyrender is required but not installed!")
        print("Install with: pip install pyrender")
        return None

    point_features_all = defaultdict(list)

    for view_idx, camera_pose in enumerate(tqdm(camera_poses, desc="Processing views")):
        try:
            # Render
            rgb, depth = render_view_pyrender(mesh, camera_pose, img_size, img_size)

            # Extract DINO
            features, feat_res = extract_dino_features(dino_model, rgb, device)

            # Project and interpolate
            pt_feats = project_and_interpolate(
                points, camera_pose, features, feat_res, img_size, depth
            )

            # Accumulate
            for pt_idx, feat in pt_feats.items():
                point_features_all[pt_idx].append(feat)

        except Exception as e:
            print(f"\nWarning: View {view_idx} failed: {e}")
            continue

    # Aggregate
    print(f"\n5. Aggregating features...")
    feature_dim = 768
    final_features = np.zeros((n_points, feature_dim), dtype=np.float32)

    points_with_feats = 0
    for pt_idx in range(n_points):
        if pt_idx in point_features_all and len(point_features_all[pt_idx]) > 0:
            final_features[pt_idx] = np.mean(point_features_all[pt_idx], axis=0)
            points_with_feats += 1

    print(f"   Points with features: {points_with_feats}/{n_points} ({points_with_feats/n_points*100:.1f}%)")

    # Fill missing with nearest neighbor
    if points_with_feats < n_points:
        print(f"\n6. Filling {n_points - points_with_feats} missing features...")
        from scipy.spatial import KDTree

        has_feat = np.array([i in point_features_all for i in range(n_points)])
        tree = KDTree(points[has_feat])
        _, indices = tree.query(points[~has_feat])
        final_features[~has_feat] = final_features[has_feat][indices]

    # Save
    print(f"\n7. Saving features to: {output_path}")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    features_tensor = torch.from_numpy(final_features)
    torch.save(features_tensor, output_path)

    print(f"   Shape: {features_tensor.shape}")
    print(f"   Size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")

    # Save metadata
    metadata = {
        'n_points': n_points,
        'feature_dim': feature_dim,
        'n_views': n_views,
        'points_with_features': points_with_feats,
    }

    metadata_path = output_path.with_suffix('.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n{'='*60}")
    print("Feature generation completed successfully!")
    print(f"{'='*60}")

    return features_tensor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('cad_path', help='Path to PLY file')
    parser.add_argument('--output', default='cad_features/obj_000001_dino_feat.pt')
    parser.add_argument('--n-views', type=int, default=100)
    parser.add_argument('--device', default='cuda')

    args = parser.parse_args()

    if args.device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA not available, using CPU")
        args.device = 'cpu'

    generate_features_simple(
        args.cad_path,
        args.output,
        args.n_views,
        args.device
    )


if __name__ == '__main__':
    main()
