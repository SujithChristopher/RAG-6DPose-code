"""
Simplified 6DOF Pose Estimation Visualization
Demonstrates pose estimation without full training
"""

import json
import numpy as np
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D

# Load data
DATASET_ROOT = Path("/media/sujith/Project/NOARK_CV/RAG-6DPose-code/chasis_dataset")
MODELS_PATH = Path("/media/sujith/Project/NOARK_CV/RAG-6DPose-code/models")

def load_cad_model():
    """Load CAD model point cloud"""
    import open3d as o3d
    ply_file = MODELS_PATH / "obj_000001.ply"
    pcd = o3d.io.read_point_cloud(str(ply_file))
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    return points, colors

def load_scene_data(scene_id='000001'):
    """Load scene ground truth poses"""
    scene_gt_file = DATASET_ROOT / "train_real" / scene_id / "scene_gt.json"
    scene_camera_file = DATASET_ROOT / "train_real" / scene_id / "scene_camera.json"

    with open(scene_gt_file) as f:
        scene_gt = json.load(f)
    with open(scene_camera_file) as f:
        scene_camera = json.load(f)

    return scene_gt, scene_camera

def rotation_matrix_to_euler(R):
    """Convert rotation matrix to Euler angles (radians)"""
    sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
    singular = sy < 1e-6

    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0

    return np.degrees(np.array([x, y, z]))

def generate_synthetic_poses(n_poses=10, seed=42):
    """Generate random 6DOF poses for demonstration"""
    np.random.seed(seed)

    poses = []
    for i in range(n_poses):
        # Random rotation (Euler angles)
        euler = np.random.uniform(-np.pi, np.pi, 3)

        # Convert to rotation matrix
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(euler[0]), -np.sin(euler[0])],
            [0, np.sin(euler[0]), np.cos(euler[0])]
        ])
        Ry = np.array([
            [np.cos(euler[1]), 0, np.sin(euler[1])],
            [0, 1, 0],
            [-np.sin(euler[1]), 0, np.cos(euler[1])]
        ])
        Rz = np.array([
            [np.cos(euler[2]), -np.sin(euler[2]), 0],
            [np.sin(euler[2]), np.cos(euler[2]), 0],
            [0, 0, 1]
        ])
        R = Rz @ Ry @ Rx

        # Random translation
        t = np.random.uniform([200, 200, 800], [600, 600, 1200], 3)

        # Random confidence
        conf = np.random.uniform(0.5, 0.99)

        poses.append({
            'id': i,
            'R': R,
            'euler': np.degrees(euler),
            't': t,
            'confidence': conf,
        })

    return poses

def project_3d_to_2d(points_3d, K, R, t):
    """Project 3D points to 2D using camera matrix K and pose (R, t)"""
    # Transform to camera frame
    points_cam = (R @ points_3d.T + t[:, None]).T

    # Project to image
    points_2d = (K @ points_cam.T).T
    points_2d = points_2d[:, :2] / points_2d[:, 2:3]

    return points_2d

def visualize_pose_3d(poses, cad_points, cad_colors):
    """Create 3D visualization of poses"""
    fig = plt.figure(figsize=(16, 10))

    # 3D CAD point cloud and poses
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    ax1.scatter(cad_points[:, 0], cad_points[:, 1], cad_points[:, 2],
               c=cad_colors, s=1, alpha=0.5)

    # Draw camera frames for first 5 poses
    colors_frame = ['red', 'green', 'blue']
    for pose in poses[:5]:
        R = pose['R']
        t = pose['t']

        # Draw axes
        scale = 100
        origin = t
        for j, color in enumerate(colors_frame):
            axis_vec = R[:, j] * scale
            end = origin + axis_vec
            ax1.plot([origin[0], end[0]],
                    [origin[1], end[1]],
                    [origin[2], end[2]],
                    color=color, linewidth=2)

        # Draw camera position
        ax1.scatter(*t, s=100, c='black', marker='o')

    ax1.set_xlabel('X (mm)')
    ax1.set_ylabel('Y (mm)')
    ax1.set_zlabel('Z (mm)')
    ax1.set_title('3D Pose Visualization\n(Red=X, Green=Y, Blue=Z axes)')

    # Rotation angles
    ax2 = fig.add_subplot(2, 2, 2)
    pose_ids = [p['id'] for p in poses]
    euler_angles = np.array([p['euler'] for p in poses])

    ax2.plot(pose_ids, euler_angles[:, 0], 'r-o', label='Roll (X)', alpha=0.7)
    ax2.plot(pose_ids, euler_angles[:, 1], 'g-s', label='Pitch (Y)', alpha=0.7)
    ax2.plot(pose_ids, euler_angles[:, 2], 'b-^', label='Yaw (Z)', alpha=0.7)
    ax2.set_xlabel('Pose ID')
    ax2.set_ylabel('Angle (degrees)')
    ax2.set_title('Rotation Angles')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Translation
    ax3 = fig.add_subplot(2, 2, 3)
    trans = np.array([p['t'] for p in poses])

    ax3.plot(pose_ids, trans[:, 0], 'r-o', label='X', alpha=0.7)
    ax3.plot(pose_ids, trans[:, 1], 'g-s', label='Y', alpha=0.7)
    ax3.plot(pose_ids, trans[:, 2], 'b-^', label='Z', alpha=0.7)
    ax3.set_xlabel('Pose ID')
    ax3.set_ylabel('Translation (mm)')
    ax3.set_title('Translation Vectors')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Confidence scores
    ax4 = fig.add_subplot(2, 2, 4)
    confidences = [p['confidence'] for p in poses]
    colors_conf = plt.cm.RdYlGn(np.array(confidences))

    bars = ax4.bar(pose_ids, confidences, color=colors_conf)
    ax4.set_xlabel('Pose ID')
    ax4.set_ylabel('Confidence')
    ax4.set_ylim([0, 1])
    ax4.set_title('Pose Estimation Confidence')
    ax4.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for i, (bar, conf) in enumerate(zip(bars, confidences)):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{conf:.2f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    return fig

def visualize_2d_projections(poses, cad_points, cad_colors, img_size=640):
    """Create 2D projection visualizations"""
    # Camera intrinsics (simplified)
    f = img_size / (2 * np.tan(np.deg2rad(30)))
    K = np.array([
        [f, 0, img_size/2],
        [0, f, img_size/2],
        [0, 0, 1]
    ])

    fig = plt.figure(figsize=(16, 12))

    for idx, pose in enumerate(poses[:9]):  # Show first 9 poses
        ax = fig.add_subplot(3, 3, idx + 1)

        # Create blank image
        img = np.ones((img_size, img_size, 3), dtype=np.uint8) * 255

        # Project CAD points
        R = pose['R']
        t = pose['t']

        try:
            points_2d = project_3d_to_2d(cad_points, K, R, t)

            # Filter points in image bounds
            mask = (points_2d[:, 0] >= 0) & (points_2d[:, 0] < img_size) & \
                   (points_2d[:, 1] >= 0) & (points_2d[:, 1] < img_size)

            if mask.any():
                valid_2d = points_2d[mask].astype(int)
                valid_colors = (cad_colors[mask] * 255).astype(np.uint8)

                for pt, color in zip(valid_2d, valid_colors):
                    cv2.circle(img, tuple(pt), 1, tuple(color.tolist()), -1)
        except:
            pass

        # Draw coordinate axes at origin (if visible)
        origin_2d = K @ np.array([0, 0, 1000])  # Point 1m along Z
        if origin_2d[2] > 0:
            origin_2d = origin_2d[:2] / origin_2d[2]
            if 0 <= origin_2d[0] < img_size and 0 <= origin_2d[1] < img_size:
                cv2.circle(img, tuple(origin_2d.astype(int)), 5, (0, 0, 0), -1)

        # Display
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax.imshow(img_rgb)
        ax.set_title(f"Pose {pose['id']}\nConf: {pose['confidence']:.2f}")
        ax.axis('off')

    plt.tight_layout()
    return fig

def load_poses_from_json(json_file):
    """Load poses from JSON file"""
    with open(json_file, 'r') as f:
        poses_data = json.load(f)

    # Convert list format to dict format if needed
    poses = []
    for p in poses_data:
        pose = {
            'id': p['id'],
            'R': np.array(p['R']),
            'euler': np.array(p['euler']),
            't': np.array(p['t']),
            'confidence': p['confidence'],
        }
        poses.append(pose)

    return poses

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Visualize 6DOF pose estimations')
    parser.add_argument('--poses', type=str, default=None, help='Path to JSON file with poses')
    parser.add_argument('--n-poses', type=int, default=10, help='Number of synthetic poses to generate (if --poses not provided)')
    args = parser.parse_args()

    print("="*60)
    print("6DOF Pose Estimation Visualization")
    print("="*60)

    # Load CAD model
    print("\n1. Loading CAD model...")
    try:
        cad_points, cad_colors = load_cad_model()
        print(f"   ✓ Loaded {len(cad_points)} points")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return

    # Load or generate poses
    if args.poses and Path(args.poses).exists():
        print(f"\n2. Loading poses from {args.poses}...")
        try:
            poses = load_poses_from_json(args.poses)
            print(f"   ✓ Loaded {len(poses)} poses")
        except Exception as e:
            print(f"   ✗ Error loading poses: {e}")
            return
    else:
        # Generate sample poses
        print(f"\n2. Generating {args.n_poses} sample 6DOF poses...")
        poses = generate_synthetic_poses(n_poses=args.n_poses)
        print(f"   ✓ Generated {len(poses)} poses")

    # Create 3D visualization
    print("\n3. Creating 3D pose visualization...")
    fig1 = visualize_pose_3d(poses, cad_points, cad_colors)
    output_file1 = Path("pose_3d_visualization.png")
    fig1.savefig(output_file1, dpi=150, bbox_inches='tight')
    print(f"   ✓ Saved to {output_file1}")

    # Create 2D projection visualization
    print("\n4. Creating 2D projection visualization...")
    fig2 = visualize_2d_projections(poses, cad_points, cad_colors)
    output_file2 = Path("pose_2d_projections.png")
    fig2.savefig(output_file2, dpi=150, bbox_inches='tight')
    print(f"   ✓ Saved to {output_file2}")

    # Print pose summary
    print("\n5. Pose Summary:")
    print(f"   {'ID':<4} {'Roll':<8} {'Pitch':<8} {'Yaw':<8} {'X':<8} {'Y':<8} {'Z':<8} {'Conf':<6}")
    print("   " + "-"*60)
    for pose in poses:
        euler = pose['euler']
        t = pose['t']
        print(f"   {pose['id']:<4} {euler[0]:>7.1f}° {euler[1]:>7.1f}° {euler[2]:>7.1f}° "
              f"{t[0]:>7.0f} {t[1]:>7.0f} {t[2]:>7.0f} {pose['confidence']:>5.2f}")

    print("\n" + "="*60)
    print("Visualization complete!")
    print("="*60)
    print(f"\nGenerated files:")
    print(f"  - {output_file1.resolve()}")
    print(f"  - {output_file2.resolve()}")

if __name__ == "__main__":
    main()
