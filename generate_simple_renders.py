"""
Simple dataset generator with basic rendering using trimesh + pyrender
Much simpler than BlenderProc, uses what we already have working
"""

import numpy as np
import json
from pathlib import Path
import cv2
from scipy.spatial.transform import Rotation
import argparse
import trimesh
import pyrender
import os

os.environ['PYOPENGL_PLATFORM'] = 'egl'


def generate_dataset(obj_path, output_dir, num_images=100):
    """Generate dataset with simple pyrender"""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create directories
    (output_dir / "rgb").mkdir(exist_ok=True)
    (output_dir / "mask_visib").mkdir(exist_ok=True)
    (output_dir / "depth").mkdir(exist_ok=True)

    print(f"="*60)
    print(f"Simple Dataset Generation")
    print(f"="*60)
    print(f"Object: {obj_path}")
    print(f"Output: {output_dir}")
    print(f"Images: {num_images}")
    print(f"="*60)

    # Load mesh
    print("\nLoading mesh...")
    mesh = trimesh.load(str(obj_path))
    obj_diameter = np.linalg.norm(mesh.bounds[1] - mesh.bounds[0])
    print(f"Object diameter: {obj_diameter:.2f} mm")

    # Camera params
    width, height = 640, 480
    fov = 50
    focal_length = width / (2 * np.tan(np.deg2rad(fov) / 2))
    K = np.array([
        [focal_length, 0, width / 2],
        [0, focal_length, height / 2],
        [0, 0, 1]
    ])

    # Pyrender setup
    scene = pyrender.Scene()
    camera = pyrender.PerspectiveCamera(yfov=np.deg2rad(fov))
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)

    # Mesh to pyrender
    mesh_pr = pyrender.Mesh.from_trimesh(mesh)

    # Renderer
    renderer = pyrender.OffscreenRenderer(width, height)

    # Storage
    scene_camera = {}
    scene_gt = {}
    scene_gt_info = {}

    print(f"\nGenerating {num_images} images...")

    for i in range(num_images):
        if i % 10 == 0:
            print(f"  {i}/{num_images}...")

        # Clear scene
        scene.clear()

        # Random pose for object
        R_obj = Rotation.random().as_matrix()
        t_obj = np.array([
            np.random.uniform(-50, 50),
            np.random.uniform(-50, 50),
            0
        ])

        obj_pose = np.eye(4)
        obj_pose[:3, :3] = R_obj
        obj_pose[:3, 3] = t_obj

        scene.add(mesh_pr, pose=obj_pose)

        # Random camera pose
        camera_distance = obj_diameter * np.random.uniform(1.5, 3.0)
        theta = np.random.uniform(0, 2 * np.pi)
        phi = np.random.uniform(np.deg2rad(20), np.deg2rad(80))

        cam_x = camera_distance * np.sin(phi) * np.cos(theta)
        cam_y = camera_distance * np.sin(phi) * np.sin(theta)
        cam_z = camera_distance * np.cos(phi)

        camera_pos = np.array([cam_x, cam_y, cam_z])

        # Look at object
        forward = -camera_pos / np.linalg.norm(camera_pos)
        right = np.cross(forward, np.array([0, 0, 1]))
        if np.linalg.norm(right) < 1e-6:
            right = np.cross(forward, np.array([0, 1, 0]))
        right = right / np.linalg.norm(right)
        up = np.cross(right, forward)

        cam_pose = np.eye(4)
        cam_pose[:3, 0] = right
        cam_pose[:3, 1] = up
        cam_pose[:3, 2] = -forward
        cam_pose[:3, 3] = camera_pos

        scene.add(camera, pose=cam_pose)
        scene.add(light, pose=cam_pose)

        # Render
        try:
            rgb, depth = renderer.render(scene)

            # Create mask
            mask = (depth > 0).astype(np.uint8) * 255

            # Save
            cv2.imwrite(str(output_dir / "rgb" / f"{i:06d}.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(output_dir / "mask_visib" / f"{i:06d}_000000.png"), mask)
            depth_mm = np.clip(depth * 1000, 0, 65535).astype(np.uint16)
            cv2.imwrite(str(output_dir / "depth" / f"{i:06d}.png"), depth_mm)

            # Compute object to camera transform
            world2cam = np.linalg.inv(cam_pose)
            obj2cam = world2cam @ obj_pose
            R = obj2cam[:3, :3]
            t = obj2cam[:3, 3]

            # Metadata
            scene_camera[str(i)] = {
                "cam_K": K.flatten().tolist(),
                "depth_scale": 1.0
            }

            scene_gt[str(i)] = [{
                "cam_R_m2c": R.flatten().tolist(),
                "cam_t_m2c": (t * 1000).tolist(),  # to mm
                "obj_id": 1
            }]

            # Bbox
            if mask.any():
                rows = np.any(mask, axis=1)
                cols = np.any(mask, axis=0)
                rmin, rmax = np.where(rows)[0][[0, -1]]
                cmin, cmax = np.where(cols)[0][[0, -1]]
                bbox = [int(cmin), int(rmin), int(cmax - cmin), int(rmax - rmin)]
                px_count = int((mask > 0).sum())
            else:
                bbox = [0, 0, 0, 0]
                px_count = 0

            scene_gt_info[str(i)] = [{
                "bbox_obj": bbox,
                "bbox_visib": bbox,
                "px_count_all": px_count,
                "px_count_valid": px_count,
                "px_count_visib": px_count,
                "visib_fract": 1.0 if px_count > 0 else 0.0
            }]

        except Exception as e:
            print(f"\nError rendering {i}: {e}")
            continue

    renderer.delete()

    # Save JSONs
    print("\nSaving metadata...")
    with open(output_dir / "scene_camera.json", 'w') as f:
        json.dump(scene_camera, f, indent=2)
    with open(output_dir / "scene_gt.json", 'w') as f:
        json.dump(scene_gt, f, indent=2)
    with open(output_dir / "scene_gt_info.json", 'w') as f:
        json.dump(scene_gt_info, f, indent=2)

    print(f"\n{'='*60}")
    print(f"✓ Dataset generated!")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--object-path', default='models/obj_000001.ply')
    parser.add_argument('--output-dir', default='chasis_dataset/train_real/000001')
    parser.add_argument('--num-images', type=int, default=100)
    args = parser.parse_args()

    generate_dataset(args.object_path, args.output_dir, args.num_images)
