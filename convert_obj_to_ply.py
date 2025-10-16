"""
Convert OBJ CAD model to PLY format required by RAG-6DPose

This script:
1. Loads an OBJ file (or other 3D formats)
2. Adds vertex colors if missing
3. Exports to PLY format with proper naming convention
4. Calculates model diameter
5. Creates models_info.json
"""

import trimesh
import open3d as o3d
import numpy as np
import json
from pathlib import Path
import argparse


def compute_model_diameter(vertices):
    """
    Compute the diameter of the model (max distance between any two vertices).
    Uses approximation for speed on large meshes.
    """
    print("Computing model diameter...")

    # For very large meshes, sample points to speed up computation
    if len(vertices) > 10000:
        print(f"  Large mesh detected ({len(vertices)} vertices), sampling 5000 points for diameter calculation...")
        indices = np.random.choice(len(vertices), size=min(5000, len(vertices)), replace=False)
        sampled_vertices = vertices[indices]
    else:
        sampled_vertices = vertices

    # Compute pairwise distances (vectorized for speed)
    from scipy.spatial.distance import pdist
    distances = pdist(sampled_vertices)
    max_distance = distances.max()

    print(f"  Model diameter: {max_distance:.2f} mm")
    return float(max_distance)


def convert_obj_to_ply(input_path, output_dir, obj_id=1, add_colors=True, color=[128, 128, 128]):
    """
    Convert OBJ (or other format) to PLY with proper formatting.

    Args:
        input_path: Path to input CAD model
        output_dir: Output directory for PLY file
        obj_id: Object ID number (for naming)
        add_colors: Whether to add default colors if missing
        color: Default RGB color [R, G, B] in range [0, 255]
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from: {input_path}")

    # Load mesh using trimesh
    mesh = trimesh.load_mesh(str(input_path))

    print(f"  Vertices: {len(mesh.vertices)}")
    print(f"  Faces: {len(mesh.faces)}")

    # Check if mesh has vertex colors
    has_colors = hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None

    if not has_colors and add_colors:
        print(f"  No vertex colors found, adding default color: RGB{color}")
        # Add vertex colors (RGBA format)
        vertex_colors = np.ones((len(mesh.vertices), 4), dtype=np.uint8) * 255
        vertex_colors[:, :3] = color
        mesh.visual.vertex_colors = vertex_colors
    elif has_colors:
        print(f"  Vertex colors found!")

    # Export to PLY with proper naming
    output_filename = f"obj_{obj_id:06d}.ply"
    output_path = output_dir / output_filename

    print(f"Exporting to: {output_path}")
    mesh.export(str(output_path))

    # Verify the exported file can be read by Open3D (requirement for the codebase)
    print("Verifying PLY file with Open3D...")
    pcd = o3d.io.read_point_cloud(str(output_path))
    o3d_mesh = o3d.io.read_triangle_mesh(str(output_path))

    print(f"  Open3D verification:")
    print(f"    Points: {len(np.asarray(pcd.points))}")
    print(f"    Has colors: {pcd.has_colors()}")
    if pcd.has_colors():
        colors = np.asarray(pcd.colors)
        print(f"    Color range: [{colors.min():.3f}, {colors.max():.3f}]")
    print(f"    Mesh vertices: {len(np.asarray(o3d_mesh.vertices))}")
    print(f"    Mesh triangles: {len(np.asarray(o3d_mesh.triangles))}")

    # Compute diameter
    diameter = compute_model_diameter(mesh.vertices)

    # Create models_info.json
    models_info_path = output_dir / "models_info.json"
    models_info = {
        str(obj_id): {
            "diameter": diameter
        }
    }

    print(f"Creating models_info.json at: {models_info_path}")
    with open(models_info_path, 'w') as f:
        json.dump(models_info, f, indent=2)

    print("\n" + "="*60)
    print("Conversion completed successfully!")
    print("="*60)
    print(f"Output PLY file: {output_path}")
    print(f"Models info: {models_info_path}")
    print(f"Object ID: {obj_id}")
    print(f"Diameter: {diameter:.2f} mm")
    print("\nNext steps:")
    print(f"1. Place '{output_filename}' in your dataset's models/ folder")
    print(f"2. Place 'models_info.json' in the same models/ folder")
    print(f"3. Prepare training images in BOP format")

    return output_path, diameter


def main():
    parser = argparse.ArgumentParser(description='Convert CAD model to PLY format for RAG-6DPose')
    parser.add_argument('input', type=str, help='Input CAD model file (OBJ, STL, etc.)')
    parser.add_argument('--output-dir', type=str, default='models', help='Output directory (default: models/)')
    parser.add_argument('--obj-id', type=int, default=1, help='Object ID number (default: 1)')
    parser.add_argument('--color', type=int, nargs=3, default=[128, 128, 128],
                        help='Default RGB color if mesh has no colors (default: 128 128 128)')

    args = parser.parse_args()

    convert_obj_to_ply(
        input_path=args.input,
        output_dir=args.output_dir,
        obj_id=args.obj_id,
        color=args.color
    )


if __name__ == '__main__':
    main()
