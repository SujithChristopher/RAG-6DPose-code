"""
Convert OBJ CAD model to PLY format with proper vertex colors for RAG-6DPose
"""

import trimesh
import open3d as o3d
import numpy as np
import json
from pathlib import Path
import argparse


def compute_model_diameter(vertices):
    """Compute the diameter of the model (max distance between any two vertices)."""
    print("Computing model diameter...")

    if len(vertices) > 10000:
        print(f"  Large mesh detected ({len(vertices)} vertices), sampling 5000 points...")
        indices = np.random.choice(len(vertices), size=min(5000, len(vertices)), replace=False)
        sampled_vertices = vertices[indices]
    else:
        sampled_vertices = vertices

    from scipy.spatial.distance import pdist
    distances = pdist(sampled_vertices)
    max_distance = distances.max()

    print(f"  Model diameter: {max_distance:.2f} mm")
    return float(max_distance)


def convert_obj_to_ply(input_path, output_dir, obj_id=1, color=[128, 128, 128]):
    """
    Convert OBJ to PLY with proper vertex colors using Open3D.
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from: {input_path}")

    # Load with Open3D for better color handling
    mesh_o3d = o3d.io.read_triangle_mesh(str(input_path))

    vertices = np.asarray(mesh_o3d.vertices)
    triangles = np.asarray(mesh_o3d.triangles)

    print(f"  Vertices: {len(vertices)}")
    print(f"  Triangles: {len(triangles)}")

    # Check if mesh has vertex colors
    if not mesh_o3d.has_vertex_colors():
        print(f"  No vertex colors found, adding default color: RGB{color}")
        # Add vertex colors (normalized to [0, 1] range for Open3D)
        vertex_colors = np.ones((len(vertices), 3)) * np.array(color) / 255.0
        mesh_o3d.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
    else:
        print(f"  Vertex colors found!")
        vertex_colors = np.asarray(mesh_o3d.vertex_colors)
        print(f"    Color range: [{vertex_colors.min():.3f}, {vertex_colors.max():.3f}]")

    # Export to PLY with proper naming
    output_filename = f"obj_{obj_id:06d}.ply"
    output_path = output_dir / output_filename

    print(f"Exporting to: {output_path}")
    o3d.io.write_triangle_mesh(str(output_path), mesh_o3d, write_vertex_colors=True)

    # Verify the exported file
    print("Verifying PLY file...")
    pcd = o3d.io.read_point_cloud(str(output_path))
    mesh_verify = o3d.io.read_triangle_mesh(str(output_path))

    print(f"  Point cloud verification:")
    print(f"    Points: {len(np.asarray(pcd.points))}")
    print(f"    Has colors: {pcd.has_colors()}")
    if pcd.has_colors():
        colors = np.asarray(pcd.colors)
        print(f"    Color range: [{colors.min():.3f}, {colors.max():.3f}]")

    print(f"  Mesh verification:")
    print(f"    Vertices: {len(np.asarray(mesh_verify.vertices))}")
    print(f"    Triangles: {len(np.asarray(mesh_verify.triangles))}")
    print(f"    Has vertex colors: {mesh_verify.has_vertex_colors()}")

    # Compute diameter using trimesh for consistency
    mesh_trimesh = trimesh.load_mesh(str(input_path))
    diameter = compute_model_diameter(mesh_trimesh.vertices)

    # Create models_info.json
    models_info_path = output_dir / "models_info.json"
    models_info = {
        str(obj_id): {
            "diameter": diameter
        }
    }

    print(f"\nCreating models_info.json at: {models_info_path}")
    with open(models_info_path, 'w') as f:
        json.dump(models_info, f, indent=2)

    print("\n" + "="*60)
    print("Conversion completed successfully!")
    print("="*60)
    print(f"Output PLY file: {output_path}")
    print(f"Models info: {models_info_path}")
    print(f"Object ID: {obj_id}")
    print(f"Diameter: {diameter:.2f} mm")

    if mesh_verify.has_vertex_colors():
        print("\n✓ Vertex colors: PRESENT")
    else:
        print("\n✗ Warning: Vertex colors not properly saved!")

    print("\nNext steps:")
    print(f"1. PLY file is ready at: {output_path}")
    print(f"2. Models info is ready at: {models_info_path}")
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
