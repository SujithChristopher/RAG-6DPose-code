import open3d as o3d
import numpy as np

input_ply_path = "models_m/obj_000001.ply"  
output_ply_path = "models/obj_000001.ply"

mesh = o3d.io.read_triangle_mesh(input_ply_path)

if not mesh.has_vertices():
    print("ERROR")
else:
    vertices_in_mm = np.asarray(mesh.vertices) * 1000
    mesh.vertices = o3d.utility.Vector3dVector(vertices_in_mm)

    o3d.io.write_triangle_mesh(output_ply_path, mesh, write_ascii=True)
