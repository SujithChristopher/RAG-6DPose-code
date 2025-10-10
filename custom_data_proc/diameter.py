import open3d as o3d
import numpy as np
from scipy.spatial import distance

def compute_model_diameter(ply_file_path):
    mesh = o3d.io.read_triangle_mesh(ply_file_path)
    
    if not mesh.has_vertices():
        raise ValueError("Error")
    
    vertices = np.asarray(mesh.vertices)
    
    max_distance = 0
    for i in range(len(vertices)):
        for j in range(i + 1, len(vertices)):
            dist = np.linalg.norm(vertices[i] - vertices[j])
            max_distance = max(max_distance, dist)


    return max_distance

ply_file = "models/obj_000001.ply" 
try:
    diameter = compute_model_diameter(ply_file)
except ValueError as e:
    print(e)
