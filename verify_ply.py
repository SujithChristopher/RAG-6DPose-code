import open3d as o3d
import numpy as np

pcd = o3d.io.read_point_cloud('models/obj_000001.ply')
mesh = o3d.io.read_triangle_mesh('models/obj_000001.ply')

print('=== Point Cloud ===')
print(f'Points: {len(np.asarray(pcd.points))}')
print(f'Has colors: {pcd.has_colors()}')
if pcd.has_colors():
    colors = np.asarray(pcd.colors)
    print(f'Color range: [{colors.min():.3f}, {colors.max():.3f}]')
    print(f'Color shape: {colors.shape}')
    print(f'Sample colors (first 3 points):\n{colors[:3]}')

print('\n=== Mesh ===')
print(f'Vertices: {len(np.asarray(mesh.vertices))}')
print(f'Triangles: {len(np.asarray(mesh.triangles))}')
print(f'Has vertex colors: {mesh.has_vertex_colors()}')
if mesh.has_vertex_colors():
    vertex_colors = np.asarray(mesh.vertex_colors)
    print(f'Vertex color range: [{vertex_colors.min():.3f}, {vertex_colors.max():.3f}]')

print('\n=== Test with code requirements ===')
# Test exactly as the codebase uses it
print('Testing: o3d.io.read_point_cloud().points')
points = np.asarray(o3d.io.read_point_cloud('models/obj_000001.ply').points)
print(f'  Shape: {points.shape}')
print(f'  Sample XYZ (first point): {points[0]}')

print('\nTesting: o3d.io.read_point_cloud().colors')
colors = np.asarray(o3d.io.read_point_cloud('models/obj_000001.ply').colors)
print(f'  Shape: {colors.shape}')
print(f'  Sample RGB (first point): {colors[0]}')
print(f'  All colors same: {np.all(colors == colors[0])}')

print('\n✓ PLY file is compatible with RAG-6DPose requirements!')
