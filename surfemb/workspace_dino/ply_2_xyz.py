import open3d as o3d
import numpy as np
import torch    

def info_cld(pcd=None):
    point_cloud = pcd

    min_xyz = np.min(point_cloud, axis=0)
    max_xyz = np.max(point_cloud, axis=0)
    
    size = max_xyz - min_xyz

    origin = min_xyz

    print("Size:", size)
    # print("Spacing:", spacing)
    print("Origin:", origin)

    return size, origin

