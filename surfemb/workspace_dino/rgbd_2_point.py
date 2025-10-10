
import open3d as o3d
import numpy as np
from PIL import Image

def create_point_cloud(rgb_image_path, depth_image_path, fx,fy,cx,cy):
    
    depth_image = Image.open(depth_image_path)
    depth_array = np.array(depth_image)

    rows, cols = depth_array.shape
    points = []
    corr_xy = []
    for v in range(rows):
        for u in range(cols):
            depth = depth_array[v, u]

            if depth == 0:
                continue

         
            z = depth  
            x = (u - cx) * z / fx
            y = (v - cy) * z / fy

            points.append([x, y, z])
            corr_xy.append([u,v]) 

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(np.array(points)[:, :3])
    corr_xy = np.array(corr_xy)
    return point_cloud,corr_xy

def save_point_cloud(point_cloud, output_path):
    # 保存点云
    o3d.io.write_point_cloud(output_path, point_cloud)

if __name__ == "__main__":

    rgb_image_path = "extracted_rgb_image.png"
    depth_image_path = "extracted_depth_image.png"
    output_path = "crop_tudl.ply"

    point_cloud = create_point_cloud(rgb_image_path, depth_image_path)
    save_point_cloud(point_cloud, output_path)

