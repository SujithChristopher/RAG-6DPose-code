
import cv2
import numpy as np


def crop_by_mask(rgb_path, dep_path,mask_path):
    rgb_image = cv2.imread(rgb_path)
    depth_image = cv2.imread(dep_path, cv2.IMREAD_UNCHANGED) 
    mask_image = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    output_rgb = np.zeros_like(rgb_image)
    output_depth = np.zeros_like(depth_image)

    output_rgb[mask_image == 255] = rgb_image[mask_image == 255]
    output_depth[mask_image == 255] = depth_image[mask_image == 255]

    return output_rgb, output_depth
