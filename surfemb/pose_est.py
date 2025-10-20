import numpy as np
import cv2
import torch
import torch_scatter
import torch.nn.functional as F
from scipy.spatial.transform import Rotation

from .utils import timer
# import pyprogressivex

from .data  import obj


def build_non_unique_2D_3D_correspondence(Pixel_position, coord_img, scale, offset):
    Point_2D = np.column_stack((Pixel_position[1], Pixel_position[0]))  # Simplified concatenation
    
    # Extract rows and columns directly
    rows = Pixel_position[0]
    cols = Pixel_position[1]

    # Extract all required values from coord_img at once
    Points_3D = coord_img[rows, cols, :3].cpu() * scale + offset
    
    return Point_2D, np.array(Points_3D)


def estimate_pose(mask_lgts: torch.tensor, 
                  K: np.ndarray,   coord_img: torch.tensor, objs: list, obj_idx: int):
    """
    Builds correspondence distribution from queries and keys,
    samples correspondences with inversion sampling,
    samples poses from correspondences with P3P,
    prunes pose hypothesis,
    and scores pose hypotheses based on estimated mask and correspondence distribution.

    :param mask_lgts: (r, r)
    :param query_img: (r, r, e)
    :param obj_pts: (m, 3)
    :param obj_normals: (m, 3)
    :param obj_keys: (m, e)
    :param alpha: exponent factor for correspondence weighing
    :param K: (3, 3) camera intrinsics
    """
    device = mask_lgts.device
    r = mask_lgts.shape[0]

    # down sample
    K = K.copy()

    mask_image = np.array(mask_lgts.cpu())
    # print(mask_image.shape)
    Points_2D = mask_image.nonzero()
    
    scale = objs[obj_idx].scale
    # offset = np.expand_dims(objs[obj_idx].offset, axis=1)
    offset = objs[obj_idx].offset
    
    # find the 2D-3D correspondences and Ransac + PnP
    # build_2D_3D_correspondence = build_non_unique_2D_3D_correspondence
    success = False
    rot = []
    tvecs = []
    # print(Points_2D)
    if Points_2D[0].size != 0:
        Points_2D,  Points_3D = build_non_unique_2D_3D_correspondence(Points_2D, coord_img, scale, offset)
        # print(Points_2D)
        
        
        if len(Points_2D) >= 6:
            success = True

            intrinsic_matrix = np.ascontiguousarray(K)

            if False:
                pose_ests, label = pyprogressivex.find6DPoses(
                                                            x1y1 = Points_2D.astype(np.float64),
                                                            x2y2z2 = Points_3D.astype(np.float64),
                                                            K = intrinsic_matrix.astype(np.float64),
                                                            threshold = 3,  
                                                            neighborhood_ball_radius=30,
                                                            spatial_coherence_weight=0.2,
                                                            maximum_tanimoto_similarity=0.9,
                                                            max_iters=1000,
                                                            minimum_point_number=4,
                                                            maximum_model_number=1
                                                        )
                if pose_ests.shape[0] != 0:
                    rot = pose_ests[0:3, :3]
                    tvecs = pose_ests[0:3, 3]
                    tvecs = tvecs.reshape((3,1))
                else:
                    rot = np.zeros((3,3))
                    tvecs = np.zeros((3,1))
                    success = False
                # 1
            
            else:
                
                _, rvecs, tvecs, inliers = cv2.solvePnPRansac(Points_3D.astype(np.float64),
                                                            Points_2D.astype(np.float64), intrinsic_matrix, distCoeffs=None,
                                                            reprojectionError=3, iterationsCount=1000, flags=cv2.SOLVEPNP_EPNP)
                rot, _ = cv2.Rodrigues(rvecs, jacobian=None)
                pred_pose = np.append(rot, tvecs, axis=1)   

    return rot, tvecs, success
            
    