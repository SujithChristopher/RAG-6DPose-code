import torch
import os
import sys

import urllib
import io
import numpy as np
from PIL import Image

from typing import Tuple

import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
from scipy.ndimage import binary_closing, binary_opening
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F

# os.environ["XFORMERS_DISABLED"] = "0" # Switch to enable xFormers

REPO_NAME = "facebookresearch/dinov2"
MODEL_NAME = "dinov2_vitb14"


DEFAULT_SMALLER_EDGE_SIZE = 448
DEFAULT_BACKGROUND_THRESHOLD = 0.05
DEFAULT_APPLY_OPENING = False
DEFAULT_APPLY_CLOSING = False


def load_array_from_url(url: str) -> np.ndarray:
    with urllib.request.urlopen(url) as f:
        array_data = f.read()
        g = io.BytesIO(array_data)
        return np.load(g)


def load_image_from_url(url: str) -> Image:
    with urllib.request.urlopen(url) as f:
        return Image.open(f).convert("RGB")


# Precomputed foreground / background projection
# STANDARD_ARRAY_URL = "https://dl.fbaipublicfiles.com/dinov2/arrays/standard.npy"
# standard_array = load_array_from_url(STANDARD_ARRAY_URL)
# print("load standard_array")
# standard_array = load_array_from_file()
# EXAMPLE_IMAGE_URL = "https://dl.fbaipublicfiles.com/dinov2/images/example.jpg"
# example_image = load_image_from_url(EXAMPLE_IMAGE_URL)


def make_transform(smaller_edge_size: int) -> transforms.Compose:
    IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
    interpolation_mode = transforms.InterpolationMode.BICUBIC

    return transforms.Compose([
        transforms.Resize(size=smaller_edge_size, interpolation=interpolation_mode, antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
    ])


def prepare_image(image,
                  smaller_edge_size: float,
                  patch_size: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    # transform = make_transform(int(smaller_edge_size))
    image_tensor = image

    # Crop image to dimensions that are a multiple of the patch size
    height, width = image_tensor.shape[1:] # C x H x W
    cropped_width, cropped_height = width - width % patch_size, height - height % patch_size
    image_tensor = image_tensor[:, :cropped_height, :cropped_width]

    grid_size = (cropped_height // patch_size, cropped_width // patch_size) # h x w (TODO: check)
    return image_tensor, grid_size


def make_foreground_mask(tokens,
                         grid_size: Tuple[int, int],
                         background_threshold: float = 0.0,
                         apply_opening: bool = True,
                         apply_closing: bool = True):

    projection = tokens @ standard_array
    mask = projection > background_threshold
    mask = mask.reshape(*grid_size)
    if apply_opening:
        mask = binary_opening(mask)
    if apply_closing:
        mask = binary_closing(mask)
    return mask.flatten()


def render_patch_pca(model, 
                     image_batch,
                     smaller_edge_size: float = 448,
                     patch_size: int = 14,
                     background_threshold: float = 0.05,
                     apply_opening: bool = False,
                     apply_closing: bool = False,
                     ):
    origin_img_shape = [image_batch.shape[2], image_batch.shape[3]]
    all_img = []
    all_grid_size = []
    for i in range(image_batch.shape[0]):
        image_tensor, grid_size = prepare_image(image_batch[i,:,:,:], smaller_edge_size, patch_size)
        all_grid_size.append(grid_size)
        all_img.append(image_tensor.unsqueeze(0))
    image_batch = torch.cat(all_img, dim=0)

    model = model
    with torch.inference_mode():
        tokens = model.get_intermediate_layers(image_batch)[0]


    all_arrays = []
    for i in range(image_batch.shape[0]):
       
        array = tokens[i]

        array = array.reshape(1 ,-1, *grid_size)

        all_arrays.append(array)

    all_arrays = torch.cat(all_arrays,dim=0)
    return all_arrays


class DINO_feat(nn.Module):
    def __init__(self):
        super(DINO_feat, self).__init__()
        self.model = None
        
    
    def load_model(self):
        self.model = torch.hub.load('/home/.cache/torch/hub/facebookresearch_dinov2_main', 'dinov2_vitb14', trust_repo=True, source='local')


    def forward(self, x):  # x.shape [32, 3, 256, 256]
        result = render_patch_pca(model=self.model,
                 image_batch=x,
                 smaller_edge_size=DEFAULT_SMALLER_EDGE_SIZE,
                 patch_size=self.model.patch_size,
                 background_threshold=DEFAULT_BACKGROUND_THRESHOLD,
                 apply_opening=DEFAULT_APPLY_OPENING,
                 apply_closing=DEFAULT_APPLY_CLOSING)
    
        return  result
  