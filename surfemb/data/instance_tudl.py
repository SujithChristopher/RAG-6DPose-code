import json
from pathlib import Path
from typing import Sequence
import warnings

import numpy as np
from tqdm import tqdm
import torch.utils.data

from .config import DatasetConfig

import json
from .sym_aware import get_symmetry_transformations

def load_json(path, keys_to_int=False):
  """Loads content of a JSON file.

  :param path: Path to the JSON file.
  :return: Content of the loaded JSON file.
  """
  # Keys to integers.
  def convert_keys_to_int(x):
    return {int(k) if k.lstrip('-').isdigit() else k: v for k, v in x.items()}

  with open(path, 'r') as f:
    if keys_to_int:
      content = json.load(f, object_hook=lambda x: convert_keys_to_int(x))
    else:
      content = json.load(f)

  return content

# Load models_info dynamically based on dataset
# Default to TUDL for backward compatibility
try:
    models_info = load_json(
            '/home/bop/datasets/tudl/models/models_info.json', keys_to_int=True)
except:
    # Will be loaded per dataset in __init__
    models_info = None
# BopInstanceDataset should only be used with test=True for debugging reasons
# use detector_crops.DetectorCropDataset for actual test inference


class BopInstanceDataset(torch.utils.data.Dataset):
    def __init__(
            self, dataset_root: Path, pbr: bool,synt: bool, test: bool, cfg: DatasetConfig,
            obj_ids: Sequence[int],
            scene_ids=None, min_visib_fract=0.1, min_px_count_visib=1024,
            auxs: Sequence['BopInstanceAux'] = tuple(), show_progressbar=True,
            infer_idx = None
    ):
        self.pbr, self.test, self.cfg = pbr, test, cfg
        if pbr:
            assert not test
            self.data_folder = dataset_root / 'train_pbr'
            self.img_folder = 'rgb'
            self.depth_folder = 'depth'
            self.img_ext = 'jpg'
            self.depth_ext = 'png'
            print("train_pbr")
        elif synt:
            assert not test
            self.data_folder = dataset_root / 'train_render'
            self.img_folder = 'rgb'
            self.depth_folder = 'depth'
            self.img_ext = 'png'
            self.depth_ext = 'png'
            print("train_render")
        else:

            self.data_folder = dataset_root / (cfg.test_folder if test else "train_real")
            self.img_folder = cfg.img_folder
            self.depth_folder = cfg.depth_folder
            self.img_ext = cfg.img_ext
            self.depth_ext = cfg.depth_ext
            print(self.data_folder)
            

        self.auxs = auxs
        obj_idxs = {obj_id: idx for idx, obj_id in enumerate(obj_ids)}
        self.instances = []
        

        if test:
            with open('/home/bop/datasets/tudl/test_targets_bop19.json', 'r') as f:
                test_targets = json.load(f)
            indexed_data = {}
            for entry in test_targets:
                key = (entry['im_id'], entry['obj_id'], entry['scene_id'])
                indexed_data[key] = entry


        idx = 0
        if scene_ids is None:
            scene_ids = sorted([int(p.name) for p in self.data_folder.glob('*')])
        for scene_id in tqdm(scene_ids, 'loading crop info') if show_progressbar else scene_ids:
            scene_folder = self.data_folder / f'{scene_id:06d}'
            scene_gt = json.load((scene_folder / 'scene_gt.json').open())
            scene_gt_info = json.load((scene_folder / 'scene_gt_info.json').open())
            scene_camera = json.load((scene_folder / 'scene_camera.json').open())

            for img_id, poses in scene_gt.items():
                img_info = scene_gt_info[img_id]
                K = np.array(scene_camera[img_id]['cam_K']).reshape((3, 3)).copy()
                if pbr:
                    warnings.warn('Altering camera matrix, since PBR camera matrix doesnt seem to be correct')
                    K[:2, 2] -= 0.5

                for pose_idx, pose in enumerate(poses):
                    obj_id = pose['obj_id']
                    
                    if not test:
                        1
                    else:
                        if (int(img_id), obj_id, scene_id) in indexed_data:
                            1
                        else:
                            continue
                    
                    if obj_ids is not None and obj_id not in obj_ids:
                        continue
                    pose_info = img_info[pose_idx]
                    if pose_info['visib_fract'] < min_visib_fract:
                        continue
                    if pose_info['px_count_visib'] < min_px_count_visib:
                        continue

                    bbox_visib = pose_info['bbox_visib']
                    bbox_obj = pose_info['bbox_obj']

                    cam_R_obj = np.array(pose['cam_R_m2c']).reshape(3, 3)
                    cam_t_obj = np.array(pose['cam_t_m2c']).reshape(3, 1)
                    
                   
                    self.instances.append(dict(
                        scene_id=scene_id, img_id=int(img_id), K=K, obj_id=obj_id, pose_idx=pose_idx,
                        bbox_visib=bbox_visib, bbox_obj=bbox_obj, cam_R_obj=cam_R_obj, cam_t_obj=cam_t_obj,
                        obj_idx=obj_idxs[obj_id], bbox = bbox_obj ########
                        # , dino_feat28=self.dino_feat[idx]
                    ))
                    
                    idx+=1

        for aux in self.auxs:
            aux.init(self)

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, i):
        instance = self.instances[i].copy()
        for aux in self.auxs:
            instance = aux(instance, self)
        return instance


class BopInstanceAux:
    def init(self, dataset: BopInstanceDataset):
        pass

    def __call__(self, data: dict, dataset: BopInstanceDataset) -> dict:
        pass


def _main():
    from .config import tless
    for pbr, test in (True, False), (False, False), (False, True):
        print(f'pbr: {pbr}, test: {test}')
        data = BopInstanceDataset(dataset_root=Path('bop/tless'), pbr=pbr, test=test, cfg=tless, obj_ids=range(1, 31))
        print(len(data))


if __name__ == '__main__':
    _main()
