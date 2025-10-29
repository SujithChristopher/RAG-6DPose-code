#!/usr/bin/env python3
"""
Test a single training batch to debug issues
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn as nn
import numpy as np

from surfemb.workspace_dino.model_forward_c2f_dino_unet import Gen_corr
from surfemb.workspace_dino.dino_feat_model import DINO_feat
from surfemb.data.instance_tudl import BopInstanceDataset
from surfemb.data import config as data_config
from surfemb.data.obj import load_objs

# Configuration
DATASET_ROOT = Path("/media/sujith/Project/NOARK_CV/RAG-6DPose-code/chasis_dataset")
MODELS_PATH = Path("/media/sujith/Project/NOARK_CV/RAG-6DPose-code/models")
CAD_FEATURES_PATH = Path("/media/sujith/Project/NOARK_CV/RAG-6DPose-code/cad_features")
N_OBJS = 3
RES_CROP = 224

print("=" * 60)
print("Test Single Batch")
print("=" * 60)

# Load objects
print("\nLoading objects...")
obj_ids = [1]
objs, _ = load_objs(MODELS_PATH, obj_ids=obj_ids, show_progressbar=False)
while len(objs) < 3:
    objs.append(objs[-1])
print(f"✓ Loaded {len(objs)} objects")

# Load DINO model
print("\nLoading DINO model...")
dino_model = DINO_feat()
dino_model.load_model()
for param in dino_model.parameters():
    param.requires_grad = False
print("✓ DINO model loaded")

# Initialize Gen_corr model
print("\nInitializing Gen_corr model...")
model = Gen_corr(objs=objs, n_objs=N_OBJS, dino_model=dino_model, one_obj_idx=0)
model.to('cuda')
model.eval()  # Set to eval mode for testing
print("✓ Model initialized")

# Setup dataset
print("\nSetting up dataset...")
ds_config = data_config.config['chasis']
auxs = model.get_auxs(objs, RES_CROP)

train_ds = BopInstanceDataset(
    dataset_root=DATASET_ROOT,
    pbr=False,
    synt=False,
    test=False,
    cfg=ds_config,
    obj_ids=obj_ids,
    min_visib_fract=0.1,
    min_px_count_visib=1024,
    auxs=auxs,
)
print(f"✓ Dataset loaded: {len(train_ds)} samples")

# Create data loader
print("\nCreating data loader...")
def custom_collate(batch):
    def make_contiguous(obj):
        if isinstance(obj, np.ndarray) and not obj.flags['C_CONTIGUOUS']:
            return obj.copy()
        return obj
    batch = [{k: make_contiguous(v) for k, v in item.items()} for item in batch]
    from torch.utils.data.dataloader import default_collate
    return default_collate(batch)

train_loader = torch.utils.data.DataLoader(
    train_ds,
    batch_size=2,
    shuffle=False,
    num_workers=0,
    collate_fn=custom_collate,
)
print(f"✓ DataLoader created: {len(train_loader)} batches")

# Test loading one batch
print("\nLoading first batch...")
for batch_idx, batch in enumerate(train_loader):
    print(f"✓ Batch {batch_idx} loaded")
    print(f"  - Keys: {batch.keys()}")
    print(f"  - rgb_crop shape: {batch['rgb_crop'].shape}")
    print(f"  - obj_coord shape: {batch['obj_coord'].shape}")
    print(f"  - obj_idx: {batch['obj_idx']}")

    # Move batch to GPU
    print("\n  Moving batch to GPU...")
    for key in batch:
        if isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key].to('cuda')
    print("  ✓ Batch on GPU")

    # Test forward pass
    print("\n  Testing forward pass...")
    try:
        with torch.no_grad():
            loss = model.step(batch, 'test')
        print(f"  ✓ Forward pass successful, loss: {loss:.4f}")
    except Exception as e:
        print(f"  ✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()

    break  # Only test first batch

print("\n" + "=" * 60)
print("Test completed!")
print("=" * 60)
