#!/usr/bin/env python3
"""
Simplified training script without PyTorch Lightning
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm

from surfemb.workspace_dino.model_forward_c2f_dino_unet import Gen_corr
from surfemb.workspace_dino.dino_feat_model import DINO_feat
from surfemb.data.instance_tudl import BopInstanceDataset
from surfemb.data import config as data_config
from surfemb.data.obj import load_objs

# Configuration
DATASET_ROOT = Path("/media/sujith/Project/NOARK_CV/RAG-6DPose-code/chasis_dataset")
MODELS_PATH = Path("/media/sujith/Project/NOARK_CV/RAG-6DPose-code/models")
OUTPUT_DIR = Path("simple_train_output")
N_OBJS = 3
RES_CROP = 224
BATCH_SIZE = 4
NUM_EPOCHS = 2
LEARNING_RATE = 1e-4

# Checkpoint saving strategy:
# 'inference_only': Save only final model (1.3 GB) - recommended for finished training
# 'last_checkpoint': Save only latest epoch (2.5 GB) - can resume from last epoch
# 'all_epochs': Save all epochs (2.5 GB × NUM_EPOCHS) - original behavior, not recommended
CHECKPOINT_STRATEGY = 'inference_only'  # ← Change this to choose strategy

OUTPUT_DIR.mkdir(exist_ok=True)

print("=" * 60)
print("Simple Training - RAG-6DPose")
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
print("✓ DINO model loaded (frozen)")

# Initialize model
print("\nInitializing model...")
model = Gen_corr(objs=objs, n_objs=N_OBJS, dino_model=dino_model, one_obj_idx=0)
model.to('cuda')
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
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    collate_fn=custom_collate,
)
print(f"✓ DataLoader created: {len(train_loader)} batches")

# Setup optimizer
print("\nSetting up optimizer...")
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
print(f"✓ Optimizer configured: Adam (lr={LEARNING_RATE})")

# Training loop
print("\n" + "=" * 60)
print("Starting training...")
print("=" * 60)

# Track training metrics
train_history = {'epoch': [], 'avg_loss': [], 'checkpoint': []}

for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0.0
    batch_losses = []

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
    for batch_idx, batch in enumerate(pbar):
        # Move batch to GPU
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to('cuda')

        # Forward pass
        optimizer.zero_grad()
        loss = model.step(batch, 'train')

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        loss_val = loss.item()
        total_loss += loss_val
        batch_losses.append(loss_val)
        pbar.set_postfix({'loss': loss_val})

    avg_loss = total_loss / len(train_loader)

    # Save checkpoint based on strategy
    if CHECKPOINT_STRATEGY == 'all_epochs':
        # Save all epoch checkpoints (NOT RECOMMENDED - uses lots of disk space)
        checkpoint_path = OUTPUT_DIR / f"epoch_{epoch+1:03d}_loss_{avg_loss:.4f}.pt"
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
            'batch_losses': batch_losses,
        }
        torch.save(checkpoint, checkpoint_path)
        train_history['checkpoint'].append(str(checkpoint_path))
        print(f"  Checkpoint saved: {checkpoint_path.name}")

    elif CHECKPOINT_STRATEGY == 'last_checkpoint':
        # Save only the latest checkpoint (overwrite previous)
        checkpoint_path = OUTPUT_DIR / "latest_checkpoint.pt"
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
            'batch_losses': batch_losses,
        }
        torch.save(checkpoint, checkpoint_path)
        train_history['checkpoint'].append(str(checkpoint_path))
        print(f"  Latest checkpoint saved: {checkpoint_path.name}")

    # Always track history
    train_history['epoch'].append(epoch + 1)
    train_history['avg_loss'].append(avg_loss)

    print(f"\nEpoch {epoch+1} - Average Loss: {avg_loss:.4f}")

# Save training history
import json
history_file = OUTPUT_DIR / "training_history.json"
with open(history_file, 'w') as f:
    json.dump(train_history, f, indent=2)
print(f"\nTraining history saved: {history_file}")

# Save final model
final_model_path = OUTPUT_DIR / "model_final.pt"
torch.save(model.state_dict(), final_model_path)
print(f"Final model saved: {final_model_path}")

print("\n" + "=" * 60)
print("Training completed successfully!")
print("=" * 60)
print(f"\nCheckpoint Strategy: {CHECKPOINT_STRATEGY.upper()}")
print(f"Output directory: {OUTPUT_DIR}")
print(f"\nSaved files:")

# List all files in output directory
output_files = list(OUTPUT_DIR.glob("*"))
total_size = 0
for file in sorted(output_files):
    if file.is_file():
        size_mb = file.stat().st_size / 1024 / 1024
        total_size += file.stat().st_size
        print(f"  - {file.name:<40} ({size_mb:>7.1f} MB)")

print(f"\n  Total storage used: {total_size / (1024**3):.2f} GB")

print(f"\nNextstep: Run inference")
print(f"  python infer_poses.py")
print(f"\nThen visualize:")
print(f"  python simple_pose_viz.py --poses inference_output/estimated_poses.json")
