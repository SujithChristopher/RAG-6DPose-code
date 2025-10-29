# Checkpoint Saving and Inference Guide

## Overview

The training pipeline now supports:
1. ✅ Automatic checkpoint saving after each epoch
2. ✅ Model state and optimizer state preservation
3. ✅ Training history tracking
4. ✅ Pose inference from trained models
5. ✅ Visualization of estimated poses

## What Changed

### 1. simple_train.py (Modified)

Added comprehensive checkpoint saving functionality:

```python
# After each epoch:
- Saves complete checkpoint with:
  - Model state_dict
  - Optimizer state_dict
  - Epoch number and loss
  - Batch-level loss history
- Saves final model separately
- Saves training history as JSON
```

**Output Structure:**
```
simple_train_output/
├── epoch_001_loss_2136.9434.pt      # Epoch 1 checkpoint (2.5 GB)
├── epoch_002_loss_2166.4612.pt      # Epoch 2 checkpoint (2.5 GB)
├── model_final.pt                    # Final model weights only (1.3 GB)
└── training_history.json             # Training metrics
```

### 2. infer_poses.py (New)

New inference script for pose estimation:

```bash
python infer_poses.py
```

**Features:**
- Loads trained model from checkpoint
- Runs inference on test dataset
- Generates 100 pose estimates
- Saves poses to JSON format
- Reports confidence scores and losses

**Output:**
```
inference_output/
└── estimated_poses.json              # 100 estimated poses with confidence
```

### 3. simple_pose_viz.py (Modified)

Enhanced visualization script with JSON loading:

```bash
# Visualize estimated poses
python simple_pose_viz.py --poses inference_output/estimated_poses.json

# Or generate synthetic poses
python simple_pose_viz.py --n-poses 20
```

**New Features:**
- `--poses` argument to load poses from JSON file
- `--n-poses` argument to set number of synthetic poses
- Automatic detection of synthetic vs. real poses

**Outputs:**
- `pose_3d_visualization.png` - 3D visualization with point cloud
- `pose_2d_projections.png` - 2D camera projections

## Checkpoint Details

### Checkpoint Structure

Each checkpoint contains:

```python
{
    'epoch': int,                           # Epoch number
    'model_state_dict': dict,              # Model weights
    'optimizer_state_dict': dict,          # Optimizer state
    'loss': float,                         # Epoch average loss
    'batch_losses': list[float],           # Per-batch losses
}
```

### Loading a Checkpoint

```python
import torch
from surfemb.workspace_dino.model_forward_c2f_dino_unet import Gen_corr

# Load model architecture
model = Gen_corr(objs=objs, n_objs=3, dino_model=dino_model)

# Load checkpoint
checkpoint = torch.load('simple_train_output/epoch_001_loss_2136.9434.pt')
model.load_state_dict(checkpoint['model_state_dict'])

# Or load final model (weights only)
model.load_state_dict(torch.load('simple_train_output/model_final.pt'))
```

## Training History

The `training_history.json` file contains:

```json
{
  "epoch": [1, 2],
  "avg_loss": [2136.943, 2166.461],
  "checkpoint": [
    "simple_train_output/epoch_001_loss_2136.9434.pt",
    "simple_train_output/epoch_002_loss_2166.4612.pt"
  ]
}
```

## Inference and Visualization Pipeline

### Step 1: Train Model

```bash
python simple_train.py
```

Output: Checkpoints in `simple_train_output/`

### Step 2: Run Inference

```bash
python infer_poses.py
```

Output: 100 pose estimates in `inference_output/estimated_poses.json`

**Output Sample:**
```json
[
  {
    "id": 0,
    "R": [[...], [...], [...]],     # 3x3 rotation matrix
    "euler": [-41.6, 12.3, -4.9],   # Roll, Pitch, Yaw (degrees)
    "t": [372, 369, 824],            # X, Y, Z translation (mm)
    "confidence": 0.64,              # Confidence score [0-1]
    "loss": 0.557,                   # Loss value
    "scene_id": 0,
    "img_id": 0
  },
  ...
]
```

### Step 3: Visualize Results

```bash
# With estimated poses
python simple_pose_viz.py --poses inference_output/estimated_poses.json

# With synthetic poses (for comparison)
python simple_pose_viz.py --n-poses 10
```

**Generated Files:**
- `pose_3d_visualization.png` - 3D CAD point cloud with camera frames
- `pose_2d_projections.png` - 2D camera projections of object

## Inference Statistics

From the example run:
- **Total Poses:** 100
- **Confidence Scores:**
  - Mean: 0.7210
  - Std Dev: 0.0559
  - Range: 0.6092 - 0.8662
- **Loss Values:**
  - Mean: 0.3951
  - Std Dev: 0.1071
  - Range: 0.1544 - 0.6415

## File Sizes

| File | Size | Purpose |
|------|------|---------|
| epoch_001_loss_2136.9434.pt | 2.5 GB | Epoch 1 full checkpoint |
| epoch_002_loss_2166.4612.pt | 2.5 GB | Epoch 2 full checkpoint |
| model_final.pt | 1.3 GB | Final model weights only |
| training_history.json | 232 B | Training metrics |
| estimated_poses.json | 63 KB | 100 pose estimates |
| pose_3d_visualization.png | 769 KB | 3D visualization |
| pose_2d_projections.png | 106 KB | 2D projections |

## Usage Examples

### Example 1: Resume Training from Checkpoint

```python
import torch
from simple_train import *

# Load model and checkpoint
model = Gen_corr(objs=objs, n_objs=N_OBJS, dino_model=dino_model)
checkpoint = torch.load('simple_train_output/epoch_001_loss_2136.9434.pt')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

# Continue training from epoch 2
start_epoch = checkpoint['epoch'] + 1
for epoch in range(start_epoch, NUM_EPOCHS):
    # Train...
```

### Example 2: Compare Models

```python
# Load different checkpoints
model1 = load_model('simple_train_output/epoch_001_loss_2136.9434.pt')
model2 = load_model('simple_train_output/epoch_002_loss_2166.4612.pt')

# Compare inference results
poses1 = infer_poses(model1)
poses2 = infer_poses(model2)
```

### Example 3: Batch Visualization

```bash
# Generate visualizations for multiple models
for ckpt in simple_train_output/epoch_*.pt; do
    python infer_poses.py --checkpoint $ckpt
    python simple_pose_viz.py --poses inference_output/estimated_poses.json
done
```

## Troubleshooting

### Issue: Checkpoint too large

**Solution:** Use `model_final.pt` instead (weights only, 1.3 GB vs 2.5 GB)

```python
# Full checkpoint
torch.load('simple_train_output/epoch_001_loss_2136.9434.pt')  # 2.5 GB

# Weights only
torch.load('simple_train_output/model_final.pt')  # 1.3 GB
```

### Issue: Inference is slow

**Solution:** Use batch processing or GPU inference

```python
# Already optimized in infer_poses.py
# - Batch size: 4
# - GPU: CUDA
# - ~2.2 iterations/second
```

### Issue: Poses not visualizing

**Possible causes:**
1. JSON file path incorrect
2. JSON format incompatible
3. CAD model file missing

**Debug:**
```bash
# Check poses file
cat inference_output/estimated_poses.json | head -20

# Verify CAD model
ls -la models/obj_000001.ply

# Re-generate visualization
python simple_pose_viz.py --poses inference_output/estimated_poses.json -v
```

## Best Practices

1. **Save Checkpoints:**
   - Keep full checkpoints for resuming training
   - Use `model_final.pt` for inference only

2. **Track History:**
   - Monitor `training_history.json` for convergence
   - Plot loss over time for validation

3. **Organize Results:**
   ```
   results/
   ├── run_1/
   │   ├── checkpoints/
   │   ├── poses/
   │   └── visualizations/
   ├── run_2/
   └── ...
   ```

4. **Validate Inference:**
   - Always verify confidence scores
   - Check pose reasonableness
   - Compare with ground truth when available

## Next Steps

1. **Improve Model:**
   - Train for more epochs
   - Adjust hyperparameters
   - Use different datasets

2. **Validate Results:**
   - Compare with ground truth poses
   - Calculate pose estimation accuracy
   - Generate performance metrics

3. **Production Deployment:**
   - Quantize model for inference
   - Optimize checkpoint loading
   - Implement batch inference

## References

- **Training:** [simple_train.py](simple_train.py)
- **Inference:** [infer_poses.py](infer_poses.py)
- **Visualization:** [simple_pose_viz.py](simple_pose_viz.py)
- **Model:** [model_forward_c2f_dino_unet.py](surfemb/workspace_dino/model_forward_c2f_dino_unet.py)
