# RAG-6DPose - 6D Object Pose Estimation

6D object pose estimation using RGB images with DINO vision transformer features and surface embedding matching.

## Quick Start

### 1. Training

```bash
python simple_train.py
```

**Output:** Trained model checkpoint in `simple_train_output/`
- `model_final.pt` - Final model weights (1.3 GB)
- `training_history.json` - Training metrics

**Configuration Options** (in `simple_train.py`):
- `BATCH_SIZE = 4` - Adjust for GPU memory
- `NUM_EPOCHS = 2` - Number of training epochs
- `CHECKPOINT_STRATEGY = 'inference_only'` - Checkpoint saving mode:
  - `'inference_only'` - Save only final model (1.3 GB)
  - `'last_checkpoint'` - Save latest checkpoint only (2.5 GB)
  - `'all_epochs'` - Save all epochs (not recommended)

### 2. Inference

```bash
python infer_poses.py
```

**Output:** 100 estimated poses in `inference_output/estimated_poses.json`
- Rotation matrices, Euler angles, translations
- Confidence scores and loss values

### 3. Visualization

```bash
python simple_pose_viz.py --poses inference_output/estimated_poses.json
```

**Output:** PNG visualizations
- `pose_3d_visualization.png` - 3D CAD point cloud with camera frames
- `pose_2d_projections.png` - 2D camera projections

## Project Structure

```
RAG-6DPose-code/
├── simple_train.py              # Main training script
├── infer_poses.py               # Inference script
├── simple_pose_viz.py           # Visualization script
├── test_single_batch.py         # Single batch test
├── quick_train.py               # PyTorch Lightning training (alternative)
│
├── models/                      # CAD models
│   ├── obj_000001.ply          # 3D model (PLY format)
│   └── models_info.json
│
├── cad_features/                # Pre-computed DINO features
│   └── obj_000001_dino_feat.pt
│
├── chasis_dataset/              # Training data (BOP format)
│   └── train_real/000001/
│       ├── rgb/                 # RGB images
│       ├── depth/               # Depth maps
│       ├── mask_visib/          # Segmentation masks
│       ├── scene_gt.json        # Ground truth poses
│       ├── scene_camera.json    # Camera intrinsics
│       └── scene_gt_info.json   # Visibility info
│
└── surfemb/                     # Model and data loading code
    ├── workspace_dino/          # DINO-enhanced model
    ├── data/                    # Dataset loaders
    └── scripts/                 # Training utilities
```

## Training Output

After training, outputs are saved to:
- **Checkpoints:** `simple_train_output/`
- **Inference:** `inference_output/`
- **Visualizations:** `pose_*.png` in root directory

### File Sizes

| File | Size | Purpose |
|------|------|---------|
| model_final.pt | 1.3 GB | Final model weights |
| estimated_poses.json | ~64 KB | 100 pose estimates |
| pose_3d_visualization.png | ~770 KB | 3D visualization |
| pose_2d_projections.png | ~106 KB | 2D projections |

## Model Architecture

**Gen_corr** - Main model combining:
- **DINO Feature Extractor** - 768-dim vision transformer features
- **Image Encoder** - ResNet backbone with UNet decoder
- **Point Cloud Processing** - Self-attention on 3D features
- **Cross-Attention** - Between image and 3D features
- **Decoders** - Embedding prediction and segmentation mask

**Training Loss:** Mask loss (L1) + NCE contrastive loss

## Requirements

- Python 3.12+
- PyTorch 2.6+
- CUDA 12.6+ (for GPU)
- Open3D, NumPy, OpenCV, Albumentations

## Key Scripts

| Script | Purpose | Time |
|--------|---------|------|
| `simple_train.py` | Train model | ~1 min/epoch |
| `infer_poses.py` | Estimate poses | ~11 sec (100 samples) |
| `simple_pose_viz.py` | Generate plots | ~15 sec |
| `test_single_batch.py` | Verify setup | ~5 sec |

## Environment Setup

```bash
# Activate environment
source /home/sujith/miniconda3/bin/activate r12

# Run any script
python simple_train.py
```

## Troubleshooting

**Out of Memory:**
- Reduce `BATCH_SIZE` in training script
- Try `BATCH_SIZE = 2` or `BATCH_SIZE = 1`

**DINO Model Download Slow:**
- First run downloads ~350MB DINOv2 model
- Cached after first run for faster subsequent runs

**Dataset Not Found:**
- Check `DATASET_ROOT` path in scripts (must be absolute path)
- Verify CAD model file: `models/obj_000001.ply`
- Verify DINO features: `cad_features/obj_000001_dino_feat.pt`

## References

- **Model Code:** [surfemb/workspace_dino/](surfemb/workspace_dino/)
- **Data Pipeline:** [surfemb/data/](surfemb/data/)
- **Project Details:** [CLAUDE.md](CLAUDE.md)

## Status

✅ Training pipeline working
✅ Inference implemented
✅ Visualization generated
✅ Checkpoint saving functional

---

**Last Updated:** 2025-10-30
