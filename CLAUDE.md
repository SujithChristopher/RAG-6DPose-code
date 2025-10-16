# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

RAG-6DPose is a 6D object pose estimation method that uses RGB images as input. The codebase is based on [Surfemb](https://github.com/rasmushaugaard/surfemb) and integrates DINO vision features with surface embedding matching for robust pose estimation.

**Key Innovation**: The method combines DINO vision transformer features with point cloud representations and cross-attention mechanisms to match query embeddings from RGB images against 3D surface embeddings of CAD models.

## Training

Train the model on BOP datasets (TUDL dataset example):

```bash
python -m surfemb.scripts.train_matching_dino_unet_tudl tudl --real
```

**Important Configuration**:
- **MUST** set your WandB API key in [surfemb/scripts/train_matching_dino_unet_tudl.py:115](surfemb/scripts/train_matching_dino_unet_tudl.py#L115): `os.environ["WANDB_API_KEY"] = "your key"`
- Default dataset root: `/home/bop/datasets` (line 48)
- Default TUDL CAD models path: `/home/workspace_dino/tudl_cad/` (lines 59-71 in [surfemb/workspace_dino/model_forward_c2f_dino_unet.py](surfemb/workspace_dino/model_forward_c2f_dino_unet.py))
- DINO model cache: `/home/.cache/torch/hub/facebookresearch_dinov2_main` (line 139 in [surfemb/workspace_dino/dino_feat_model.py](surfemb/workspace_dino/dino_feat_model.py))

**Training Flags**:
- `--real`: Use real training images
- `--synth`: Use synthetic (PBR + rendered) images (default enabled, disable with `--no-synth`)
- `--ckpt`: Resume from checkpoint
- `--gpus`: GPU device IDs (default: `[0,1]`)
- `--batch-size`: Default 32
- `--max-steps`: Default 500,000
- `--res-crop`: Crop resolution (default 224)

**Training Strategy**: DDP (Distributed Data Parallel) with unused parameter detection enabled.

## Evaluation

Use the BOP toolkit to evaluate pose estimation results:

```bash
python scripts/eval_bop19_pose.py --renderer_type=vispy --result_filenames=NAME_OF_CSV_WITH_RESULTS
```

Datasets are available at: https://bop.felk.cvut.cz/datasets/

## Custom Dataset Processing

To convert custom datasets (images + 3D models) to BOP format, use preprocessing scripts in [custom_data_proc/](custom_data_proc/):

- [mask_converter.py](custom_data_proc/mask_converter.py): Convert mask formats
- [m2mm_converter.py](custom_data_proc/m2mm_converter.py): Unit conversion (meters to millimeters)
- [diameter.py](custom_data_proc/diameter.py): Calculate object diameters
- [json_key_counter.py](custom_data_proc/json_key_counter.py): JSON utility

## Architecture

### Core Components

**1. Model Architecture** ([surfemb/workspace_dino/model_forward_c2f_dino_unet.py](surfemb/workspace_dino/model_forward_c2f_dino_unet.py))

`Gen_corr` is the main training model that extends `SurfaceEmbeddingModel`:

- **DINO Feature Extractor** (`DINO_feat`): Extracts 768-dim features from DINOv2 ViT-B/14
- **Image Encoder**: ResNet-based encoder + UNet decoder for dense feature extraction
- **Point Cloud Processing**:
  - CAD model point clouds with XYZ + RGB + DINO features (3 + 3 + 768 = 774 dims)
  - Self-attention on point cloud features
  - Cross-attention between image features (28x28) and point cloud tokens
- **Multi-head Decoder**:
  - Embedding decoder: Predicts dense query embeddings (256-dim)
  - Mask decoder: Predicts object segmentation mask
- **MLP Networks**:
  - `mlps`: 3D coordinate → 128-dim embedding (SIREN)
  - `mlps_gt_corr`: DINO features (768) → 256-dim
  - `mlps_fusion`: Fuses coordinate + DINO embeddings (384) → 256-dim final keys

**Training Loss**: Mask loss (L1) + NCE contrastive loss

**2. Base Surface Embedding** ([surfemb/surface_embedding.py](surfemb/surface_embedding.py))

`SurfaceEmbeddingModel`: Original simpler model with ResNet-UNet architecture without DINO features.

**3. Pose Estimation** ([surfemb/pose_est_matching.py](surfemb/pose_est_matching.py))

`estimate_pose()`: Samples pose hypotheses via:
1. Build correspondence distribution from query/key similarities
2. Sample correspondences with inverse sampling
3. Generate pose hypotheses with P3P (solveP3P with AP3P)
4. Prune invalid hypotheses (depth bounds, 2D distance, surface normal visibility)
5. Score poses using mask + correspondence likelihoods

**4. Data Pipeline** ([surfemb/data/](surfemb/data/))

- `instance_tudl.py`: BOP dataset loader with scene/object/pose annotations
- `config.py`: Dataset-specific configs (TUDL, TLESS, HB, ITODD)
- `obj.py`: 3D object model loading
- `renderer.py`: Synthetic view rendering
- Data augmentation pipeline with Albumentations

### Key Design Patterns

**Inference Two-Stage Process**:
1. `infer_cnn()`: Query embeddings from RGB crop (rotation ensemble: 4 rotations averaged)
2. `infer_mlp()`: Key embeddings from 3D surface points + nearest DINO features

**Coordinate Normalization**: Objects are normalized via `scale` and `offset` to [-1, 1] range.

**Multi-object Support**: Model handles multiple objects with separate MLP heads per object (indexed by `obj_idx`).

### Directory Structure

```
surfemb/
├── scripts/
│   └── train_matching_dino_unet_tudl.py  # Main training script
├── workspace_dino/                        # DINO-enhanced components
│   ├── model_forward_c2f_dino_unet.py    # Core Gen_corr model
│   ├── dino_feat_model.py                # DINO feature extractor
│   ├── endecoder.py                      # UNet encoder/decoder variants
│   ├── cross_attention.py                # Attention modules
│   ├── modules/                          # ResNet, PointNet2 layers
│   └── panda3d_renderer/                 # 3D rendering utilities
├── data/                                  # Data loading and augmentation
│   ├── instance_tudl.py                  # BOP dataset loader
│   ├── config.py                         # Dataset configurations
│   ├── obj.py                            # 3D object model handling
│   └── pose_auxs.py                      # Pose-related data transforms
├── surface_embedding.py                   # Base embedding model
├── pose_est_matching.py                   # Pose estimation pipeline
├── pose_est.py                           # Pose utilities
└── dep/                                   # Dependencies
    ├── unet.py                           # ResNet-UNet architecture
    └── siren.py                          # SIREN MLP

custom_data_proc/                          # Dataset preprocessing utilities
```

## Important Implementation Details

**PyTorch Lightning**: Training uses PyTorch Lightning with custom callbacks for model checkpointing. Checkpoints saved to `data/models/` with format `{dataset}-{wandb_run_id}`.

**Hard-coded Paths**: Several absolute paths are hard-coded and must be updated for your environment:
- TUDL CAD point clouds: `/home/workspace_dino/tudl_cad/cad_points_xyz_color/obj_*.ply`
- TUDL CAD DINO features: `/home/workspace_dino/tudl_cad/cad_points_feat_768/cad_points_feat_*.pt`
- TUDL models info: `/home/bop/datasets/tudl/models/models_info.json`
- Test targets: `/home/bop/datasets/tudl/test_targets_bop19.json`

**Multi-GPU Training**: Uses DDP strategy with `ddp_find_unused_parameters_true` to handle conditional graph execution.

**Point Cloud Sampling**: 3000 points sampled from each CAD model (fixed seed 42) for training/inference.

**Rotation Ensemble**: At inference, queries 4x rotations (0°, 90°, 180°, 270°) and averages for robustness.

## Dependencies

Key dependencies (inferred from imports):
- PyTorch + PyTorch Lightning
- NumPy, OpenCV
- Albumentations (image augmentation)
- WandB (logging)
- torch_scatter (pose scoring)
- Open3D (point cloud I/O)
- scipy (transformations)
- DINOv2 (via torch.hub)

## Notes

- This codebase is research code with dataset-specific hard-coded paths
- Currently configured for TUDL dataset with 3 objects (obj_000001, obj_000002, obj_000003)
- To adapt for other datasets/objects, modify point cloud loading in `Gen_corr.__init__()` and update `n_objs` parameter
