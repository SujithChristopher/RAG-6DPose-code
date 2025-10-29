# Quick Start Guide - Training RAG-6DPose

## Overview
After debugging, two training scripts are now available:
- **simple_train.py** ✅ (Recommended) - Fast, reliable training without PyTorch Lightning
- **quick_train.py** - Original script with fixes, uses PyTorch Lightning

## Running Training

### Option 1: Simple Training (Recommended)
```bash
python simple_train.py
```

**Features:**
- ✅ Stable and predictable
- ✅ Fast initialization (~10 seconds)
- ✅ ~1.14 seconds per batch
- ✅ Easy to understand and modify
- ✅ No external logging framework required

**Output:**
```
============================================================
Simple Training - RAG-6DPose
============================================================
...
Epoch 1/1: 100%|██████████| 25/25 [00:28<00:00,  1.14s/it, loss=1.8e-11]
Epoch 1 - Average Loss: 144.3885
============================================================
Training completed successfully!
```

### Option 2: PyTorch Lightning Training
```bash
python quick_train.py
```

**Features:**
- Supports distributed training (if configured)
- Checkpoint management built-in
- Integration with Weights & Biases

**Note:** Takes longer to start due to Lightning framework overhead.

## Testing

### Test Single Batch
Quick validation that everything is working:
```bash
python test_single_batch.py
```

Output:
```
✓ Batch loaded with rgb_crop shape: torch.Size([2, 224, 224, 3])
✓ Forward pass successful, loss: 8.1084
```

## Configuration

All scripts use the same configuration:

```python
DATASET_ROOT = Path("/media/sujith/Project/NOARK_CV/RAG-6DPose-code/chasis_dataset")
MODELS_PATH = Path("/media/sujith/Project/NOARK_CV/RAG-6DPose-code/models")
RES_CROP = 224           # Crop resolution
BATCH_SIZE = 4           # Batch size
NUM_EPOCHS = 2           # Number of epochs
```

To modify configuration, edit the constants at the top of each script.

## Data Requirements

The training pipeline expects:
```
chasis_dataset/
├── train_real/          # Real training images
│   ├── 000000/          # Scene folder
│   │   ├── scene_gt.json
│   │   ├── scene_gt_info.json
│   │   ├── scene_camera.json
│   │   ├── rgb/         # RGB images
│   │   ├── depth/       # Depth maps
│   │   └── mask_visib/  # Visibility masks
│   └── ...

models/
├── obj_000001.ply       # 3D CAD model
└── models_info.json     # Model metadata

cad_features/
└── obj_000001_dino_feat.pt  # Precomputed DINO features
```

## Troubleshooting

### Issue: DINO model download takes long time
**Solution:** It's normal on first run. The 336MB model is downloaded and cached automatically.

### Issue: Out of memory
**Solution:** Reduce `BATCH_SIZE` in the script (try 2 instead of 4).

### Issue: Slow training
**Solution:** Check that GPU is being used:
```python
import torch
print("CUDA available:", torch.cuda.is_available())
print("CUDA device:", torch.cuda.get_device_name(0))
```

### Issue: Loss values are unstable/diverging
**Solution:** This is expected for random initialization. The loss should stabilize after a few batches.

## Output

Training outputs are saved to:
- **simple_train.py:** `simple_train_output/`
- **quick_train.py:** `quick_train_output/`

Both directories contain:
- Checkpoints (PyTorch Lightning)
- Logs and metrics

## Next Steps

1. Start with `python simple_train.py` for quick validation
2. Monitor loss values - should decrease initially
3. Adjust hyperparameters as needed:
   - Learning rate: `LEARNING_RATE = 1e-4` in simple_train.py
   - Batch size: `BATCH_SIZE = 4`
   - Number of epochs: `NUM_EPOCHS = 1`

## References

- **Model:** [model_forward_c2f_dino_unet.py](surfemb/workspace_dino/model_forward_c2f_dino_unet.py)
- **Data Pipeline:** [instance_tudl.py](surfemb/data/instance_tudl.py)
- **Auxiliaries:** [std_auxs.py](surfemb/data/std_auxs.py), [pose_auxs.py](surfemb/data/pose_auxs.py)
- **Debug Summary:** [TRAINING_DEBUG_SUMMARY.md](TRAINING_DEBUG_SUMMARY.md)
