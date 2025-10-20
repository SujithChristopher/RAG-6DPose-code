# RAG-6DPose CHASIS Training Setup

6D object pose estimation for CHASIS object using RGB images with DINO features.

---

## 🎯 Current Status

**Phase:** Ready for Training (5/8 Complete - 62.5%)

✅ CAD Model Ready
✅ DINO Features Generated
✅ BOP Dataset Created
✅ Code Modified for CHASIS
📋 Training Pipeline Test (Next)

---

## 📁 Project Structure

```
RAG-6DPose-code/
├── models/
│   ├── obj_000001.ply          # CHASIS CAD (66K vertices)
│   └── models_info.json         # Diameter: 260mm
│
├── cad_features/
│   └── obj_000001_dino_feat.pt  # DINO features (195MB, 768-dim)
│
├── chasis_dataset/
│   └── train_real/000001/       # Training data (50 images)
│       ├── rgb/                 # RGB images
│       ├── mask_visib/          # Segmentation masks
│       ├── depth/               # Depth maps
│       ├── scene_camera.json    # Camera intrinsics
│       ├── scene_gt.json        # Ground truth poses
│       └── scene_gt_info.json   # Visibility info
│
├── surfemb/                     # Main codebase (modified)
│   ├── workspace_dino/          # DINO-enhanced model
│   ├── data/                    # Dataset loaders
│   └── scripts/                 # Training scripts
│
└── Tools/
    ├── convert_obj_to_ply_fixed.py              # CAD conversion
    ├── generate_cad_dino_features_simple.py     # DINO generation
    ├── generate_minimal_dataset.py              # Quick dataset
    └── verify_ply.py                            # PLY validation
```

---

## 🚀 Quick Start

### 1. Configure Training Script

Edit [surfemb/scripts/train_matching_dino_unet_tudl.py](surfemb/scripts/train_matching_dino_unet_tudl.py):

```python
# Line ~48: Dataset path
dataset_root = Path('/media/sujith/Project/NOARK_CV/RAG-6DPose-code/chasis_dataset')

# Line ~65: Config
from surfemb.data.config import config
cfg = config['chasis']
n_objs = 1

# Line ~115: WandB key
os.environ["WANDB_API_KEY"] = "your_key"  # From wandb.ai/authorize
```

### 2. Test Configuration

```bash
source /home/sujith/miniconda3/bin/activate r12

# Test config
python -c "from surfemb.data.config import config; print(f'✓ {config[\"chasis\"].train_folder}')"

# Test data loading
python -c "import open3d as o3d, torch; pcd=o3d.io.read_point_cloud('models/obj_000001.ply'); feat=torch.load('cad_features/obj_000001_dino_feat.pt'); print(f'✓ {len(pcd.points)} points')"
```

### 3. Run Training Test

```bash
# Short test (100 steps, ~10 minutes)
python -m surfemb.scripts.train_matching_dino_unet_tudl chasis --real --max-steps 100 --batch-size 4

# Full training (after test succeeds)
python -m surfemb.scripts.train_matching_dino_unet_tudl chasis --real
```

---

## 📖 Documentation

- **[CLEAN_PATH_SUMMARY.md](CLEAN_PATH_SUMMARY.md)** - Overview and next steps ⭐ Start here
- **[CODE_CHANGES_APPLIED.md](CODE_CHANGES_APPLIED.md)** - Code modifications + troubleshooting
- **[CLAUDE.md](CLAUDE.md)** - Codebase architecture reference
- **[archive_docs/](archive_docs/)** - Historical documentation

---

## 🔧 Key Files Modified

| File | Change | Status |
|------|--------|--------|
| `surfemb/workspace_dino/model_forward_c2f_dino_unet.py` | CAD/DINO paths | ✅ |
| `surfemb/data/config.py` | CHASIS config | ✅ |
| `surfemb/data/instance_tudl.py` | models_info fix | ✅ |
| `surfemb/scripts/train_matching_dino_unet_tudl.py` | Dataset/WandB | 📋 Manual |

---

## 🐛 Troubleshooting

**CUDA out of memory?**
```bash
--batch-size 2  # or even 1
```

**Dataset not found?**
Check paths in training script are absolute paths.

**More help:** See [CODE_CHANGES_APPLIED.md](CODE_CHANGES_APPLIED.md#troubleshooting)

---

## 📊 Training Data

**Current:** 50 placeholder images (for pipeline testing)
**Purpose:** Validate training code works
**Next:** Generate proper synthetic data after pipeline verified

---

## 🎯 Next Actions

1. [ ] Edit training script (3 files changes above)
2. [ ] Run quick tests
3. [ ] Test train for 100 steps
4. [ ] Debug any errors
5. [ ] Full training run

---

## 📝 Notes

- DINO features generated on Linux (took ~12 min with GPU)
- Current dataset is minimal (50 images) for testing only
- Model will "train" but not learn useful features yet
- Generate better synthetic data after pipeline works

---

## 🔗 Resources

- Original Paper: RAG-6DPose
- Base Code: [Surfemb](https://github.com/rasmushaugaard/surfemb)
- BOP Toolkit: https://github.com/thodan/bop_toolkit
- DINOv2: https://github.com/facebookresearch/dinov2

---

**Last Updated:** 2025-10-20
**Environment:** Linux, Python 3.12, PyTorch 2.6, CUDA 12.6
