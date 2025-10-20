# Code Changes Applied for CHASIS Training

**Date:** 2025-10-20
**Status:** Code updated, ready for testing

---

## ✅ Changes Applied

### 1. Model Paths Updated
**File:** [surfemb/workspace_dino/model_forward_c2f_dino_unet.py](surfemb/workspace_dino/model_forward_c2f_dino_unet.py) (lines 59-77)

**Changed:**
- ✅ Updated CAD model path to `/media/sujith/Project/NOARK_CV/RAG-6DPose-code/models/obj_000001.ply`
- ✅ Updated DINO features path to `/media/sujith/Project/NOARK_CV/RAG-6DPose-code/cad_features/obj_000001_dino_feat.pt`
- ✅ Loads CHASIS point cloud (66,230 vertices)
- ✅ Loads CHASIS DINO features (768-dim)
- ✅ Duplicates to p5/p6 for compatibility with 3-object code

### 2. Dataset Config Added
**File:** [surfemb/data/config.py](surfemb/data/config.py) (lines 29-37)

**Added:**
```python
config['chasis'] = chasis = DatasetConfig()
chasis.model_folder = 'models'
chasis.train_folder = 'train_real'
chasis.test_folder = 'test'
chasis.img_folder = 'rgb'
chasis.depth_folder = 'depth'
chasis.img_ext = 'png'
chasis.depth_ext = 'png'
```

### 3. Models Info Loading Fixed
**File:** [surfemb/data/instance_tudl.py](surfemb/data/instance_tudl.py) (lines 33-40)

**Changed:**
- ✅ Made models_info.json path dynamic (wrapped in try/except)
- ✅ Falls back gracefully if TUDL path doesn't exist

---

## 📋 Remaining Manual Changes Needed

You still need to update the training script manually:

### File: [surfemb/scripts/train_matching_dino_unet_tudl.py](surfemb/scripts/train_matching_dino_unet_tudl.py)

**Change 1 - Dataset root (around line 48):**
```python
# Find this line:
dataset_root = Path...

# Change to:
dataset_root = Path('/media/sujith/Project/NOARK_CV/RAG-6DPose-code/chasis_dataset')
```

**Change 2 - Config and n_objs (around lines 60-70):**
```python
# Find:
from surfemb.data.config import tudl as cfg
n_objs = 3

# Change to:
from surfemb.data.config import config
cfg = config['chasis']
n_objs = 1
```

**Change 3 - WandB API Key (line 115):**
```python
# Find:
os.environ["WANDB_API_KEY"] = ...

# Change to:
os.environ["WANDB_API_KEY"] = "your_key_from_wandb.ai"
```

Get your key from: https://wandb.ai/authorize

---

## 🧪 Testing the Changes

### Step 1: Quick Python Test

```bash
source /home/sujith/miniconda3/bin/activate r12

# Test if CHASIS config loads
python -c "
from surfemb.data.config import config
cfg = config['chasis']
print(f'✓ Config loaded: {cfg.train_folder}')
"

# Test if model paths work
python -c "
import open3d as o3d
import torch

cad_path = '/media/sujith/Project/NOARK_CV/RAG-6DPose-code/models/obj_000001.ply'
feat_path = '/media/sujith/Project/NOARK_CV/RAG-6DPose-code/cad_features/obj_000001_dino_feat.pt'

pcd = o3d.io.read_point_cloud(cad_path)
feat = torch.load(feat_path)

print(f'✓ CAD points: {len(pcd.points)}')
print(f'✓ DINO features: {feat.shape}')
"
```

**Expected output:**
```
✓ Config loaded: train_real
✓ CAD points: 66230
✓ DINO features: torch.Size([66230, 768])
```

### Step 2: Test Dataset Loading

```bash
python -c "
from pathlib import Path
from surfemb.data.config import config
from surfemb.data.instance_tudl import BopInstanceDataset

dataset_root = Path('/media/sujith/Project/NOARK_CV/RAG-6DPose-code/chasis_dataset')
cfg = config['chasis']

try:
    dataset = BopInstanceDataset(
        dataset_root=dataset_root,
        pbr=False,
        synt=False,
        test=False,
        cfg=cfg,
        obj_ids=[1],
        scene_ids=[1],
        show_progressbar=False
    )
    print(f'✓ Dataset loaded: {len(dataset)} instances')
    print(f'✓ Data folder: {dataset.data_folder}')
except Exception as e:
    print(f'✗ Error: {e}')
    import traceback
    traceback.print_exc()
"
```

**Expected output:**
```
/media/sujith/Project/NOARK_CV/RAG-6DPose-code/chasis_dataset/train_real
✓ Dataset loaded: 50 instances
✓ Data folder: /media/sujith/Project/NOARK_CV/RAG-6DPose-code/chasis_dataset/train_real
```

### Step 3: Test Training Script Imports

After making the manual changes to train_matching_dino_unet_tudl.py:

```bash
python -c "
import sys
sys.argv = ['', 'chasis', '--real', '--max-steps', '10']

# Just test imports, don't actually train
from surfemb.scripts import train_matching_dino_unet_tudl
print('✓ Training script imports successfully')
"
```

---

## 🚀 Running Training

After all tests pass, run training:

```bash
source /home/sujith/miniconda3/bin/activate r12

# Short test run (100 steps, ~5-10 minutes)
python -m surfemb.scripts.train_matching_dino_unet_tudl chasis \
    --real \
    --max-steps 100 \
    --batch-size 4

# Full training (after test succeeds)
python -m surfemb.scripts.train_matching_dino_unet_tudl chasis \
    --real \
    --max-steps 500000
```

---

## 🐛 Troubleshooting

### Error: "FileNotFoundError: models/obj_000001.ply"
**Fix:** Check absolute paths in `model_forward_c2f_dino_unet.py` (lines 60-61)

### Error: "KeyError: 'chasis'"
**Fix:** Make sure you added the chasis config in `config.py`

### Error: "No such file or directory: train_real"
**Fix:** Update dataset_root in training script to full absolute path

### Error: "models_info.json not found"
**Solution:** The code now handles this gracefully (try/except added)

### Error: "CUDA out of memory"
**Fix:** Reduce batch size: `--batch-size 2` or `--batch-size 1`

### Error: "WandB authentication failed"
**Fix:** Set your API key from https://wandb.ai/authorize

---

## 📊 What to Expect During Training

### Initialization (first 30 seconds):
```
Loading CAD model...
✓ Loaded 66230 points
✓ Loaded 66230 DINO features (768-dim)
Loading dataset...
✓ Found 50 training instances
Initializing model...
```

### Training Loop:
```
Epoch 0: [step 0] loss: 2.34, mask_loss: 0.45, nce_loss: 1.89
Epoch 0: [step 10] loss: 2.01, mask_loss: 0.42, nce_loss: 1.59
...
```

### Checkpoints:
- Saved to: `data/models/chasis-{wandb_run_id}/`
- Frequency: Every 10,000 steps (configurable)

### WandB Logging:
- View at: https://wandb.ai/your-username/your-project
- Metrics: loss, mask_loss, nce_loss, learning_rate
- Images: Predicted masks, embeddings visualizations

---

## ✅ Success Criteria

Training is working if:
- ✅ Model loads CHASIS data without errors
- ✅ Loss decreases over time
- ✅ No NaN/Inf in losses
- ✅ Checkpoints are saved
- ✅ GPU utilization is high (80-100%)
- ✅ WandB logs are updating

---

## 📝 Files Modified

| File | Changes | Status |
|------|---------|--------|
| `surfemb/workspace_dino/model_forward_c2f_dino_unet.py` | Updated CAD/DINO paths | ✅ Applied |
| `surfemb/data/config.py` | Added CHASIS config | ✅ Applied |
| `surfemb/data/instance_tudl.py` | Fixed models_info loading | ✅ Applied |
| `surfemb/scripts/train_matching_dino_unet_tudl.py` | Dataset root, n_objs, WandB | 📋 Manual |

---

## 🎯 Next Actions

1. **Apply manual changes** to training script
2. **Run Step 1 test** (config loading)
3. **Run Step 2 test** (dataset loading)
4. **Run short training** (100 steps)
5. **Debug any errors**
6. **Run full training** (if test succeeds)

---

## 💡 Notes

**About Placeholder Data:**
- Current dataset has 50 placeholder images
- Model will train but won't learn useful features
- This is OK for testing the pipeline!
- Once pipeline works, generate better synthetic data

**About Full Training:**
- Default: 500,000 steps
- With 50 images: ~2-3 hours
- With proper data (1000+ images): 1-3 days
- Monitor WandB to see if model is learning

**About Data Quality:**
- Placeholder data → Pipeline testing ✅
- Synthetic data (BlenderProc) → Better model 📈
- Real images → Best performance 🎯

---

**Current Status:** Code changes applied, ready for testing!
**Next Step:** Apply manual training script changes and run tests

Good luck! 🚀
