# Training Pipeline - Issues Found and Fixes

## Summary
Attempted to run the training pipeline with 1-2 epochs on the CHASIS dataset. Multiple issues were identified and fixed during the process. Training successfully initializes but encounters a data pipeline issue.

## Issues Found & Fixed

### ✅ Issue 1: Module Import Path Errors
**Error**: `ModuleNotFoundError: No module named 'surfemb.data.instance'`

**Root Cause**: The codebase uses `instance_tudl.py` but imports reference a non-existent `instance.py` module.

**Files Affected**:
- `surfemb/data/__init__.py`
- `surfemb/data/pose_auxs.py`
- `surfemb/data/std_auxs.py`

**Fix Applied**:
```python
# Before
from .instance import BopInstanceAux

# After
from .instance_tudl import BopInstanceAux
```

**Status**: ✅ Fixed

---

### ✅ Issue 2: Incorrect Module Name (surfemb_t vs surfemb)
**Error**: `ModuleNotFoundError: No module named 'surfemb_t'`

**Root Cause**: Line 22 in `model_forward_c2f_dino_unet.py` has a typo: `surfemb_t` instead of `surfemb`

**Location**: `surfemb/workspace_dino/model_forward_c2f_dino_unet.py:22`

**Fix Applied**:
```python
# Before
import surfemb_t.workspace_dino.modules.imagenet as imagenet

# After
import surfemb.workspace_dino.modules.imagenet as imagenet
```

**Status**: ✅ Fixed

---

### ✅ Issue 3: BopInstanceDataset Constructor Signature Mismatch
**Error**: `TypeError: BopInstanceDataset.__init__() got an unexpected keyword argument 'split'`

**Root Cause**: The actual constructor expects different parameters than used in quick_train.py

**Correct Signature**:
```python
def __init__(
    self, dataset_root: Path, pbr: bool, synt: bool, test: bool, cfg: DatasetConfig,
    obj_ids: Sequence[int],
    scene_ids=None, min_visib_fract=0.1, min_px_count_visib=1024,
    auxs: Sequence['BopInstanceAux'] = tuple(), show_progressbar=True,
    infer_idx = None
)
```

**Fix Applied**:
```python
train_ds = BopInstanceDataset(
    dataset_root=DATASET_ROOT,
    pbr=False,
    synt=False,
    test=False,
    cfg=data_config.config['chasis'],
    obj_ids=[1],  # obj_000001
    min_visib_fract=0.1,
    min_px_count_visib=1024,
    auxs=[],
)
```

**Status**: ✅ Fixed

---

### ✅ Issue 4: Gen_corr Model Expects 3 Objects
**Error**: `IndexError: list index out of range` at line 125 of `model_forward_c2f_dino_unet.py`

**Root Cause**: The `Gen_corr` model is hardcoded to expect 3 objects:
```python
self.scales = [objs[0].scale, objs[1].scale, objs[2].scale]
self.offsets = [objs[0].offset, objs[1].offset, objs[2].offset]
```

But we only loaded 1 object (CHASIS).

**Fix Applied**: Duplicate the loaded object to match expected count:
```python
objs, loaded_obj_ids = load_objs(MODELS_PATH, obj_ids=obj_ids, show_progressbar=False)

# Model expects 3 objects, duplicate if we have fewer
while len(objs) < 3:
    objs.append(objs[-1])  # Duplicate the last object
```

**Status**: ✅ Fixed

---

### ✅ Issue 5: PyTorch Lightning `gpus` Parameter Deprecated
**Error**: `TypeError: Trainer.__init__() got an unexpected keyword argument 'gpus'`

**Root Cause**: Newer versions of PyTorch Lightning (≥2.0) replaced the `gpus` parameter with `devices`

**Old Code**:
```python
trainer = pl.Trainer(
    max_epochs=NUM_EPOCHS,
    gpus=GPUS if torch.cuda.is_available() else None,
    accelerator='gpu' if torch.cuda.is_available() else 'cpu',
    ...
)
```

**Fix Applied**:
```python
trainer_kwargs = {
    'max_epochs': NUM_EPOCHS,
    'accelerator': 'gpu' if torch.cuda.is_available() else 'cpu',
    'callbacks': [checkpoint_callback],
    'log_every_n_steps': 1,
    'enable_checkpointing': True,
    'default_root_dir': OUTPUT_DIR,
}

if torch.cuda.is_available():
    trainer_kwargs['devices'] = GPUS

trainer = pl.Trainer(**trainer_kwargs)
```

**Status**: ✅ Fixed

---

### ⚠️ Issue 6: Data Pipeline Batch Key Mismatch - rgb_crop Missing
**Error**: `KeyError: 'rgb_crop'` at line 303 of `model_forward_c2f_dino_unet.py`

**Root Cause**: The model's `training_step` method expects batch data with key `'rgb_crop'`:
```python
img = batch['rgb_crop'].float().permute(0,3,1,2)  # (B, 3, H, W)
```

But the dataset loader only provides `'rgb'` (the full image), not the cropped version.

**Status**: ⚠️ **REQUIRES PROPER IMPLEMENTATION**

---

### ⚠️ Issue 7: Data Augmentation Missing - K_crop Not Available
**Error**: `KeyError: 'K_crop'` at line 26 of `pose_auxs.py`

**Root Cause**: The `ObjCoordAux` class expects camera intrinsic matrix to be pre-computed and stored as `'K_crop'`:
```python
K = inst['K_crop'].copy()  # Line 26 in pose_auxs.py
```

This key is not created by the raw dataset - it requires preprocessing that:
1. Loads cropped camera parameters
2. Computes image crops
3. Adjusts intrinsics for the crop

**Status**: ⚠️ **REQUIRES PROPER IMPLEMENTATION**

---

## Full Data Pipeline Requirements

The complete data pipeline requires:

### Input Data (from BOP format):
```
rgb/          # Full RGB images
depth/        # Depth maps
mask_visib/   # Visibility masks
scene_gt.json # Ground truth poses
scene_camera.json # Camera intrinsics
```

### Expected Batch Output (for training_step):
```python
batch = {
    'rgb_crop': tensor,           # Cropped RGB image (B, H, W, 3)
    'mask_visib_crop': tensor,    # Cropped visibility mask
    'K_crop': np.array,           # Adjusted intrinsics for crop
    'cam_R_obj': np.array,        # Rotation matrix (object to camera)
    'cam_t_obj': np.array,        # Translation vector
    'obj_idx': int,               # Object ID
    # ... other fields
}
```

### Current Status:
The raw dataset provides only:
- `rgb` - Full RGB image
- `mask_visib` - Full visibility mask
- Camera info in `scene_camera.json`
- Pose info in `scene_gt.json`

### What's Missing:
A preprocessing module that:
1. ✅ Loads RGB → `RgbLoader` partially handles this
2. ❌ Crops RGB to get `rgb_crop` → Needs proper crop computation
3. ❌ Crops mask to get `mask_visib_crop` → Needs proper crop computation
4. ❌ Adjusts camera matrix → Needs `K_crop` computation from crop region
5. ❌ Loads pose information → Needs pose loader aux

### Temporary Workaround for Testing:
Until the full pipeline is implemented, we can:
1. Skip augmentation and let the model handle raw RGB
2. Create a simple crop utility or modify the training_step

---

## Summary of Changes Made

| Issue | File | Line | Type | Status |
|-------|------|------|------|--------|
| Module import (instance) | `surfemb/data/__init__.py` | 1 | Fix | ✅ |
| Module import (pose_auxs) | `surfemb/data/pose_auxs.py` | 5 | Fix | ✅ |
| Module import (std_auxs) | `surfemb/data/std_auxs.py` | 6 | Fix | ✅ |
| Module typo (surfemb_t) | `surfemb/workspace_dino/model_forward_c2f_dino_unet.py` | 22 | Fix | ✅ |
| Dataset constructor | `quick_train.py` | 89-99 | Fix | ✅ |
| Object duplication | `quick_train.py` | 125-133 | Fix | ✅ |
| PyTorch Lightning | `quick_train.py` | 174-188 | Fix | ✅ |
| Data pipeline | N/A | N/A | Pending | ⚠️ |

---

## Next Steps

### To Complete Training

1. **Add data augmentation to quick_train.py**:

```python
from surfemb.data.std_auxs import RgbLoader
from surfemb.data.pose_auxs import ObjCoordAux

# Inside quick_train.py main() function, update dataset loading:
auxs = [
    RgbLoader(),  # Loads RGB images as 'rgb'
    ObjCoordAux(objs=objs, res=RES_CROP),  # Creates 'rgb_crop', 'mask_visib_crop', etc.
]

train_ds = BopInstanceDataset(
    dataset_root=DATASET_ROOT,
    pbr=False,
    synt=False,
    test=False,
    cfg=ds_config,
    obj_ids=obj_ids,
    min_visib_fract=0.1,
    min_px_count_visib=1024,
    auxs=auxs,  # Add augmentation
)
```

2. **Run training again**:
```bash
source /home/sujith/miniconda3/bin/activate r12
python quick_train.py
```

---

## Environment & Versions

- **Python**: 3.12
- **PyTorch**: Latest (in r12 env)
- **PyTorch Lightning**: ≥2.0 (uses `devices` instead of `gpus`)
- **CUDA**: Available (GPU training enabled)
- **Dataset**: CHASIS (1 object with 100 training samples)

---

## Key Insights

1. **Hardcoded 3-Object Limitation**: The `Gen_corr` model is designed for 3 objects and duplication is a workaround. For multi-object training, the model architecture needs refactoring.

2. **Data Pipeline Complexity**: The model expects preprocessed data with cropped images and augmented masks. The dataset loader must include the proper auxs modules.

3. **Import Organization Issue**: The codebase has outdated import paths (references to non-existent modules) that need updating.

4. **PyTorch Lightning Version Compatibility**: The code was written for older PyTorch Lightning versions and needs updates for ≥2.0.

---

## Recommendations

1. ✅ Apply all fixes above before running training
2. ⚠️ Properly configure data augmentation pipeline
3. 📝 Update documentation with corrected import paths
4. 🔧 Refactor `Gen_corr` to support flexible number of objects
5. 🧪 Test with 1-2 epochs to validate before full training

