# Checkpoint Management - Quick Reference

## TL;DR

**For your setup:** Change line 36 in `simple_train.py` to use the strategy you want:

```python
CHECKPOINT_STRATEGY = 'inference_only'  # Recommended: 1.3 GB total
# or
CHECKPOINT_STRATEGY = 'last_checkpoint'  # Alternative: 2.5 GB total (can resume)
# or
CHECKPOINT_STRATEGY = 'all_epochs'  # Original: 2.5 GB × NUM_EPOCHS (not recommended)
```

## The 3 Strategies Explained in 30 Seconds

### 1️⃣ Inference Only (RECOMMENDED) ✅
- **Saves:** Only `model_final.pt` (1.3 GB)
- **Use case:** Training finished, just want to run inference
- **Can resume training:** ❌ No
- **Storage per run:** 1.3 GB
- **For 10 epochs:** Still 1.3 GB

```bash
# Training output
simple_train_output/
├── model_final.pt           (1.3 GB) ← Only this is saved
└── training_history.json    (tiny)
```

### 2️⃣ Last Checkpoint Only (BALANCED) ⚖️
- **Saves:** Only `latest_checkpoint.pt` (2.5 GB)
- **Use case:** Training might be interrupted, need to resume from last epoch
- **Can resume training:** ✅ Yes (from last epoch only)
- **Storage per run:** 2.5 GB
- **For 10 epochs:** Still 2.5 GB (overwrites each epoch)

```bash
# Training output
simple_train_output/
├── latest_checkpoint.pt     (2.5 GB) ← Overwrites each epoch
└── training_history.json    (tiny)
```

### 3️⃣ All Epochs (ORIGINAL) ❌
- **Saves:** `epoch_001.pt`, `epoch_002.pt`, ... (2.5 GB each)
- **Use case:** Comparing multiple epochs, want all checkpoints
- **Can resume training:** ✅ Yes (from any epoch)
- **Storage per run:** 2.5 GB × NUM_EPOCHS
- **For 10 epochs:** 25 GB (NOT RECOMMENDED)

```bash
# Training output (10 epochs = 25 GB!)
simple_train_output/
├── epoch_001_loss_2136.9434.pt  (2.5 GB) ← ALL saved
├── epoch_002_loss_2166.4612.pt  (2.5 GB)
├── epoch_003_loss_2145.1234.pt  (2.5 GB)
├── ... (7 more)
└── training_history.json        (tiny)
```

## Quick Decision

**Choose ONE:**

```
                    ╔════════════════════════════════════╗
                    ║  "Will I need to resume training?" ║
                    ╚════════════════════════════════════╝
                                    │
                        ┌───────────┴──────────┐
                        │                      │
                       NO                     YES
                        │                      │
                        ▼                      ▼
        ┌──────────────────────┐  ┌─────────────────────────┐
        │  inference_only ✅   │  │  last_checkpoint ⚖️     │
        │  (1.3 GB)            │  │  (2.5 GB)               │
        │  Best for:           │  │  Best for:              │
        │  - Finished training │  │  - Interrupted training │
        │  - Inference only    │  │  - Want to resume       │
        │  - Minimal storage   │  │  - Balanced approach    │
        └──────────────────────┘  └─────────────────────────┘
```

## File Size Comparison

| Epochs Trained | Strategy | Storage | Data/Epoch |
|---|---|---|---|
| 1 | inference_only | **1.3 GB** | 1.3 GB |
| 1 | last_checkpoint | 2.5 GB | 2.5 GB |
| 1 | all_epochs | 2.5 GB | 2.5 GB |
| 5 | inference_only | **1.3 GB** | 0.3 GB |
| 5 | last_checkpoint | **2.5 GB** | 0.5 GB |
| 5 | all_epochs | **12.5 GB** | 2.5 GB |
| 10 | inference_only | **1.3 GB** | 0.1 GB |
| 10 | last_checkpoint | **2.5 GB** | 0.3 GB |
| 10 | all_epochs | **25 GB** | 2.5 GB |

## How to Use Each Strategy

### Strategy 1: Inference Only ✅ (Recommended)

```bash
# Edit simple_train.py line 36
CHECKPOINT_STRATEGY = 'inference_only'

# Run training
python simple_train.py

# Output (1.3 GB)
# simple_train_output/
# ├── model_final.pt           (1.3 GB)
# └── training_history.json

# Do inference
python infer_poses.py

# Visualize
python simple_pose_viz.py --poses inference_output/estimated_poses.json
```

### Strategy 2: Last Checkpoint (for Resume)

```bash
# Edit simple_train.py line 36
CHECKPOINT_STRATEGY = 'last_checkpoint'

# Run training
python simple_train.py

# Output (2.5 GB)
# simple_train_output/
# ├── latest_checkpoint.pt     (2.5 GB)
# ├── model_final.pt           (1.3 GB)
# └── training_history.json

# Resume if interrupted
python simple_train_resume.py  # (create this script using checkpoint loading code)

# Or do inference
python infer_poses.py
```

### Strategy 3: All Epochs (Original - Not Recommended)

```bash
# Edit simple_train.py line 36
CHECKPOINT_STRATEGY = 'all_epochs'

# Run training
python simple_train.py

# Output (6.3 GB for 2 epochs)
# simple_train_output/
# ├── epoch_001_loss_2136.9434.pt  (2.5 GB)
# ├── epoch_002_loss_2166.4612.pt  (2.5 GB)
# ├── model_final.pt               (1.3 GB)
# └── training_history.json
```

## What Can You Do With Each?

| Task | inference_only | last_checkpoint | all_epochs |
|------|---|---|---|
| Run inference | ✅ | ✅ | ✅ |
| Resume training | ❌ | ✅ (last epoch) | ✅ (any epoch) |
| Compare epochs | ❌ | ❌ | ✅ |
| Find best model | ❌ | ❌ | ✅ |
| Minimal storage | ✅ | ⚠️ | ❌ |

## Recommendation Summary

### Current Setup (100 samples, 2 epochs)
✅ **Use `inference_only`**
- Training is fast and completes reliably
- No need to resume interrupted training
- Saves 5 GB of disk space
- Inference takes only ~11 seconds

### Large Dataset (thousands of samples, 100 epochs)
⚖️ **Use `last_checkpoint`**
- Training might take days
- Resuming from last epoch saves time
- Storage overhead is acceptable (2.5 GB fixed)
- Inference is still fast

### Research/Debugging (many runs, need to compare)
ℹ️ **Use `all_epochs`**
- Comparing multiple model versions
- Need to analyze per-epoch performance
- Storage is not a concern
- Have tools to manage multiple checkpoints

## How to Switch Strategies

Super easy - one-line change in simple_train.py:

```python
# Line 36 - Change this:
CHECKPOINT_STRATEGY = 'inference_only'  # ← Just change the string
```

That's it! No other code changes needed.

## Storage Cleanup

### If You're Using Old Checkpoints

```bash
# Clean up old checkpoints
cd simple_train_output

# Keep only inference models
rm epoch_*.pt latest_checkpoint.pt

# Keep only the final model for inference
ls -lh model_final.pt  # Should be ~1.3 GB
```

### Archive Strategy

For archiving trained models:

```bash
# Archive with minimal storage
tar -czf model_run_1.tar.gz simple_train_output/model_final.pt simple_train_output/training_history.json
# Creates ~500 MB compressed

# Only decompress when needed for inference
tar -xzf model_run_1.tar.gz
```

## Example Output

### With `inference_only` (Recommended)

```
============================================================
Training completed successfully!
============================================================

Checkpoint Strategy: INFERENCE_ONLY
Output directory: simple_train_output

Saved files:
  - model_final.pt                                 (  1297.2 MB)
  - training_history.json                          (    0.0 MB)

  Total storage used: 1.27 GB

Next step: Run inference
  python infer_poses.py

Then visualize:
  python simple_pose_viz.py --poses inference_output/estimated_poses.json
```

### With `last_checkpoint`

```
============================================================
Training completed successfully!
============================================================

Checkpoint Strategy: LAST_CHECKPOINT
Output directory: simple_train_output

Saved files:
  - latest_checkpoint.pt                           (  2486.1 MB)
  - model_final.pt                                 (  1297.2 MB)
  - training_history.json                          (    0.0 MB)

  Total storage used: 3.78 GB

Next step: Run inference
  python infer_poses.py
```

## Final Tips

1. **Start with `inference_only`** - It's the most efficient for completed training
2. **Switch to `last_checkpoint`** - Only if training is long or unreliable
3. **Avoid `all_epochs`** - Unless you have specific research needs
4. **Archive old runs** - Use tar.gz to compress trained models by ~60%
5. **Monitor storage** - Check `training_history.json` to track progress

---

**Questions?** See [CHECKPOINT_STRATEGY.md](CHECKPOINT_STRATEGY.md) for detailed explanation.
