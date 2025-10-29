# Checkpoint Storage Strategy

## File Size Breakdown

### Current Checkpoint Sizes

| File | Size | Contains |
|------|------|----------|
| `epoch_001_loss_2136.9434.pt` | 2.5 GB | Model + Optimizer state + metadata |
| `epoch_002_loss_2166.4612.pt` | 2.5 GB | Model + Optimizer state + metadata |
| `model_final.pt` | 1.3 GB | Model weights only |
| **Total** | **6.3 GB** | **2 epochs** |

### What's Inside Each Checkpoint?

**Full Checkpoint (2.5 GB):**
```python
{
    'epoch': 1,                          # Small (~1 KB)
    'model_state_dict': {...},           # Model weights (~1.3 GB)
    'optimizer_state_dict': {...},       # Optimizer state (~1.2 GB, mostly Adam momentum/variance)
    'loss': 2136.94,                     # Small (~1 B)
    'batch_losses': [8.15, 8.13, ...]   # Small (~2 KB)
}
```

**Model-Only (1.3 GB):**
```python
# Just the model weights - no optimizer state
# This is what you actually need for inference
```

## When You Need Each Type

### For Inference Only (Deployment)
✅ Need: `model_final.pt` (1.3 GB)
❌ Don't Need: Full checkpoints (2.5 GB each)

**Use case:** Running predictions on new data
```python
model = Gen_corr(...)
model.load_state_dict(torch.load('simple_train_output/model_final.pt'))
model.eval()
# Inference...
```

### For Resuming Training
✅ Need: Latest full checkpoint (2.5 GB)
❌ Don't Need: All previous checkpoints
❌ Don't Need: `model_final.pt` (can regenerate it)

**Use case:** Training interrupted, want to continue from last epoch
```python
checkpoint = torch.load('simple_train_output/epoch_002_loss_2166.4612.pt')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch'] + 1
# Continue training...
```

### For Best Model Selection
✅ Need: Best K checkpoints (e.g., top 3 by loss)
❌ Don't Need: All intermediate checkpoints

**Use case:** Model validation and comparison
```python
best_checkpoints = [
    'epoch_001_loss_2136.9434.pt',  # Best epoch
    'epoch_002_loss_2166.4612.pt',  # 2nd best
    'epoch_003_loss_2145.1234.pt',  # 3rd best
]
# Keep only these 3, delete others
```

## Recommended Strategies

### Strategy 1: Inference Only (Recommended for Current Setup)
**Keep only:** `model_final.pt`
**Delete:** All `epoch_*.pt` files
**Storage:** 1.3 GB per trained model
**Tradeoff:** Can't resume training, but that's fine if training is complete

```bash
# Keep only the final model
rm simple_train_output/epoch_*.pt
ls -lh simple_train_output/  # Should show only model_final.pt
```

### Strategy 2: Resume Training + Inference (Balanced)
**Keep:** Last epoch checkpoint + `model_final.pt`
**Delete:** All earlier epoch checkpoints
**Storage:** 3.8 GB per model (2.5 GB + 1.3 GB)
**Tradeoff:** Can resume training OR do inference, but costs extra disk space

```bash
# Keep only the latest epoch and final model
rm simple_train_output/epoch_001_*.pt
# Keep: epoch_002_*.pt + model_final.pt
```

### Strategy 3: Selective Best Checkpoints (Advanced)
**Keep:** Best K checkpoints by validation metric
**Delete:** All others
**Storage:** K × 2.5 GB (e.g., 7.5 GB for top 3)
**Tradeoff:** Can resume from best or 2nd-best epoch, good for hyperparameter tuning

```bash
# Keep top 3 checkpoints by loss, delete others
# Useful when training multiple runs
```

## Modified Training Strategy

### Option A: Minimal (Inference Only)

Don't save full checkpoints, only save final model:

```python
# In simple_train.py, remove:
# torch.save(checkpoint, checkpoint_path)

# Keep only:
torch.save(model.state_dict(), final_model_path)
```

**Result:**
- Saves ~1.3 GB per training run
- Can't resume training if interrupted
- Perfect for final deployments

### Option B: Last Checkpoint Only (Balanced)

Save only the latest checkpoint (overwrite each epoch):

```python
latest_checkpoint_path = OUTPUT_DIR / "latest_checkpoint.pt"

for epoch in range(NUM_EPOCHS):
    # ... training ...

    checkpoint = {
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_loss,
    }
    # Overwrite previous checkpoint (keeps only latest)
    torch.save(checkpoint, latest_checkpoint_path)
```

**Result:**
- Saves only 2.5 GB (latest checkpoint)
- Can resume training from last epoch
- Cannot go back to earlier epochs

### Option C: Keep Best K Checkpoints (Advanced)

```python
best_k_losses = []
best_k_checkpoints = []

for epoch in range(NUM_EPOCHS):
    # ... training ...

    # Track best K checkpoints
    best_k_losses.append(avg_loss)
    checkpoint_path = OUTPUT_DIR / f"epoch_{epoch+1:03d}_loss_{avg_loss:.4f}.pt"
    torch.save(checkpoint, checkpoint_path)

    # Keep only top K by loss
    if len(best_k_losses) > 3:
        # Find worst of the best K
        worst_idx = np.argmax(best_k_losses)
        # Delete old checkpoint
        old_checkpoint = best_k_checkpoints[worst_idx]
        old_checkpoint.unlink()  # Delete file
        # Update tracking
        best_k_losses.pop(worst_idx)
        best_k_checkpoints.pop(worst_idx)

    best_k_checkpoints.append(checkpoint_path)
```

**Result:**
- Saves up to 7.5 GB for top 3 checkpoints
- Can resume from best or alternate epochs
- Good for comparing model versions

## Storage Cost Analysis

### Scenario 1: 10 Epoch Training

| Strategy | Checkpoints Kept | Storage |
|----------|-----------------|---------|
| Inference Only | 0 | 1.3 GB |
| Last Checkpoint | 1 | 2.5 GB |
| **Current (All Epochs)** | **10** | **27.5 GB** |
| Best 3 | 3 | 7.5 GB |

### Scenario 2: Multi-GPU Training (8 epochs, multiple runs)

| Strategy | Per Run | 5 Runs | Archive |
|----------|---------|--------|---------|
| Inference Only | 1.3 GB | 6.5 GB | ✅ Fast |
| Last Checkpoint | 2.5 GB | 12.5 GB | ⚠️ Medium |
| **Current** | **25 GB** | **125 GB** | ❌ Large |
| Best 3 | 7.5 GB | 37.5 GB | ⚠️ Medium |

## My Recommendation

For your current setup, I recommend **Strategy 1: Inference Only**

### Why?
1. **Your training completes successfully** - no need to resume interrupted training
2. **You only need the final model** - for running inference and generating poses
3. **Storage is precious** - saves 5 GB per training run (1.3 GB vs 6.3 GB)
4. **Inference is fast enough** - only ~11 seconds for 100 samples

### Implementation:

Simply modify `simple_train.py` to NOT save full checkpoints:

```python
# Remove this section:
# checkpoint_path = OUTPUT_DIR / f"epoch_{epoch+1:03d}_loss_{avg_loss:.4f}.pt"
# torch.save(checkpoint, checkpoint_path)

# Keep only this:
final_model_path = OUTPUT_DIR / "model_final.pt"
torch.save(model.state_dict(), final_model_path)
```

### Result:
- ✅ 1.3 GB per training run (instead of 6.3 GB)
- ✅ Training history still saved (JSON, tiny file)
- ✅ Can do inference and visualization
- ✅ Clean, organized outputs
- ❌ Cannot resume if training interrupted (acceptable for quick training)

## If You Need Resume Capability

Use **Strategy 2: Last Checkpoint Only**

```python
latest_checkpoint_path = OUTPUT_DIR / "latest_checkpoint.pt"

for epoch in range(NUM_EPOCHS):
    # ... training ...

    checkpoint = {
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_loss,
    }
    torch.save(checkpoint, latest_checkpoint_path)  # Overwrite
```

### Result:
- ✅ 2.5 GB per training run
- ✅ Can resume from last epoch
- ✅ Save disk space vs current approach
- ❌ Cannot go back to earlier epochs

## Quick Decision Tree

```
Do you need to resume training if interrupted?
├─ NO  → Use Inference Only (1.3 GB) ✅ Recommended
└─ YES → Do you need to try multiple epochs?
    ├─ NO  → Use Last Checkpoint (2.5 GB)
    └─ YES → Use Best K (K × 2.5 GB, e.g., 7.5 GB for K=3)
```

For your current setup: **Go with Inference Only** ✅
