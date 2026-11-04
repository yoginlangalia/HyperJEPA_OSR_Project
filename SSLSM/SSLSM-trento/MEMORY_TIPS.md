# Memory Reduction Guide

## What Affects Memory Usage (Priority Order)

### 🔴 HIGH IMPACT (Change These First)

1. **batch_size** - Directly proportional to memory
   - Current: `batch_size = 4` ✅ (Already very low!)
   - Could reduce to: `batch_size = 2` if still having issues
   - **Impact**: ~50% reduction per halving

2. **patch_size** - Quadratic impact (patch_size²)
   - Current: `patch_size = 19`
   - Recommended: `patch_size = 17` or `15`
   - **Impact**: ~10-20% reduction per 2 units

### 🟡 MODERATE IMPACT

3. **encoder_depth** - Affects model size
   - Current: `encoder_depth = 8`
   - Recommended: `encoder_depth = 6` or `4`
   - **Impact**: ~15-20% reduction per 2 layers

4. **decoder_depth** - Smaller impact
   - Current: `decoder_depth = 2` ✅ (Already low)
   - Keep as is

### 🟢 LOW IMPACT

5. **pca_components** - Minimal impact
   - Current: `pca_components = 15` ✅ (Already low)
   - Keep as is

---

## Recommended Changes (If Still Getting Memory Errors)

### Option 1: Minimal Changes (Try This First)
```python
batch_size = 2        # Reduce from 4 to 2
patch_size = 17       # Reduce from 19 to 17
encoder_depth = 6     # Reduce from 8 to 6
```

**Expected Memory Reduction**: ~30-40%

### Option 2: More Aggressive (If Option 1 Fails)
```python
batch_size = 2        # Reduce from 4 to 2
patch_size = 15       # Reduce from 19 to 15
encoder_depth = 4     # Reduce from 8 to 4
```

**Expected Memory Reduction**: ~50-60%

### Option 3: Maximum Reduction (Last Resort)
```python
batch_size = 1        # Minimum batch size
patch_size = 13       # Smaller patches
encoder_depth = 4     # Smaller model
decoder_depth = 1     # Minimal decoder
```

**Expected Memory Reduction**: ~70-80%
**Warning**: Training will be slower, but should use minimal memory

---

## Understanding Batch Size Impact

**Batch size = 4** means:
- 4 samples processed simultaneously
- Memory = (sample_size) × 4

**Batch size = 2** means:
- 2 samples processed simultaneously  
- Memory = (sample_size) × 2
- **50% less memory!**

**Batch size = 1** means:
- 1 sample at a time
- Minimal memory but slower training

---

## Quick Reference: What to Change in main.py

```python
# Current (you have):
batch_size = 4        # ✅ Already optimized
patch_size = 19       # Change to 17 or 15
encoder_depth = 8     # Change to 6 or 4

# Recommended for Kaggle (low memory):
batch_size = 2        # Change this if needed
patch_size = 17       # Change this
encoder_depth = 6     # Change this
```

---

## Trade-offs

| Setting | Memory | Training Speed | Accuracy |
|---------|--------|----------------|----------|
| batch_size = 4 → 2 | ✅ Lower | ⚠️ Slower | ✅ Similar |
| patch_size = 19 → 17 | ✅ Lower | ✅ Faster | ⚠️ Slightly lower |
| encoder_depth = 8 → 6 | ✅ Lower | ✅ Faster | ⚠️ Slightly lower |

**Note**: Smaller batch sizes train slower but often give similar or better accuracy due to more frequent weight updates.

---

## Quick Fix for Your Code

In `main.py`, change lines 12-17:

```python
# Change from:
patch_size = 19
batch_size = 4  # (you already have this)
encoder_depth = 8

# To:
patch_size = 17      # or 15 if still issues
batch_size = 2       # if batch_size=4 still causes issues
encoder_depth = 6    # or 4 if still issues
```

---

## Testing Memory Usage

You can check GPU memory usage in Kaggle:

```python
import torch
print(f"GPU Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory Allocated: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
    print(f"Memory Reserved: {torch.cuda.memory_reserved()/1024**3:.2f} GB")
```

---

## Summary

**YES, batch_size directly reduces RAM usage!**

Since you already have `batch_size = 4`, I recommend:

1. **First try**: Reduce `patch_size = 19` → `17` and `encoder_depth = 8` → `6`
2. **If still fails**: Also reduce `batch_size = 4` → `2`
3. **Last resort**: Further reduce patch_size to 15 and encoder_depth to 4

The order of effectiveness:
1. **batch_size** (biggest impact, but you already reduced it)
2. **patch_size** (second biggest impact)
3. **encoder_depth** (moderate impact)

