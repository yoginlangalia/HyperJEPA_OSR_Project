# 🔍 Deep Analysis: SSLSM-houston vs SSLSM-trento

> **Generated on**: 2025-12-22  
> **Purpose**: Comprehensive comparison of differences between Houston and Trento dataset implementations

---

## 📁 Structural Differences

| Aspect | SSLSM-houston | SSLSM-trento |
|--------|---------------|--------------|
| **Dataset** | houston.mat (168 MB) | trento.mat (12 MB) |
| **Data Keys** | `input`, `TR`, `TE` | `HSI_data`, `ground` |
| **Built-in Split** | ✅ Yes (TR/TE predefined) | ❌ No (single ground truth) |
| **Documentation** | DIFFERENCES_ANALYSIS.md | MEMORY_TIPS.md |
| **Results Path** | `./results/houston/` | `./results/trento/` |
| **Model Paths** | `net_houston_*.pkl` | `net_trento_*.pkl` |

---

## 🔧 Hyperparameter Differences

| Parameter | Houston | Trento | Notes |
|-----------|---------|--------|-------|
| **class_num** | 16 | 7 | Trento has fewer land cover classes |
| **EPOCH** | 100 | 50 | Houston uses more epochs |
| **patch_size** | 15 | 19 | Trento uses larger patches |
| **pca_components** | 15 | 63 | Trento uses ALL bands (no PCA) |
| **batch_size** | 64 | 32 | Houston uses larger batches |
| **encoder_depth** | 8 | 8 | Same |
| **decoder_depth** | 2 | 2 | Same |
| **mask_ratio** | 0.5 | 0.5 | Same |
| **seed** | 42 | 42 | Same |

---

## 📝 File-by-File Detailed Comparison

### 1. `main.py` - Major Differences

#### **Imports**
```python
# Houston
from utils import applyPCA, createImageCubes

# Trento
from utils import createImageCubes, splitTrainTestSet  # No PCA import!
```

#### **Data Loading**

| Aspect | Houston | Trento |
|--------|---------|--------|
| **Mat Keys** | `data['input']`, `data['TR']`, `data['TE']` | `data['HSI_data']`, `data['ground']` |
| **Split Method** | Predefined TR/TE in dataset | 80/20 random split via `splitTrainTestSet()` |
| **PCA** | ✅ Uses `applyPCA()` | ❌ No PCA (uses all 63 bands) |

**Houston Data Loading:**
```python
data = sio.loadmat('./data/houston.mat')
X = data['input']
TR = data['TR']  # Training labels (predefined)
TE = data['TE']  # Test labels (predefined)

X_pca = applyPCA(X, numComponents=pca_components)  # PCA applied!
X_TR, Y_TR = createImageCubes(X_pca, TR, ...)
X_TE, Y_TE = createImageCubes(X_pca, TE, ...)
```

**Trento Data Loading:**
```python
data = sio.loadmat('./data/trento.mat')
X = data['HSI_data']  # Hyperspectral image data
y = data['ground']    # Ground truth labels (single matrix)

# NO PCA - uses raw X directly!
X_full, Y_full = createImageCubes(X, y, ...)  # All labeled pixels
X_TR, X_TE, Y_TR, Y_TE = splitTrainTestSet(X_full, Y_full, testRatio=0.2, ...)  # 80/20 split
```

#### **Train Function Call**

**Houston (simplified - no data leakage):**
```python
accuracy = train(X_TR, Y_TR, X_TE, Y_TE, class_num, N_RUNS, ...)
```

**Trento (uses X_full for pretraining):**
```python
accuracy = train(X_full, Y_full, X_TR, Y_TR, X_TE, Y_TE, ratio_self, None, class_num, N_RUNS, ...)
```

---

### 2. `train.py` - Significant Differences

#### **Function Signature**

**Houston:**
```python
def train(X_TR, Y_TR, X_TE, Y_TE, class_num, N_RUNS,
          patch_size, pca_components, batch_size, device, EPOCH, path, fine_path, 
          mask_ratio, encoder_depth, decoder_depth, seed):
```

**Trento:**
```python
def train(X_full, Y_full, X_TR, Y_TR, X_TE, Y_TE, test_ratio, ratio_fine, class_num, N_RUNS,
          patch_size, pca_components, batch_size, device, EPOCH, path, fine_path, 
          mask_ratio, encoder_depth, decoder_depth, seed):
```

#### **Pretraining Data Source**

| Aspect | Houston | Trento |
|--------|---------|--------|
| **Data Source** | 100% of X_TR only | 50% of X_full (all labeled pixels) |
| **Data Leakage** | ❌ No (TR only) | ⚠️ Possible (full data includes test area pixels) |

**Houston Pretraining:**
```python
# Pre-training: Use 100% of X_TR (NO data leakage)
Xtrain_pretrain = X_TR.reshape(-1, patch_size, patch_size, pca_components)
Xtrain_pretrain = Xtrain_pretrain.transpose(0, 3, 1, 2)
Y_pretrain_dummy = np.zeros(len(Xtrain_pretrain))
```

**Trento Pretraining:**
```python
# Pre-training uses ratio_self (50%) of X_full for pre-training
num_samples = X_full.shape[0]
indices = np.arange(num_samples)
train_idx, _, _, _ = splitTrainTestSet(indices, Y_full, test_ratio, randomState=seed)
Xtrain_pretrain = X_full[train_idx]
Xtrain_pretrain = Xtrain_pretrain.transpose(0, 3, 1, 2)
```

#### **Training Loop Differences**

| Feature | Houston | Trento |
|---------|---------|--------|
| **Progress Printing** | ❌ No epoch progress | ✅ Every 10 epochs |
| **Memory Cleanup** | ❌ No explicit cleanup | ✅ `del loss, data` + `torch.cuda.empty_cache()` |
| **Loss Handling** | `epoch_loss += loss / len(train_loader)` | `epoch_loss += loss.detach().item() / len(train_loader)` |

**Trento's Memory Optimizations (not in Houston):**
```python
# Detach loss from computation graph to prevent memory accumulation
epoch_loss += loss.detach().item() / len(train_loader)
# Clear intermediate tensors to free memory
del loss, data

# Print progress every 10 epochs
if (epoch + 1) % 10 == 0 or (epoch + 1) == EPOCH*2:
    elapsed_time = time.time() - time1
    print(f"  Epoch [{epoch+1}/{EPOCH*2}] - Time: {elapsed_time:.2f}s")
    # Clear GPU cache periodically to prevent memory fragmentation
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
```

---

### 3. `train_fine.py` - Minor Differences

| Feature | Houston | Trento |
|---------|---------|--------|
| **Function Signature** | Same | Same |
| **Progress Printing** | ❌ No progress | ✅ Every 10 epochs |
| **Memory Cleanup** | ❌ No cleanup | ✅ `del loss, data, outs, label` |
| **Test Info** | ❌ None | ✅ `print(f"Testing on {len(testset):,} samples...")` |

**Trento's Additional Features:**
```python
print(f"  Fine-tuning: {len(Xtrain):,} samples | Test: {len(Xtest):,} samples | Epochs: {EPOCH}")

# Clear intermediate tensors to free memory
del loss, data, outs, label

# Print progress every 10 epochs
if (epoch + 1) % 10 == 0 or (epoch + 1) == EPOCH:
    elapsed_time = time.time() - time1
    print(f"    Fine-tuning Epoch [{epoch+1}/{EPOCH}] - Time: {elapsed_time:.2f}s")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

print(f"  Testing on {len(testset):,} samples...")
```

---

### 4. `utils.py` - **IDENTICAL** ✅

Both files are exactly the same (7,075 bytes each).

---

### 5. `net/sslsm.py` and `net/vit.py` - **IDENTICAL** ✅

The neural network architectures are the same in both versions.

---

### 6. `ColorFunctions.py` - **IDENTICAL** ✅

---

## 📊 Summary of Key Differences

### **Dataset Characteristics**

| Feature | Houston | Trento |
|---------|---------|--------|
| **Number of Classes** | 16 | 7 |
| **Spectral Bands** | 144 → 15 (PCA) | 63 (no PCA) |
| **Spatial Size** | 349 × 1905 | Smaller |
| **Built-in Split** | ✅ TR/TE predefined | ❌ Random 80/20 |
| **Data File Size** | 168 MB | 12 MB |

### **Pretraining Protocol**

| Aspect | Houston | Trento |
|--------|---------|--------|
| **Data Used** | 100% of TR only | 50% of all labeled pixels |
| **Data Leakage Risk** | ❌ None | ⚠️ Possible |
| **PCA Applied** | ✅ Yes (15 components) | ❌ No (all 63 bands) |

### **Memory Optimizations**

| Feature | Houston | Trento |
|---------|---------|--------|
| **Epoch Progress** | ❌ | ✅ |
| **Memory Cleanup** | ❌ | ✅ |
| **Cache Clearing** | ❌ | ✅ |
| **Detached Loss** | ❌ | ✅ |

### **Training Configuration**

| Parameter | Houston | Trento | Difference |
|-----------|---------|--------|------------|
| EPOCH | 100 | 50 | Houston 2× more |
| batch_size | 64 | 32 | Houston 2× larger |
| patch_size | 15 | 19 | Trento +4 pixels |
| pca_components | 15 | 63 | Trento uses all bands |

---

## ⚠️ Important Notes

1. **PCA Difference**: Houston applies PCA (144→15 bands), Trento uses all 63 bands directly. This affects model input dimensions.

2. **Data Split**: Houston has predefined TR/TE splits in the dataset file. Trento creates an 80/20 split at runtime.

3. **Pretraining Protocol**: 
   - Houston: Uses only training set (no leakage)
   - Trento: Uses all labeled pixels with 50% split (potential leakage)

4. **Memory Optimizations**: Trento has extensive memory cleanup code that Houston lacks. This is likely due to Trento's larger patch_size (19 vs 15) and full band usage (63 vs 15).

5. **Training Time**: Despite lower epochs, Trento may train longer due to larger patches and more bands.

---

## 📁 Files Changed Summary

| File | Status | Notes |
|------|--------|-------|
| `main.py` | 🔄 **Different** | Data loading, PCA, split method |
| `train.py` | 🔄 **Different** | Function signature, pretraining, memory |
| `train_fine.py` | 🔄 **Different** | Memory optimizations, progress printing |
| `utils.py` | ✅ **Identical** | Same utilities |
| `net/sslsm.py` | ✅ **Identical** | Same architecture |
| `net/vit.py` | ✅ **Identical** | Same architecture |
| `ColorFunctions.py` | ✅ **Identical** | Same visualization |
