# 🔍 Deep Analysis: SSLSM-houston vs SSLSM-orignial

> **Generated on**: 2025-12-21  
> **Updated on**: 2025-12-22 (Pretraining now uses 100% of TR only - NO data leakage)  
> **Purpose**: Comprehensive comparison of changes made by Cursor when adapting SSLSM from Indian Pines to Houston dataset

---

## 🚨 LATEST UPDATE (2025-12-22)

The pretraining has been updated to use **100% of TR (training set only)** instead of 50% of TR+TE combined:

| Stage | Previous | Current |
|-------|----------|---------|
| **Pre-training** | 50% of (TR + TE) - Data leakage ❌ | 100% of TR only ✅ |
| **Fine-tuning** | N samples per class from TR | N samples per class from TR (unchanged) |
| **Testing** | 100% of TE | 100% of TE (unchanged) |

---

## 📁 Structural Differences

| Aspect | SSLSM-orignial | SSLSM-houston |
|--------|---------------|---------------|
| **Total Python Files** | 6 files | 6 files |
| **Data Files** | Indian_pines_corrected.mat + Indian_pines_gt.mat | houston.mat (with TR/TE splits) |
| **net/__init__.py** | ❌ Missing | ✅ Added |
| **requirements.txt** | ❌ Missing | ✅ Added |
| **Results Path** | `./results/indian/` | `./results/houston/` |
| **Model Paths** | `net_ip_train.pkl`, `net_ip_fine.pkl` | `net_houston_train.pkl`, `net_houston_fine.pkl` |

---

## 🔧 Hyperparameter Changes

| Parameter | Original (Indian Pines) | Houston | Reason |
|-----------|-------------------------|---------|--------|
| **EPOCH** | 100 | 50 | Reduced training time |
| **patch_size** | 17 | 15 | Smaller patches |
| **pca_components** | 30 | 15 | Houston has different spectral characteristics |
| **batch_size** | 128 | 32 | Memory optimization |
| **N_RUNS** | 5 | 1 | Faster experimentation |
| **ratio_fine** | 0.95 (used) | ~~Commented out~~ | Replaced with samples_per_class |

---

## 📝 File-by-File Detailed Comparison

### 1. `main.py` - Major Restructuring

#### **Original (Indian Pines)**
```python
# Data loading - single X, y arrays
X = sio.loadmat('./data/Indian_pines_corrected.mat')['indian_pines_corrected']
y = sio.loadmat('./data/Indian_pines_gt.mat')['indian_pines_gt']

# Single image cube creation
X, Y = createImageCubes(X_pca, y, windowSize=patch_size)

# Train call
accuracy = train(X, Y, ratio_self, ratio_fine, class_num, N_RUNS, ...)
```

#### **Houston Version**
```python
# Random seed for reproducibility
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# Data loading - predefined TR/TE splits
data = sio.loadmat('./data/houston.mat')
X = data['input']
TR = data['TR']  # Training labels
TE = data['TE']  # Test labels

# Directory creation
os.makedirs('./log', exist_ok=True)
os.makedirs('./results/houston', exist_ok=True)

# Three separate image cube sets
y_full = TR + TE  # Combined for pre-training (data leakage - intentional)
X_full, Y_full = createImageCubes(X_pca, y_full, windowSize=patch_size, removeZeroLabels=True)
X_TR, Y_TR = createImageCubes(X_pca, TR, windowSize=patch_size, removeZeroLabels=True)
X_TE, Y_TE = createImageCubes(X_pca, TE, windowSize=patch_size, removeZeroLabels=True)

# New train call with separate datasets
accuracy = train(X_full, Y_full, X_TR, Y_TR, X_TE, Y_TE, ratio_self, None, class_num, N_RUNS, ..., seed)
```

#### **Key Changes in main.py:**
- ✅ Added random seed (42) for reproducibility
- ✅ Changed from single X,y to X, TR, TE structure
- ✅ Added automatic directory creation
- ✅ Creates 3 separate image cube sets (full, TR, TE)
- ✅ Added `os` import
- ⚠️ Pre-training uses TR+TE combined (noted as intentional data leakage)

---

### 2. `train.py` - Significant Algorithm Changes

#### **Original Function Signature**
```python
def train(X, Y, test_ratio, ratio_fine, class_num, N_RUNS,
          patch_size, pca_components, batch_size, device, EPOCH, 
          path, fine_path, mask_ratio, encoder_depth, decoder_depth):
```

#### **Houston Function Signature**
```python
def train(X_full, Y_full, X_TR, Y_TR, X_TE, Y_TE, test_ratio, ratio_fine, 
          class_num, N_RUNS, patch_size, pca_components, batch_size, device, 
          EPOCH, path, fine_path, mask_ratio, encoder_depth, decoder_depth, seed):
```

#### **Original Metrics Storage**
```python
Kappa = []
CA = np.zeros([class_num, N_RUNS])
OA = []
AA = []
total_time = []
```

#### **Houston Metrics Storage**
```python
samples_per_class_list = [1, 5, 10]
metrics_per_config = {
    n_samples: {
        'Kappa': [], 'OA': [], 'AA': [], 'F1': [], 
        'CA': np.zeros([class_num, N_RUNS]), 'time': []
    } 
    for n_samples in samples_per_class_list
}
```

#### **Original Data Split**
```python
Xtrain, Xtest, ytrain, ytest = splitTrainTestSet(X, Y, test_ratio)
```

#### **Houston Data Split**
```python
# Pre-training: Use 50% of TR+TE combined
X_full_2d = X_full.reshape(num_samples, -1)
Xtrain_pretrain_2d, _, _, _ = splitTrainTestSet(X_full_2d, Y_full, test_ratio, randomState=seed)
Xtrain_pretrain = Xtrain_pretrain_2d.reshape(-1, patch_size, patch_size, pca_components)

# Dummy labels for self-supervised pre-training
Y_pretrain_dummy = np.zeros(len(Xtrain_pretrain))
```

#### **Original Fine-tuning Call**
```python
conf_mat, fine_time = train_fine(X, Y, ratio_fine, class_num, run, ...)
```

#### **Houston Fine-tuning Call (Loop through 3 configurations)**
```python
for samples_per_class in samples_per_class_list:  # [1, 5, 10]
    fine_path_samples = fine_path.replace('.pkl', '_{}samples.pkl'.format(samples_per_class))
    conf_mat, fine_time, y_true_test, y_pred_test = train_fine(
        X_TR, Y_TR, X_TE, Y_TE, samples_per_class, class_num, run, ..., seed)
    
    # Calculate F1 score (NEW)
    f1 = weighted_f1_score(y_true_test, y_pred_test)
```

#### **Key Changes in train.py:**
- ✅ New function signature with separate TR/TE datasets + seed
- ✅ Metrics stored per samples_per_class configuration (1, 5, 10)
- ✅ Added F1 score calculation
- ✅ Pre-training uses 50% of combined TR+TE
- ✅ Fine-tuning loops through 3 sample configurations
- ✅ Creates unique model paths per configuration
- ✅ Fixed number formatting ({:.4} → {:.4f})

---

### 3. `train_fine.py` - Complete Redesign

#### **Original Function Signature**
```python
def train_fine(X, Y, test_ratio, class_num, run,
               patch_size, pca_components, batch_size, device, 
               EPOCH, train_path, fine_path, encoder_depth, decoder_depth):
```

#### **Houston Function Signature**
```python
def train_fine(X_TR, Y_TR, X_TE, Y_TE, samples_per_class, class_num, run,
               patch_size, pca_components, batch_size, device, 
               EPOCH, train_path, fine_path, encoder_depth, decoder_depth, seed):
```

#### **Original Data Sampling**
```python
from utils import splitTrainTestSet, TrainDS, TestDS
# ...
Xtrain, Xtest, ytrain, ytest = splitTrainTestSet(X, Y, test_ratio)
```

#### **Houston Data Sampling**
```python
from utils import samplePerClass, TrainDS, TestDS
# ...
# Sample N samples per class from TR for training
Xtrain, _, ytrain, _ = samplePerClass(X_TR, Y_TR, samples_per_class, class_num, randomState=seed)
# Use TE directly for testing
Xtest = X_TE
ytest = Y_TE
```

#### **Original Return**
```python
return conf_mat, fine_time
```

#### **Houston Return**
```python
y_true_test = ytest
return conf_mat, fine_time, y_true_test, y_pred_test
```

#### **Key Changes in train_fine.py:**
- ✅ New function signature with separate TR/TE + samples_per_class + seed
- ✅ Uses `samplePerClass` instead of `splitTrainTestSet`
- ✅ Training uses N samples per class from TR
- ✅ Testing uses all of TE (not a random split)
- ✅ Returns y_true and y_pred for F1 calculation
- ✅ Updated print statement to show samples_per_class

---

### 4. `utils.py` - New Utility Functions Added

#### **New Import**
```python
# Original
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score

# Houston - Added f1_score
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score, f1_score
```

#### **New Function: samplePerClass**
```python
def samplePerClass(X, y, samples_per_class, class_num, randomState=345):
    """
    Sample N samples per class for training, rest goes to test set.
    Returns: X_train, X_test, y_train, y_test
    """
    np.random.seed(randomState)
    train_indices = []
    test_indices = []
    
    for class_id in range(class_num):
        class_indices = np.where(y == class_id)[0]
        
        if len(class_indices) == 0:
            continue
        
        # Randomly sample N samples for training
        n_samples = min(samples_per_class, len(class_indices))
        np.random.shuffle(class_indices)
        train_class_indices = class_indices[:n_samples]
        test_class_indices = class_indices[n_samples:]
        
        train_indices.extend(train_class_indices)
        test_indices.extend(test_class_indices)
    
    # ... shuffle and return splits
    return X_train, X_test, y_train, y_test
```

#### **New Function: weighted_f1_score**
```python
def weighted_f1_score(y_true, y_pred):
    return f1_score(y_true, y_pred, average='weighted')
```

#### **Key Changes in utils.py:**
- ✅ Added `f1_score` import
- ✅ Added `samplePerClass()` function for few-shot sampling
- ✅ Added `weighted_f1_score()` function

---

### 5. `net/sslsm.py` - **NO CHANGES** ✅

The neural network architecture is **identical** between both versions.

---

### 6. `net/vit.py` - **NO CHANGES** ✅

The Vision Transformer architecture is **identical** between both versions.

---

### 7. `ColorFunctions.py` - **NO CHANGES** ✅

The visualization utility is **identical** between both versions.

---

## 📊 Summary of All Changes

### **Dataset Adaptation**
| Aspect | Original | Houston |
|--------|----------|---------|
| Dataset | Indian Pines | Houston |
| Label Structure | Single `y` array | Separate `TR` and `TE` matrices |
| Train/Test Split | Random ratio-based | Predefined in dataset |

### **Fine-tuning Strategy**
| Aspect | Original | Houston |
|--------|----------|---------|
| Method | Ratio-based (95% test) | Per-class sampling |
| Configurations | Single (ratio_fine=0.95) | Three (1, 5, 10 samples/class) |
| Training Samples | ~5% of data | N × num_classes samples |

### **Reproducibility**
| Aspect | Original | Houston |
|--------|----------|---------|
| Random Seed | None | seed=42 |
| np.random.seed | ❌ | ✅ |
| torch.manual_seed | ❌ | ✅ |
| torch.cuda.manual_seed_all | ❌ | ✅ |

### **Metrics**
| Metric | Original | Houston |
|--------|----------|---------|
| OA (Overall Accuracy) | ✅ | ✅ |
| AA (Average Accuracy) | ✅ | ✅ |
| Kappa | ✅ | ✅ |
| F1 Score (Weighted) | ❌ | ✅ |
| Per-class Accuracy | ✅ | ✅ |

### **Code Quality**
| Aspect | Original | Houston |
|--------|----------|---------|
| requirements.txt | ❌ | ✅ |
| net/__init__.py | ❌ | ✅ |
| Directory creation | ❌ | ✅ (os.makedirs) |
| Number formatting | {:.4} | {:.4f} |

---

## ⚠️ Important Notes

1. **Data Leakage in Pre-training**: The Houston version intentionally combines TR+TE for pre-training (self-supervised phase). This is noted in comments as matching "original output" behavior.

2. **Network Architecture Unchanged**: The core SSLSM and ViT models are completely unchanged - only the training pipeline was modified.

3. **Few-Shot Learning Focus**: The Houston version is designed for few-shot classification experiments (1, 5, 10 samples per class).

---

## 📁 Files Changed Summary

| File | Status |
|------|--------|
| `main.py` | 🔄 **Major Changes** |
| `train.py` | 🔄 **Major Changes** |
| `train_fine.py` | 🔄 **Major Changes** |
| `utils.py` | 🔄 **New functions added** |
| `net/sslsm.py` | ✅ **Unchanged** |
| `net/vit.py` | ✅ **Unchanged** |
| `ColorFunctions.py` | ✅ **Unchanged** |
| `net/__init__.py` | ➕ **New file** |
| `requirements.txt` | ➕ **New file** |
