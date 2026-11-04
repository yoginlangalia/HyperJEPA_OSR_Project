# SSLSM Codebase Deep Analysis

## Overview
This is a **Self-Supervised Learning with Spectral Masking (SSLSM)** framework for hyperspectral image classification using Vision Transformers (ViT). The codebase implements a two-stage training approach: self-supervised pre-training followed by fine-tuning.

---

## Architecture Overview

### Training Pipeline (Two-Stage Approach)

1. **Stage 1: Self-Supervised Pre-training** (`train.py`)
   - Trains the model using masked spectral reconstruction
   - Uses masking ratio (default 0.5) to mask spectral bands
   - Model learns to reconstruct masked spectral patches
   - No labels required during this stage
   - Runs for `EPOCH*2` iterations
   - Model saved to `train_path`

2. **Stage 2: Fine-tuning** (`train_fine.py`)
   - Loads pre-trained model from Stage 1
   - Uses very low masking ratio (0.00001) - essentially no masking
   - Fine-tunes on labeled data with CrossEntropyLoss
   - Runs for `EPOCH` iterations
   - Evaluates on test set and generates confusion matrix
   - Model saved to `fine_path`

---

## Key Components

### 1. **Main Entry Point** (`main.py`)

**Key Responsibilities:**
- Defines hyperparameters
- Loads dataset from .mat files
- Applies PCA dimensionality reduction
- Creates image patches (cubes)
- Calls training function
- Saves results

**Critical Parameters:**
```python
class_num = 16              # Number of classes (DATASET-SPECIFIC)
EPOCH = 100                 # Number of epochs
ratio_self = 0.5            # Train/test split for self-training
ratio_fine = 0.95           # Train/test split for fine-tuning
patch_size = 17             # Size of spatial patch (e.g., 17x17)
pca_components = 30         # PCA reduced dimensions
batch_size = 128
mask_ratio = 0.5            # Masking ratio for self-supervised learning
encoder_depth = 8           # ViT encoder depth
decoder_depth = 2           # Decoder depth
```

**Data Loading:**
```python
# Currently configured for Indian Pines dataset
X = sio.loadmat('./data/Indian_pines_corrected.mat')['indian_pines_corrected']
y = sio.loadmat('./data/Indian_pines_gt.mat')['indian_pines_gt']
```

**Data Format Expected:**
- `X`: Hyperspectral image cube `(H, W, B)` where:
  - H = height
  - W = width  
  - B = number of spectral bands
- `y`: Ground truth labels `(H, W)` with pixel-wise class labels

---

### 2. **Data Processing** (`utils.py`)

#### `applyPCA(X, numComponents)`
- Applies PCA to reduce spectral dimensionality
- Reshapes from (H, W, B) → (H*W, B) → applies PCA → (H, W, numComponents)
- Uses whitening for better results

#### `createImageCubes(X, y, windowSize)`
- Extracts spatial patches around each pixel
- Creates cubes of size `(windowSize, windowSize, numComponents)`
- Handles padding for edge pixels
- **Removes zero-labeled pixels** (background/unlabeled)
- **Shifts labels from 1-indexed to 0-indexed** (subtracts 1)
- Returns:
  - `patchesData`: (N, windowSize, windowSize, numComponents)
  - `patchesLabels`: (N,) with class indices 0 to class_num-1

#### `splitTrainTestSet(X, y, testRatio, randomState=345)`
- Uses stratified split to maintain class distribution
- Fixed random state for reproducibility

#### Dataset Classes:
- `TrainDS`: PyTorch Dataset for training
- `TestDS`: PyTorch Dataset for testing

---

### 3. **Model Architecture**

#### Vision Transformer (`net/vit.py`)

**ViT Class:**
- Takes hyperspectral patches as input
- Splits patches into smaller sub-patches (3x3)
- Uses positional encoding for spectral bands
- Implements standard Transformer encoder
- Outputs classification logits

**Key Parameters:**
```python
image_size=9           # Fixed: patches are split into 9x9
patch_size=3           # Fixed: each sub-patch is 3x3
num_classes            # Number of classes (from class_num)
dim=patch_size**2      # Dimension = 289 for patch_size=17
depth                  # Encoder depth (configurable)
bands                  # Number of PCA components (spectral bands)
```

**Architecture Details:**
- Patch embedding: `Rearrange('b c h w -> b (h w) c')` + Linear projection
- Adds CLS token and positional embeddings
- Multi-head self-attention + Feed-forward
- Final classification head

#### SSLSM Model (`net/sslsm.py`)

**Self-Supervised Learning with Spectral Masking:**

**Key Components:**
1. **Encoder**: ViT encoder (shared)
2. **Masking Strategy**: Randomly masks spectral bands (not spatial patches)
3. **Decoder**: Transformer decoder to reconstruct masked bands
4. **Classification Head**: Fully connected layers for classification

**Forward Pass:**
1. Extracts spectral patches from input
2. Randomly masks `masking_ratio * bands` spectral bands
3. Encodes unmasked bands using encoder
4. Decodes to reconstruct masked bands
5. Computes reconstruction loss (MSE)
6. Also outputs classification logits from decoder tokens

**Important Notes:**
- Uses **spectral masking** (bands), not spatial masking
- `masking_ratio=0.5` during pre-training
- `masking_ratio=0.00001` during fine-tuning (essentially no masking)
- Classification uses flattened decoder tokens → BatchNorm → FC layers

---

### 4. **Training Functions**

#### `train()` in `train.py`

**Self-Supervised Pre-training Stage:**

1. **Model Initialization:**
   - Creates ViT encoder
   - Wraps in SSLSM model
   - Uses `mask_ratio` for masking

2. **Data Preparation:**
   - Splits data using `ratio_self` (50% test split)
   - Reshapes to (N, patch_size, patch_size, pca_components)
   - Transposes to (N, pca_components, patch_size, patch_size)

3. **Training Loop:**
   - Runs for `EPOCH*2` iterations
   - Only uses reconstruction loss (no labels)
   - Optimizes with Adam (lr=3e-4)
   - Saves model to `train_path`

4. **Fine-tuning Call:**
   - Calls `train_fine()` with full dataset and `ratio_fine`
   - Collects metrics from fine-tuning stage

5. **Metrics Collection:**
   - Per-class accuracy (CA)
   - Overall accuracy (OA)
   - Average accuracy (AA)
   - Kappa coefficient
   - Training time
   - Runs `N_RUNS` times for averaging

#### `train_fine()` in `train_fine.py`

**Supervised Fine-tuning Stage:**

1. **Model Initialization:**
   - Creates same architecture
   - **Loads pre-trained weights** from `train_path`
   - Uses very low masking ratio (0.00001)

2. **Data Preparation:**
   - Splits using `ratio_fine` (95% test split - different from Stage 1!)
   - Same reshaping/transposing as Stage 1

3. **Training Loop:**
   - Runs for `EPOCH` iterations
   - Uses CrossEntropyLoss with labels
   - Optimizes with Adam (lr=3e-4)

4. **Evaluation:**
   - Evaluates on test set
   - Generates confusion matrix
   - Returns confusion matrix and training time

---

## Data Flow

```
1. Load .mat files
   X: (H, W, B) - hyperspectral image
   y: (H, W) - ground truth labels

2. Apply PCA
   X_pca: (H, W, pca_components)

3. Create Image Cubes
   X: (N, patch_size, patch_size, pca_components)
   Y: (N,) - class labels (0-indexed)

4. Reshape for Training
   X: (N, pca_components, patch_size, patch_size)
   [This is the format expected by the model]

5. Stage 1: Self-Supervised Pre-training
   - Split with ratio_self (50% test)
   - Train with masked spectral reconstruction
   - Save model

6. Stage 2: Fine-tuning
   - Split with ratio_fine (95% test) - uses FULL dataset again!
   - Load pre-trained model
   - Fine-tune with labels
   - Evaluate and compute metrics
```

---

## What Needs to be Changed for Different Datasets

### 1. **Dataset-Specific Changes in `main.py`:**

#### a) Dataset Loading (Lines 25-32)
```python
# Current (Indian Pines):
X = sio.loadmat('./data/Indian_pines_corrected.mat')['indian_pines_corrected']
y = sio.loadmat('./data/Indian_pines_gt.mat')['indian_pines_gt']

# For other datasets, change:
# - File paths
# - Variable names in .mat files (e.g., 'paviaU', 'Data_gt')
```

#### b) Number of Classes (Line 8)
```python
class_num = 16  # MUST CHANGE based on your dataset's class count
```

#### c) Model Save Paths (Lines 34-37)
```python
train_path = r'./log/net_ip_train.pkl'      # Change 'ip' to dataset identifier
fine_path = r'./log/net_ip_fine.pkl'        # Change 'ip' to dataset identifier
accuracy_path = r'./results/indian/'        # Change to dataset-specific folder
```

### 2. **Potential Data Format Issues:**

#### a) Label Indexing
- Code assumes labels start at 1 in ground truth
- Automatically converts to 0-indexed (subtracts 1)
- Zero labels are filtered out (background/unlabeled pixels)
- **If your dataset uses 0-indexed labels, you may need to adjust**

#### b) Data Shape Validation
- Ensure X.shape = (H, W, B) - height, width, spectral bands
- Ensure y.shape = (H, W) - same spatial dimensions
- Ensure labels are integers from 1 to class_num (or 0 to class_num-1)

#### c) Spectral Bands
- Original bands will be reduced via PCA
- `pca_components = 30` reduces to 30 spectral dimensions
- This should work for most hyperspectral datasets
- **May need to adjust if dataset has very few bands (<30)**

### 3. **Hyperparameter Adjustments (May Need Tuning):**

#### a) Patch Size (Line 12)
```python
patch_size = 17  # Current setting
# Larger patches = more spatial context, more computation
# Smaller patches = less context, faster training
# Should be odd number (for proper padding)
```

#### b) PCA Components (Line 14)
```python
pca_components = 30  # Current setting
# Should be <= original number of bands
# More components = more information, more computation
```

#### c) Train/Test Ratios (Lines 10-11)
```python
ratio_self = 0.5   # For self-supervised pre-training
ratio_fine = 0.95  # For fine-tuning (note: 95% for TEST, 5% for TRAIN!)
```

**IMPORTANT:** `ratio_fine = 0.95` means 95% test, 5% train for fine-tuning!
This seems unusual - verify if this is correct for your use case.

#### d) Model Architecture (Lines 17-18)
```python
encoder_depth = 8   # ViT encoder layers
decoder_depth = 2   # Decoder layers
```

### 4. **Network Architecture Hard-coded Values:**

#### In `net/vit.py` and `net/sslsm.py`:
- `image_size=9` and `patch_size=3` are **hard-coded** in training functions
- These should work for any input patch_size (they refer to internal patch splitting)
- `channels=30` default in ViT (but `bands` parameter overrides this)

#### In `net/sslsm.py`:
- Line 103: `nn.Linear(512, 81)` - hard-coded dimension
- Line 47: `nn.BatchNorm1d(bands*decoder_dim)` - depends on bands
- Line 49: `nn.Linear(bands*decoder_dim, 1024)` - depends on bands

**These should adapt automatically to `bands=pca_components`**

### 5. **Results and Logging:**

- Results saved to CSV files in `accuracy_path`
- Creates confusion matrices
- Computes OA, AA, Kappa, per-class accuracy
- Multiple runs (`N_RUNS=5`) for averaging

---

## Summary of Critical Changes for New Datasets

### MUST CHANGE:
1. ✅ `class_num` in `main.py` (line 8)
2. ✅ Dataset file paths and variable names in `main.py` (lines 25-26)
3. ✅ Model save paths in `main.py` (lines 34-37)

### SHOULD REVIEW/ADJUST:
4. ⚠️ `patch_size` - depends on spatial resolution needs
5. ⚠️ `pca_components` - should be <= original bands
6. ⚠️ `ratio_fine` - verify if 95% test is intended
7. ⚠️ `ratio_self` - adjust train/test split for pre-training
8. ⚠️ Label indexing - verify if labels are 0-indexed or 1-indexed
9. ⚠️ Data directory structure - ensure `./data/` and `./log/` and `./results/` exist

### LIKELY NO CHANGES NEEDED:
- Network architecture (adapts to `bands` and `class_num`)
- Training loop structure
- Evaluation metrics
- Utility functions (should work for standard hyperspectral data formats)

---

## Potential Issues to Watch For

1. **Data Shape Mismatches:**
   - Ensure .mat files contain arrays with expected dimensions
   - Verify spectral dimension is last axis

2. **Label Range:**
   - Code filters out label=0 and subtracts 1 from all labels
   - If your dataset uses 0-indexed labels starting from 0, you may need adjustment

3. **Class Count Mismatch:**
   - If `class_num` doesn't match actual classes in data, confusion matrix will be wrong

4. **Memory Issues:**
   - Large datasets may require reducing `batch_size` or `patch_size`
   - Multiple runs can be memory intensive

5. **Directory Structure:**
   - Ensure `./log/`, `./results/`, and `./data/` directories exist
   - Create subdirectories if needed (e.g., `./results/indian/`)

6. **Dependencies:**
   - Requires: torch, numpy, scipy, sklearn, einops
   - Ensure all dependencies are installed

---

## Code Execution Flow

```
main.py
  ↓
1. Load data (.mat files)
  ↓
2. Apply PCA
  ↓
3. Create image cubes
  ↓
4. Call train()
  ↓
  ┌─────────────────────────┐
  │  Stage 1: Pre-training  │
  │  - Split with ratio_self│
  │  - Train SSLSM (masking)│
  │  - Save model           │
  └─────────────────────────┘
  ↓
  ┌─────────────────────────┐
  │  Stage 2: Fine-tuning   │
  │  - Load pre-trained     │
  │  - Split with ratio_fine│
  │  - Fine-tune with labels│
  │  - Evaluate             │
  └─────────────────────────┘
  ↓
5. Save results to CSV
```

---

This analysis provides a complete understanding of the codebase structure, data flow, and what needs to be modified for different datasets.

