# SSLSM Open Set Recognition (OSR) Benchmark Suite

This directory contains the complete Open Set Recognition (OSR) pipeline adapted for **SSLSM** (*Self-Supervised Learning with Spectral Masking for Hyperspectral Image Classification*).

The benchmark adheres strictly to the evaluation protocol established for **MorphJEPA**, **DMVL**, and **GGCL**, providing unified, rigorous cross-method comparability.

---

## 1. SSLSM Architecture & Representation Mapping

In SSLSM:
- **Input Transformation**: PCA dimensionality reduction with whitening (`pca_components = 15`), spatial patch size $15 \times 15$.
- **Backbone**: Vision Transformer encoder (`ViT`, depth = 8, heads = 8, dim = 225) + Transformer decoder (depth = 2, heads = 8, dim = 512).
- **Pre-training**: Masked Spectral Reconstruction with `masking_ratio = 0.5`. The network reconstructs masked spectral bands using Mean Squared Error ($MSE$).
- **Classification Head**:
  ```python
  self.fc = nn.Sequential(
      nn.Flatten(),
      nn.BatchNorm1d(bands * decoder_dim), # 15 * 512 = 7680
      nn.Dropout(0.5),
      nn.Linear(bands * decoder_dim, 1024),
      nn.ReLU(),
      nn.BatchNorm1d(1024),
      nn.Linear(1024, num_classes),
      nn.LogSoftmax(dim=1)
  )
  ```
- **Representation Mapping (`OS_LAYER`)**:
  With zero masking (`masking_ratio = 0.00001`) in `eval()` mode, the representation is extracted right before the final classification projection:
  $$\mathbf{z} = \text{fc}_{[:6]}(\text{decoded\_tokens}) \in \mathbb{R}^{1024}$$
  This provides a rich, 1024-dimensional normalized feature embedding fed into the downstream `RobustProbe`.

---

## 2. Benchmark Protocol & Datasets

### A. Houston Cross-Scene (Houston 2013 $\rightarrow$ Houston 2018)
- **PCA Components**: 15 (whitened)
- **Spatial Patch**: $15 \times 15$ with zero-padding
- **Total Classes**: 7 shared classes (0-indexed: `0–6`)
- **Test Pixel Filter**: Class 4 (`idx=3`) has only 4 test pixels $\rightarrow$ `ALWAYS_KNOWN = [3]`
- **Eligible Unknowns**: 6 classes $\rightarrow \binom{6}{3} = \mathbf{20}$ **splits**
- **Few-shot Support**: 1-shot, 5-shot, 10-shot
- **Random Seeds**: `[42, 123, 456, 789, 1024]` (5 seeds)

### B. HyRank Cross-Scene (Dioni $\rightarrow$ Loukia)
- **PCA Components**: 15 (whitened)
- **Spatial Patch**: $15 \times 15$ with zero-padding
- **Total Classes**: 12 shared classes (0-indexed: `0–11`)
- **Test Pixel Filter**: Classes 1, 2, 4 (`idx=0, 1, 3`) have $< 50$ test pixels ($41, 10, 15$) $\rightarrow$ `ALWAYS_KNOWN = [0, 1, 3]`
- **Eligible Unknowns**: 9 classes $\rightarrow \binom{9}{3} = \mathbf{84}$ **splits**
- **Few-shot Support**: 1-shot, 5-shot, 10-shot
- **Random Seeds**: `[42, 123, 456, 789, 1024]` (5 seeds)

---

## 3. Metrics Computed & Reported

1. **Class-Balanced AUROC**:
   Mean AUROC over the 3 unknown classes, distinguishing each unknown class from all known test pixels:
   $$\text{AUROC}_{\text{balanced}} = \frac{1}{|U|} \sum_{u \in U} \text{AUROC}(u \text{ vs. all } K)$$
2. **Known Accuracy (Multi-Class)**:
   Standard classification accuracy on test samples belonging to known classes:
   $$\text{Acc}_{\text{known}} = \frac{1}{|D_{\text{known}}| \sum_{x \in D_{\text{known}}} \mathbb{I}(\hat{y} = y)$$

Both metrics are reported and saved with means and standard deviations:
```
[encoder] [task] K-shot class-balanced AUROC XX.XX ± X.XX (seeds), ± X.XX (splits) | known acc XX.XX%
```

---

## 4. Directory Structure

```
SSLSM-Open-Set/
├── README.md
├── run_all.py                        # Master runner for both Houston & HyRank
├── SSLSM_Open_Set_Kaggle.ipynb       # Kaggle GPU ready-to-run notebook
├── cross-scene/
│   ├── net/                          # ViT and SSLSM architectures
│   ├── dataloader.py                 # Houston 13/18 PCA and patch loading
│   ├── kappa.py                      # NumPy 2.0+ compatible Cohen's Kappa & AA
│   ├── step1_extract_features.py     # Pre-trains & extracts 1024-dim features
│   ├── step2_closed_set.py           # Closed-set SVM evaluation (OA, AA, Kappa)
│   ├── run_osr.py                    # MSP RobustProbe OSR (AUROC + Known Acc)
│   └── run.py                        # Complete Houston cross-scene runner
└── hyrank/
    ├── net/                          # ViT and SSLSM architectures
    ├── dataloader.py                 # Dioni/Loukia PCA and patch loading
    ├── kappa.py                      # NumPy 2.0+ compatible Cohen's Kappa & AA
    ├── step1_extract_features.py     # Pre-trains & extracts 1024-dim features
    ├── step2_closed_set.py           # Closed-set SVM evaluation
    ├── run_osr.py                    # MSP RobustProbe OSR (AUROC + Known Acc)
    └── run.py                        # Complete HyRank cross-scene runner
```

---

## 5. How to Run

### Option A: Complete Pipeline
```bash
# Houston Cross-Scene
cd cross-scene
python run.py

# HyRank Cross-Scene
cd hyrank
python run.py
```

### Option B: Run OSR Only (when features exist)
```bash
# Houston
cd cross-scene
python run_osr.py

# HyRank
cd hyrank
python run_osr.py

# Both
python run_all.py
```

### Option C: Kaggle GPU Execution
1. Upload `SSLSM-Open-Set.zip` to Kaggle.
2. Open `SSLSM_Open_Set_Kaggle.ipynb`.
3. Enable GPU (Tesla P100 / T4) and click **Run All**.
