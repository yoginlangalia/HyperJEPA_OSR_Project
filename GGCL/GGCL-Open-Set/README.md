# GGCL Open Set Recognition (OSR) Benchmark Suite

This directory contains the complete Open Set Recognition (OSR) pipeline adapted for **GGCL** (*Guided Group Contrastive Learning for Hyperspectral Image Classification*, Li et al., IEEE TGRS 2024).

The benchmark mirrors the exact evaluation protocol established for **MorphJEPA** and **DMVL**, enabling direct, rigorously fair cross-method comparison.

---

## 1. GGCL Adaptation Architecture

In GGCL, the backbone model is **`HSINet`**:
- **Patch Embedding**: Spatial-spectral 2D Conv block (spectral channels $\rightarrow$ 64 dimensions)
- **Transformer Encoder**: 1 Transformer layer, 8 attention heads, hidden dim = 64
- **Readout / Representation**: The readout passed into the linear classification head (`self.mlp_head`) is **`logit_pixel`** (`x_pixel[:, 0]`), which is the **64-dimensional ViT `[CLS]` token embedding** produced by `self.encoder_block(x)`.
- **Contrastive Learning**: InfoNCE loss using odd-even spectral masking views (`augment_odd_even`).

### Representation Mapping (`OS_LAYER`):
For downstream Open Set Recognition:
$$\mathbf{z} = \text{encoder\_block}(x)[0] \in \mathbb{R}^{64}$$
The 64-dimensional embedding is extracted from the pretrained GGCL encoder and used as input to the `RobustProbe`.

---

## 2. Benchmark Protocol & Datasets

### A. Houston Cross-Scene (Houston 2013 $\rightarrow$ Houston 2018)
- **Spectral Bands**: 48 bands (no PCA)
- **Normalization**: Z-score (`(x - mean) / std`)
- **Spatial Patch**: $15 \times 15$ with zero-padding
- **Total Classes**: 7 shared classes (0-indexed: `0–6`)
- **Test Pixel Filter**: Class 4 (`idx=3`) has only 4 test pixels $\rightarrow$ `ALWAYS_KNOWN = [3]`
- **Eligible Unknowns**: 6 classes $\rightarrow \binom{6}{3} = \mathbf{20}$ **splits**
- **Few-shot Support**: 1-shot, 5-shot, 10-shot
- **Random Seeds**: `[42, 123, 456, 789, 1024]` (5 seeds)

### B. HyRank Cross-Scene (Dioni $\rightarrow$ Loukia)
- **Spectral Bands**: 176 bands (no PCA)
- **Normalization**: Z-score (`(x - mean) / std`)
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

Both metrics are saved and displayed with means and standard deviations across seeds and splits:
```
[encoder] [task] K-shot class-balanced AUROC XX.XX ± X.XX (seeds), ± X.XX (splits) | known acc XX.XX%
```

---

## 4. Directory Structure

```
GGCL-Open-Set/
├── README.md
├── run_all.py                        # Runs complete OSR suite for both datasets
├── GGCL_Open_Set_Kaggle.ipynb        # Ready-to-run Kaggle GPU notebook
├── cross-scene/
│   ├── models/                       # GGCL HSINet architecture
│   ├── augment.py                    # Odd-even spectral augmentation
│   ├── dataloader.py                 # Cross-scene data loading
│   ├── trainer.py                    # GGCL contrastive trainer
│   ├── utils.py, kappa.py            # NumPy 2.0+ compatible utilities
│   ├── step1_extract_features.py     # Pretrains & extracts 64-dim H13/H18 features
│   ├── step2_closed_set.py           # Closed-set SVM evaluation
│   ├── run_osr.py                    # MSP RobustProbe OSR evaluation
│   └── run.py                        # Complete Houston pipeline runner
└── hyrank/
    ├── models/
    ├── augment.py
    ├── dataloader.py
    ├── trainer.py
    ├── utils.py, kappa.py
    ├── step1_extract_features.py     # Pretrains & extracts 64-dim Dioni/Loukia features
    ├── step2_closed_set.py
    ├── run_osr.py                    # MSP RobustProbe OSR evaluation
    └── run.py                        # Complete HyRank pipeline runner
```

---

## 5. How to Run

### Option A: Complete Pipeline (from scratch or cached)
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
1. Upload `GGCL-Open-Set.zip` to Kaggle.
2. Open `GGCL_Open_Set_Kaggle.ipynb`.
3. Enable GPU (Tesla P100 / T4) and click **Run All**.
