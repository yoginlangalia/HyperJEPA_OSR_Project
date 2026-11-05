# SpectralDINO Codebase & Architecture Analysis

## 1. Executive Summary

This document provides the comprehensive reverse-engineering and architecture specification of **SpectralDINO: Dual Mixture-of-Subspaces Low-Rank Adaptation for Cross-Domain Hyperspectral Image Few-Shot Classification** and details the construction of the **SpectralDINO Open Set Recognition (OSR) benchmark suite**.

---

## 2. Reverse-Engineered Architecture Specifications

Based on the execution traces and outputs from the Kaggle experiments across Houston Cross-Scene, HyRank, and Trento:

| Component | Specification | Details |
|---|---|---|
| **Base Architecture** | Vision Transformer (ViT-Small) | 12 Transformer encoder blocks, hidden dimension $d=384$, 6 attention heads |
| **Base Parameters** | 21,564,964 | Constant across all datasets |
| **Band Projection** | $B \times 256$ | Learnable $1 \times 1$ spectral projection layer |
| **Total Trainable Parameters** | $21,564,964 + 256 \times B$ | 21,577,252 (H13 $\rightarrow$ H18, 48 bands)<br>21,610,020 (HyRank, 176 bands)<br>21,581,092 (Trento, 63 bands) |
| **Input Spatial Patch** | $15 \times 15$ | Centered spatial patch around labeled pixel |
| **Token Representation** | 16 Spectral-Spatial Tokens | 4x4 spatial token grouping + [CLS] token |
| **Adaptation Mechanism** | Dual-MoS LoRA | Dual Mixture-of-Subspaces Low-Rank Adaptation (rank $r=4$) |
| **Subspace Routing** | Gating Network | Softmax routing across spectral and spatial subspace experts |
| **Self-Distillation** | DINO Head | Multi-layer projection head with sharpening and EMA teacher |
| **Downstream Feature** | 384-dimensional CLS | L2-normalized feature vector from transformer backbone |

---

## 3. Dataset Preprocessing & Splits

### Houston Cross-Scene
- **Source**: Houston 2013 (`Houston13.mat`, 48 bands, 7 classes, 2,530 labeled pixels)
- **Target**: Houston 2018 (`Houston18.mat`, 48 bands, 7 classes, 53,200 labeled pixels)
- **Normalization**: Z-score normalization per scene
- **Target Split**: 80% train (42,563 patches), 20% test (10,637 patches)

### HyRank Cross-Scene
- **Source**: Dioni (`Dioni.mat`, 176 bands, 12 classes, 20,024 labeled pixels, uint16)
- **Target**: Loukia (`Loukia.mat`, 176 bands, 12 classes, 10,317 labeled pixels, uint16)
- **Normalization**: Two-step: MinMax scaling to $[0, 1]$ followed by Z-score standardization
- **Target Split**: 80% train (8,259 patches), 20% test (2,058 patches)

---

## 4. OSR Pipeline Architecture

The `SpectralDINO-Open-Set` benchmark suite implements the exact 3-step protocol established for DMVL, GGCL, and SSLSM:

1. **Step 1: Feature Extraction (`step1_extract_features.py`)**
   - Initializes SpectralDINO with Dual-MoS LoRA backbone
   - Pre-trains on source domain via self-distillation / representation learning
   - Extracts 384-dimensional L2-normalized embeddings for Source, Target Train, and Target Test sets
   - Saves HDF5 files:
     - `Houston13_feature.h5` / `Dioni_feature.h5`
     - `Houston18_feature_train.h5` / `Loukia_feature_train.h5`
     - `Houston18_feature_test.h5` / `Loukia_feature_test.h5`

2. **Step 2: Closed-Set Few-Shot Baseline (`step2_closed_set.py`)**
   - Zero-shot Transfer: Source $\rightarrow$ Target Test (1, 5, 10 shots)
   - Few-shot Adaptation: Target Train $\rightarrow$ Target Test (1, 5, 10 shots)
   - Reports OA (Overall Accuracy), AA (Average Accuracy), and Cohen's Kappa

3. **Step 3: Open Set Recognition Evaluation (`run_osr.py`)**
   - MSP (Maximum Softmax Probability) novelty detection
   - 20 splits (Houston) / 84 splits (HyRank) with $N_{\text{unknown}}=3$
   - 5 random seeds ($42, 123, 456, 789, 1024$)
   - 2-layer MLP `RobustProbe` trained strictly on known classes
   - Reports Macro AUROC and Known Accuracy with mean and standard deviation
