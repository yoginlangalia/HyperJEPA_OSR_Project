# SpectralDINO Open Set Recognition (OSR) Benchmark Suite

This directory contains the complete Open Set Recognition (OSR) benchmark suite adapted for **SpectralDINO** (*Dual Mixture-of-Subspaces Low-Rank Adaptation for Cross-Domain Hyperspectral Image Few-Shot Classification*).

The benchmark strictly mirrors the standardized experimental protocols established for **MorphJEPA**, **DMVL**, **GGCL**, and **SSLSM**, enabling fair and rigorous cross-method comparisons.

---

## 1. SpectralDINO Architecture & Representation Mapping

In SpectralDINO:
- **Input Spatial Patch**: $15 \times 15$ centered spatial patch.
- **Spectral Tokenizer**: Learnable $1 \times 1$ convolution projecting input spectral bands ($B$) to 256 channels ($B \times 256$ parameters), followed by adaptive spatial pooling to 16 spectral-spatial tokens.
- **Backbone**: Vision Transformer (ViT-Small configuration: 12 Transformer encoder blocks, $d=384$, 6 attention heads, 21,564,964 base parameters).
- **Dual Mixture-of-Subspaces (Dual-MoS) LoRA**: Multi-expert subspace low-rank adaptation (rank $r=4$) with softmax gating routing over dual subspace experts:
  - Subspace 1: Spectral-domain adaptation
  - Subspace 2: Spatial-contextual adaptation
- **Representation Mapping**:
  The downstream feature representation $\mathbf{z}$ is extracted from the L2-normalized [CLS] token output of the ViT backbone:
  $$\mathbf{z} = \frac{\text{CLS}}{\|\text{CLS}\|_2} \in \mathbb{R}^{384}$$
  This 384-dimensional normalized embedding is fed into the downstream `RobustProbe` and closed-set SVM classifiers.

---

## 2. Benchmark Protocol & Datasets

### A. Houston Cross-Scene (Houston 2013 $\rightarrow$ Houston 2018)
- **Spectral Bands**: 48 bands
- **Spatial Patch**: $15 \times 15$
- **Total Classes**: 7 shared classes (0-indexed: `0–6`)
- **Test Pixel Filter**: Class 4 (`idx=3`, Trees) has only 4 test pixels $\rightarrow$ `ALWAYS_KNOWN = [3]`
- **Eligible Unknowns**: 6 classes $\rightarrow \binom{6}{3} = \mathbf{20}$ **splits**
- **Few-shot Settings**: 1-shot, 5-shot, 10-shot
- **Random Seeds**: `[42, 123, 456, 789, 1024]` (5 seeds)

### B. HyRank Cross-Scene (Dioni $\rightarrow$ Loukia)
- **Spectral Bands**: 176 bands
- **Spatial Patch**: $15 \times 15$
- **Total Classes**: 12 shared classes (0-indexed: `0–11`)
- **Test Pixel Filter**: Classes with $< 50$ test pixels are kept as known
- **Eligible Unknowns**: 9 classes $\rightarrow \binom{9}{3} = \mathbf{84}$ **splits**
- **Few-shot Settings**: 1-shot, 5-shot, 10-shot
- **Random Seeds**: `[42, 123, 456, 789, 1024]` (5 seeds)

---

## 3. Directory Structure

```
SpectralDINO-Open-Set/
├── cross-scene/
│   ├── data/                          # Features & MAT files
│   ├── net/
│   │   ├── __init__.py
│   │   └── spectraldino.py            # SpectralDINO Dual-MoS LoRA ViT
│   ├── dataloader.py                  # Cross-scene data loading & normalization
│   ├── kappa.py                       # Cohen's Kappa & AA calculation
│   ├── step1_extract_features.py      # Pretraining & 384-dim feature extraction
│   ├── step2_closed_set.py            # Closed-set SVM baseline (Zero-shot & Adaptation)
│   ├── run_osr.py                     # MSP-based OSR (AUROC + Known Accuracy)
│   └── run.py                         # Complete pipeline runner
├── hyrank/
│   ├── data/
│   ├── net/
│   │   ├── __init__.py
│   │   └── spectraldino.py
│   ├── dataloader.py
│   ├── kappa.py
│   ├── step1_extract_features.py
│   ├── step2_closed_set.py
│   ├── run_osr.py
│   └── run.py
├── run_all.py                         # Runs both benchmarks sequentially
├── README.md                          # Documentation
└── SpectralDINO_Open_Set_Kaggle.ipynb  # Interactive Kaggle notebook
```

---

## 4. Execution Instructions

### Run Houston Cross-Scene:
```bash
cd SpectralDINO-Open-Set/cross-scene
python run.py
```

### Run HyRank Cross-Scene:
```bash
cd SpectralDINO-Open-Set/hyrank
python run.py
```

### Run All Benchmarks:
```bash
cd SpectralDINO-Open-Set
python run_all.py
```
