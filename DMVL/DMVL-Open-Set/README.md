# DMVL Open Set Recognition (OSR) Pipeline

This directory contains the complete, self-contained Open Set Recognition (OSR) extension for DMVL on cross-scene hyperspectral datasets without altering any original code in `DMVL-cross-scene/` or `DMVL-hyrank/`.

---

## 1. Directory Structure

```
DMVL-Open-Set/
├── README.md                          # Documentation
├── run_all.py                         # Runs OSR on both datasets
│
├── cross-scene/                       # Houston 2013 -> 2018
│   ├── model.py                       # DMVL ResNet-50 architecture
│   ├── utils.py                       # Patch & augmentation transforms
│   ├── kappa.py                       # Cohen's kappa calculation
│   ├── step1-generate-data-cross-scene.py  # 80/20 target split data gen
│   ├── step2-train-cross-scene.py     # Source pretraining + 128-dim feature extraction
│   ├── step3-test-cross-scene.py      # Closed-set few-shot SVM baseline
│   ├── step4-osr-cross-scene.py       # Open-set recognition evaluation (alias)
│   ├── run_osr.py                     # Main OSR script
│   └── run.py                         # Complete end-to-end runner (Steps 1–4)
│
└── hyrank/                            # Dioni -> Loukia
    ├── model.py                       # DMVL ResNet-50 architecture
    ├── utils.py                       # Patch & augmentation transforms
    ├── kappa.py                       # Cohen's kappa calculation
    ├── step1-generate-data-hyrank.py  # 80/20 target split data gen
    ├── step2-train-hyrank.py          # Source pretraining + 128-dim feature extraction
    ├── step3-test-hyrank.py           # Closed-set few-shot SVM baseline
    ├── step4-osr-hyrank.py            # Open-set recognition evaluation (alias)
    ├── run_osr.py                     # Main OSR script
    └── run.py                         # Complete end-to-end runner (Steps 1–4)
```

---

## 2. Methodology & Mapping from MorphJEPA

| Parameter / Concept | MorphJEPA Value | DMVL Implementation | Note |
|---|---|---|---|
| **OS_LAYER** | `'norm2'` | **Projection head (128-dim `out`)** | Exactly the representation saved to `.h5` and used in closed-set tables for 1:1 parity |
| **Pretraining** | Source domain | Houston-13 (2,530 patches) / Dioni (20,024 patches) | Unsupervised contrastive learning |
| **Evaluation domain** | Target domain | Houston-18 / Loukia (20% stratified test split) | 10,640 test samples for H18; 2,064 for Loukia |
| **Shots (`OS_SHOTS`)** | `(1, 5, 10)` | `(1, 5, 10)` | Few-shot samples per known class |
| **`N_UNKNOWN`** | `3` | `3` | Number of classes hidden per OSR split |
| **`MIN_TEST_PX`** | `50` | `50` | Classes with < 50 test samples cannot be unknown |
| **`ALWAYS_KNOWN`** | `[3]` | `[3]` (Houston) / `[0, 1, 3]` (HyRank) | Prevents low-frequency classes from being selected as unknown |
| **Classifier** | `RobustProbe` | 2-layer MLP (`Linear(128, 64) -> ReLU -> Dropout -> Linear(64, n_known)`) | Lightweight probe regularized for small few-shot sets |
| **Novelty Score** | MSP | `1.0 - max(softmax(logits))` | Higher score = more likely unknown class |
| **Metric** | Macro AUROC | Per-unknown-class ROC-AUC averaged across unknown classes | Evaluated over 5 seeds `[42, 123, 456, 789, 1024]` |

---

## 3. How to Run

### Option A: Run OSR Only (Assumes pre-extracted features exist)
```bash
# Houston Cross-Scene:
cd DMVL-Open-Set/cross-scene
python run_osr.py

# HyRank Cross-Scene:
cd DMVL-Open-Set/hyrank
python run_osr.py

# Or run both simultaneously:
cd DMVL-Open-Set
python run_all.py
```

### Option B: Run Full Pipeline End-to-End (Data Gen -> Pretrain -> Closed-set -> Open-set)
```bash
# Houston Cross-Scene:
cd DMVL-Open-Set/cross-scene
python run.py

# HyRank Cross-Scene:
cd DMVL-Open-Set/hyrank
python run.py
```

---

## 4. Output Files & Formats

- **Houston Results**:
  - Closed-set: `cross-scene/CrossScene/CrossScene_results.txt`
  - Open-set: `cross-scene/CrossScene/CrossScene_OSR_results.txt`
- **HyRank Results**:
  - Closed-set: `hyrank/HyRank/HyRank_results.txt`
  - Open-set: `hyrank/HyRank/HyRank_OSR_results.txt`

Console output format matching MorphJEPA:
```
[encoder] [zero-shot (H13 -> H18)]  1-shot class-balanced AUROC XX.XX ± X.XX (seeds), ± X.XX (splits) | known acc XX.XX
[encoder] [zero-shot (H13 -> H18)]  5-shot class-balanced AUROC XX.XX ± X.XX (seeds), ± X.XX (splits) | known acc XX.XX
[encoder] [zero-shot (H13 -> H18)] 10-shot class-balanced AUROC XX.XX ± X.XX (seeds), ± X.XX (splits) | known acc XX.XX
[encoder] [adaptation (H18 -> H18)]  1-shot class-balanced AUROC XX.XX ± X.XX (seeds), ± X.XX (splits) | known acc XX.XX
...
```
