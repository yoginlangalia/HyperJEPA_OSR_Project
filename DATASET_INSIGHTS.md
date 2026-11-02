# Dataset Analysis Insights

## Overview
Analysis of hyperspectral remote sensing datasets: **Trento**, **Houston**, and **Cross-Scene (Houston 2013/2018)**.

---

## TRENTO DATASET

### Key Characteristics
- **Spatial Size**: 166 × 600 pixels (99,600 total pixels)
- **Spectral Bands**: 63 bands
- **Classes**: 7 classes (0-6, where 0 is background)
- **Data Size**: ~24 MB for HSI data

### Data Components

#### 1. HSI_data (Hyperspectral Image)
- **Shape**: (166, 600, 63)
- **Data Type**: Stored as int32, but represents float32 values (range: 6.00 to 6427.00)
- **Mean**: 1114.85
- **Std**: 777.50
- **Interpretation**: Hyperspectral image with 63 spectral bands

#### 2. LiDAR_data
- **Shape**: (166, 600)
- **Data Type**: uint8 (0-255)
- **Mean**: 46.13
- **Non-zero pixels**: 86.3% of the scene
- **Interpretation**: Elevation/intensity data from LiDAR sensor

#### 3. Ground Truth Labels
- **Shape**: (166, 600)
- **Classes**: 7 classes (0-6)
- **Class Distribution**:
  - Class 0 (Background): 69.66% (69,386 pixels)
  - Class 1: 4.05% (4,034 pixels)
  - Class 2: 2.91% (2,903 pixels)
  - Class 3: 0.48% (479 pixels)
  - Class 4: 9.16% (9,123 pixels)
  - Class 5: 10.54% (10,501 pixels)
  - Class 6: 3.19% (3,174 pixels)

### Key Insights
1. **Multi-modal Dataset**: Contains both hyperspectral and LiDAR data, enabling multi-modal fusion approaches
2. **Smaller Dataset**: More manageable size for algorithm development and testing
3. **Class Imbalance**: Significant class imbalance with background (class 0) dominating
4. **No Pre-defined Split**: Training/test split needs to be created manually

---

## HOUSTON DATASET

### Key Characteristics
- **Spatial Size**: 349 × 1905 pixels (664,845 total pixels)
- **Spectral Bands**: 144 bands
- **Classes**: 16 classes (0-15, where 0 is background)
- **Data Size**: ~183 MB for HSI data

### Data Components

#### 1. input (Hyperspectral Image)
- **Shape**: (349, 1905, 144)
- **Data Type**: uint16 (0-65535)
- **Mean**: 6039.75
- **Std**: 4213.70
- **Interpretation**: Hyperspectral image with 144 spectral bands (2.3× more bands than Trento)

#### 2. TR (Training Labels)
- **Shape**: (349, 1905)
- **Classes**: 16 classes
- **Training Pixels**: 2,832 (18.84% of labeled data)
- **Distribution**: ~190 pixels per class (highly balanced training set)
- **Background**: 99.57% unlabeled

#### 3. TE (Test Labels)
- **Shape**: (349, 1905)
- **Classes**: 16 classes
- **Test Pixels**: 12,197 (81.16% of labeled data)
- **Distribution**: More pixels per class in test set (~500-1100 pixels per class)
- **Background**: 98.17% unlabeled

### Train/Test Split Statistics
- **Total Labeled Pixels**: 15,029 (2.26% of total scene)
- **Training Set**: 2,832 pixels (18.84%)
- **Test Set**: 12,197 pixels (81.16%)
- **Unlabeled Pixels**: 649,816 (97.74% of scene)

### Key Insights
1. **Standard Benchmark**: Widely used hyperspectral classification benchmark dataset
2. **High Spectral Resolution**: 144 bands provide rich spectral information
3. **Large Scene**: 6.7× larger than Trento dataset
4. **Pre-defined Split**: Comes with standard training/test split
5. **Highly Imbalanced**: Less than 3% of pixels are labeled (semi-supervised learning scenario)
6. **Single Modal**: Only HSI data (no additional modalities)

---

## COMPARATIVE ANALYSIS

### Size Comparison
| Aspect | Trento | Houston | Ratio |
|--------|--------|---------|-------|
| Spatial Size | 99,600 pixels | 664,845 pixels | 6.7× |
| Spectral Bands | 63 | 144 | 2.3× |
| Number of Classes | 7 | 16 | 2.3× |
| Data Size | ~24 MB | ~183 MB | 7.6× |

### Spectral Resolution
- **Trento**: 63 bands - moderate spectral resolution
- **Houston**: 144 bands - high spectral resolution, better for detailed spectral analysis

### Spatial Resolution
- **Trento**: Smaller scene, easier to process, good for quick experiments
- **Houston**: Much larger scene, more representative of real-world applications

### Classification Complexity
- **Trento**: 7 classes - simpler classification task
- **Houston**: 16 classes - more complex, fine-grained classification

### Multi-modal Capabilities
- **Trento**: ✅ Multi-modal (HSI + LiDAR) - unique advantage for fusion methods
- **Houston**: ❌ Single modal (HSI only)

### Training Data
- **Trento**: No pre-defined split - flexible for custom experimental setups
- **Houston**: Pre-defined split - standard benchmark evaluation

---

## RECOMMENDED USE CASES

### Trento Dataset
- **Multi-modal Fusion Research**: Testing HSI + LiDAR fusion algorithms
- **Algorithm Development**: Smaller size makes it ideal for prototyping
- **Memory-Constrained Applications**: Lower computational requirements
- **Multi-modal Feature Learning**: Exploring complementary information from different sensors

### Houston Dataset
- **Hyperspectral Classification Benchmark**: Standard evaluation dataset
- **High Spectral Resolution Analysis**: Testing spectral feature extraction methods
- **Large-Scale Scene Understanding**: Real-world application scenarios
- **Semi-Supervised Learning**: Large amount of unlabeled data (97.74%)
- **Deep Learning Research**: Sufficient data for training complex models

---

## TECHNICAL NOTES

### Data Format Issues Resolved
- **Trento HSI_data**: Initially appeared as very large integers (int32), but actually represents float32 values. Corrected by viewing as float32, revealing reasonable hyperspectral reflectance values (6-6427 range).

---

## CROSS-SCENE DATASET (Houston 2013 / Houston 2018)

### Overview
The Cross-Scene dataset consists of **two temporally different hyperspectral images** of the same Houston area captured in 2013 and 2018. This dataset is designed for **domain adaptation** and **transfer learning** research in hyperspectral image classification.

### Dataset Files
Located in: `Dataset/cross-scene/`
- `Houston13.mat` - Hyperspectral image from 2013 (52.88 MB)
- `Houston13_7gt.mat` - Ground truth labels for 2013 (0.01 MB)
- `Houston18.mat` - Hyperspectral image from 2018 (45.02 MB)
- `Houston18_7gt.mat` - Ground truth labels for 2018 (0.05 MB)

### Data Loading Notes
> [!IMPORTANT]
> These files are **MATLAB v7.3 (HDF5) format**. Use `h5py` library instead of `scipy.io.loadmat()`.
> Arrays must be **transposed** after loading to convert from MATLAB to Python convention.

```python
import h5py
import numpy as np

with h5py.File('Houston13.mat', 'r') as f:
    hsi_data = np.transpose(f['ori_data'][()])  # Key: 'ori_data'

with h5py.File('Houston13_7gt.mat', 'r') as f:
    gt_labels = np.transpose(f['map'][()])  # Key: 'map'
```

---

### Houston 2013 (Source Domain)

#### HSI Data
- **Key**: `'ori_data'`
- **Shape**: (210, 954, 48)
- **Spatial Size**: 210 × 954 = 200,340 pixels
- **Spectral Bands**: 48
- **Data Type**: float64
- **Value Range**: [0.0, 1.0] (pre-normalized)
- **Mean**: 0.0872, **Std**: 0.0573

#### Ground Truth
- **Key**: `'map'`
- **Shape**: (210, 954)
- **Classes**: 8 (0 = background, 1-7 = land cover classes)
- **Total Labeled Pixels**: 2,530 (1.26% of scene)

| Class | Pixels | Percentage |
|-------|--------|------------|
| 0 (Background) | 197,810 | 98.74% |
| 1 | 345 | 0.17% |
| 2 | 365 | 0.18% |
| 3 | 365 | 0.18% |
| 4 | 285 | 0.14% |
| 5 | 319 | 0.16% |
| 6 | 408 | 0.20% |
| 7 | 443 | 0.22% |

---

### Houston 2018 (Target Domain)

#### HSI Data
- **Key**: `'ori_data'`
- **Shape**: (210, 954, 48)
- **Spatial Size**: 210 × 954 = 200,340 pixels
- **Spectral Bands**: 48
- **Data Type**: float64
- **Value Range**: [0.0, 1.0] (pre-normalized)
- **Mean**: 0.1273, **Std**: 0.0854

#### Ground Truth
- **Key**: `'map'`
- **Shape**: (210, 954)
- **Classes**: 8 (0 = background, 1-7 = land cover classes)
- **Total Labeled Pixels**: 53,200 (26.55% of scene)

| Class | Pixels | Percentage |
|-------|--------|------------|
| 0 (Background) | 147,140 | 73.45% |
| 1 | 1,353 | 0.68% |
| 2 | 4,888 | 2.44% |
| 3 | 2,766 | 1.38% |
| 4 | 22 | 0.01% |
| 5 | 5,347 | 2.67% |
| 6 | 32,459 | 16.20% |
| 7 | 6,365 | 3.18% |

---

### Key Characteristics

| Aspect | Houston 2013 | Houston 2018 |
|--------|--------------|--------------|
| Spatial Size | 210 × 954 | 210 × 954 |
| Total Pixels | 200,340 | 200,340 |
| Spectral Bands | 48 | 48 |
| Labeled Pixels | 2,530 (1.26%) | 53,200 (26.55%) |
| Value Range | [0, 1] | [0, 1] |
| Mean Value | 0.0872 | 0.1273 |

### Domain Shift Observations
1. **Temporal Domain Shift**: Same geographic area captured 5 years apart exhibits different spectral characteristics (different mean: 0.087 vs 0.127)
2. **Class Imbalance Difference**: H18 has severe imbalance with Class 6 (16.2%) and very few Class 4 samples (22 pixels)
3. **Label Density**: H13 is sparsely labeled (1.26%), H18 is moderately labeled (26.55%)
4. **Same Spectral Alignment**: Both datasets have identical 48 bands, enabling direct comparison

### Per-Band Statistics Sample

**Houston 2013:**
| Band | Min | Max | Mean | Std |
|------|-----|-----|------|-----|
| 0 | 0.00 | 0.18 | 0.01 | 0.00 |
| 1 | 0.00 | 0.57 | 0.02 | 0.01 |
| 47 | 0.00 | 0.98 | 0.10 | 0.04 |

**Houston 2018:**
| Band | Min | Max | Mean | Std |
|------|-----|-----|------|-----|
| 0 | 0.01 | 0.09 | 0.02 | 0.00 |
| 1 | 0.01 | 0.20 | 0.03 | 0.01 |
| 47 | 0.00 | 0.44 | 0.13 | 0.06 |

---

### Implementation Guidelines

#### Typical Experimental Setup
- **Source Domain**: Houston 2013 (training)
- **Target Domain**: Houston 2018 (testing)
- **Alternative**: Reverse transfer (H18 → H13)

#### Key Configuration Parameters
```python
# Dataset parameters
NUM_CLASSES = 7  # Classes 1-7 (exclude background 0)
NUM_BANDS = 48
SPATIAL_SIZE_H = 210
SPATIAL_SIZE_W = 954

# Data is already normalized to [0, 1]
# Additional normalization may not be needed
```

#### Critical Considerations
> [!WARNING]
> **Class 4 in H18 has only 22 labeled pixels** - models may struggle with this class during evaluation.

> [!NOTE]
> Data is **pre-normalized** to [0, 1] range. Check if your method expects raw reflectance values or normalized values.

#### Recommended Use Cases
- **Domain Adaptation**: Transfer learning from 2013 to 2018 or vice versa
- **Semi-Supervised Learning**: Sparse labels in H13 (1.26%)
- **Temporal Change Analysis**: Same scene at different times
- **Cross-Scene Classification**: Testing model generalization

---

## HYRANK CROSS-SCENE DATASET (Dioni / Loukia)

### Overview
The HyRank Cross-Scene dataset consists of **two geographically different hyperspectral images** from the HyRANK benchmark: Dioni and Loukia. This dataset is designed for **cross-scene classification** and **domain adaptation** research, similar to the Houston 2013/2018 dataset but with different scenes rather than temporal differences.

### Dataset Files
Located in: `Dataset/HyRank-cross-scene/`
- `Dioni.mat` - Hyperspectral image of Dioni area (85.28 MB)
- `Dioni_gt_out68.mat` - Ground truth labels for Dioni (0.01 MB)
- `Loukia.mat` - Hyperspectral image of Loukia area (58.23 MB)
- `Loukia_gt_out68.mat` - Ground truth labels for Loukia (0.01 MB)

### Data Loading Notes
> [!TIP]
> These files are **MATLAB v5 format**. Use `scipy.io.loadmat()` - **EASIER** than Houston Cross-Scene.
> **No transpose needed** - data is already in Python-compatible order.

```python
import scipy.io as sio
import numpy as np

# Load Dioni
dioni_hsi = sio.loadmat('Dioni.mat')['ori_data']  # Key: 'ori_data'
dioni_gt = sio.loadmat('Dioni_gt_out68.mat')['map']  # Key: 'map'

# Load Loukia  
loukia_hsi = sio.loadmat('Loukia.mat')['ori_data']
loukia_gt = sio.loadmat('Loukia_gt_out68.mat')['map']
```

---

### Dioni (Source/Target Domain)

#### HSI Data
- **Key**: `'ori_data'`
- **Shape**: (250, 1376, 176)
- **Spatial Size**: 250 × 1376 = 344,000 pixels
- **Spectral Bands**: 176
- **Data Type**: uint16
- **Value Range**: [0, 13069] (raw reflectance, NOT normalized)
- **Mean**: 1362.93, **Std**: 1290.89

#### Ground Truth
- **Key**: `'map'`
- **Shape**: (250, 1376)
- **Classes**: 13 (0 = background, 1-12 = land cover classes)
- **Total Labeled Pixels**: 20,024 (5.82% of scene)

| Class | Pixels | Percentage |
|-------|--------|------------|
| 0 (Background) | 323,976 | 94.18% |
| 1 | 1,262 | 0.37% |
| 2 | 204 | 0.06% |
| 3 | 614 | 0.18% |
| 4 | 150 | 0.04% |
| 5 | 1,768 | 0.51% |
| 6 | 361 | 0.10% |
| 7 | 5,035 | 1.46% |
| 8 | 6,374 | 1.85% |
| 9 | 1,754 | 0.51% |
| 10 | 492 | 0.14% |
| 11 | 1,612 | 0.47% |
| 12 | 398 | 0.12% |

---

### Loukia (Source/Target Domain)

#### HSI Data
- **Key**: `'ori_data'`
- **Shape**: (249, 945, 176)
- **Spatial Size**: 249 × 945 = 235,305 pixels
- **Spectral Bands**: 176
- **Data Type**: uint16
- **Value Range**: [0, 11590] (raw reflectance, NOT normalized)
- **Mean**: 1348.50, **Std**: 1372.67

#### Ground Truth
- **Key**: `'map'`
- **Shape**: (249, 945)
- **Classes**: 13 (0 = background, 1-12 = land cover classes)
- **Total Labeled Pixels**: 10,317 (4.38% of scene)

| Class | Pixels | Percentage |
|-------|--------|------------|
| 0 (Background) | 224,988 | 95.62% |
| 1 | 206 | 0.09% |
| 2 | 54 | 0.02% |
| 3 | 426 | 0.18% |
| 4 | 79 | 0.03% |
| 5 | 1,107 | 0.47% |
| 6 | 422 | 0.18% |
| 7 | 2,996 | 1.27% |
| 8 | 2,361 | 1.00% |
| 9 | 399 | 0.17% |
| 10 | 453 | 0.19% |
| 11 | 1,393 | 0.59% |
| 12 | 421 | 0.18% |

---

### Key Characteristics

| Aspect | Dioni | Loukia |
|--------|-------|--------|
| Spatial Size | 250 × 1376 | 249 × 945 |
| Total Pixels | 344,000 | 235,305 |
| Spectral Bands | 176 | 176 |
| Labeled Pixels | 20,024 (5.82%) | 10,317 (4.38%) |
| Value Range | [0, 13069] | [0, 11590] |
| Mean Value | 1362.93 | 1348.50 |

### Domain Shift Observations
1. **Geographic Domain Shift**: Different geographic regions with similar land cover classes
2. **Spatial Size Difference**: Dioni is larger (344k vs 235k pixels) - **different dimensions**
3. **Class Imbalance**: Both have severe imbalance with Classes 2 and 4 having very few samples
4. **Same Spectral Alignment**: Both datasets have identical 176 bands
5. **Raw Values**: Data is NOT normalized - requires preprocessing

### Per-Band Statistics Sample

**Dioni:**
| Band | Min | Max | Mean | Std |
|------|-----|-----|------|-----|
| 0 | 0 | 10310 | 2594.71 | 530.00 |
| 88 | 0 | 5562 | 1417.64 | 605.30 |
| 175 | 0 | 572 | 72.41 | 50.44 |

**Loukia:**
| Band | Min | Max | Mean | Std |
|------|-----|-----|------|-----|
| 0 | 0 | 6947 | 2388.81 | 248.58 |
| 88 | 0 | 4365 | 1502.35 | 722.54 |
| 175 | 0 | 553 | 44.44 | 46.62 |

---

### Comparison with Houston Cross-Scene

| Aspect | HyRank (Dioni/Loukia) | Houston (H13/H18) |
|--------|----------------------|-------------------|
| **File Format** | MATLAB v5 (scipy) ✅ | MATLAB v7.3 (h5py) |
| **Loading** | No transpose needed ✅ | Transpose required |
| **Data Type** | uint16 (raw) | float64 (normalized) |
| **Normalization** | Required ⚠️ | Pre-normalized [0,1] |
| **Spectral Bands** | 176 | 48 |
| **Classes** | 12 | 7 |
| **Spatial Size** | Different between scenes | Same (210×954) |
| **Domain Shift** | Geographic | Temporal |

---

### Implementation Guidelines

#### Key Configuration Parameters
```python
# Dataset parameters
NUM_CLASSES = 12  # Classes 1-12 (exclude background 0)
NUM_BANDS = 176
DIONI_SIZE = (250, 1376)
LOUKIA_SIZE = (249, 945)

# Normalization required!
def normalize_data(data):
    # Option 1: Min-Max normalization
    return (data - data.min()) / (data.max() - data.min())
    
    # Option 2: Per-band standardization
    # mean = data.mean(axis=(0,1), keepdims=True)
    # std = data.std(axis=(0,1), keepdims=True)
    # return (data - mean) / (std + 1e-8)
```

#### Critical Considerations
> [!WARNING]
> **Classes 2 and 4 have very few samples** in both scenes (54-204 pixels in Dioni, 54-79 in Loukia) - models may struggle with these classes.

> [!WARNING]
> **Different spatial dimensions** between Dioni and Loukia - cannot directly use patch-based methods that assume same scene size.

> [!IMPORTANT]
> Data is **NOT normalized** - you MUST normalize before training (unlike Houston Cross-Scene which is pre-normalized).

#### Ease of Use Comparison
| Task | HyRank | Houston Cross-Scene |
|------|--------|---------------------|
| Loading | ✅ Easier (scipy, no transpose) | ❌ Harder (h5py, transpose) |
| Preprocessing | ❌ Need normalization | ✅ Pre-normalized |
| Spatial handling | ❌ Different dimensions | ✅ Same dimensions |
| Adaptation | Minor code changes | Reference implementation |

#### Recommended Use Cases
- **Cross-Scene Classification**: Testing model generalization across different geographic areas
- **Domain Adaptation**: Transfer learning from Dioni to Loukia or vice versa
- **High Spectral Resolution**: 176 bands for detailed spectral analysis
- **Semi-Supervised Learning**: Sparse labels in both scenes (~5%)

---

## CONCLUSION

All four datasets are valuable for hyperspectral remote sensing research:
- **Trento** offers unique multi-modal capabilities (HSI + LiDAR) with smaller computational requirements
- **Houston** provides a standard, high-resolution benchmark for algorithm evaluation with 144 spectral bands
- **Cross-Scene (H13/H18)** enables domain adaptation and transfer learning research with temporal domain shift
- **HyRank (Dioni/Loukia)** provides cross-scene classification with geographic domain shift and high spectral resolution (176 bands)

