# GGCL Trento

GGCL (Guided Group Contrastive Learning) implementation for Trento dataset few-shot classification.

## Dataset

**Trento** hyperspectral + LiDAR dataset:
- **Spatial Size**: 166 × 600 pixels
- **Spectral Bands**: 63 bands
- **Classes**: 6 (+ background)
  - Class 1: Apple trees
  - Class 2: Buildings
  - Class 3: Ground
  - Class 4: Wood
  - Class 5: Vineyard
  - Class 6: Roads

**Key Difference from Houston**: No predefined train/test split - uses **80/20 stratified split**.

## Configuration

```python
patch_size = 13
spectral_bands = 63  # No PCA
epochs = 200
seed = 42
train/test = 80/20 stratified split
```

## Usage

### Run All Experiments (Kaggle)
```bash
python run_kaggle.py
```

### Individual Experiments
```bash
python main.py --data_path ./data/Trento.mat --n_shot 10
python main.py --data_path ./data/Trento.mat --n_shot 5
python main.py --data_path ./data/Trento.mat --n_shot 1
```

## Data Preparation

Place the Trento.mat file in the `data/` directory:
```
GGCL-trento/
├── data/
│   └── Trento.mat  # Place dataset here
├── run_kaggle.py
├── main.py
└── ...
```

## Differences from Houston

| Aspect | Houston | Trento |
|--------|---------|--------|
| Spectral bands | 144 | 63 |
| Classes | 15 | 6 |
| Train/Test split | Predefined | 80/20 stratified |
| LiDAR data | No | Available (not used in GGCL) |
