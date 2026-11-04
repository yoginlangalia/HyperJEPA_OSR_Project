# GGCL for Houston Dataset

**Guided Group Contrastive Learning for Hyperspectral Image Classification**

Based on the IEEE TGRS 2024 paper:
> "Enhancing Hyperspectral Image Classification: Leveraging Unsupervised Information With Guided Group Contrastive Learning"
> 
> Authors: Ben Li, Leyuan Fang, Ning Chen, Jitong Kang, Jun Yue

## Overview

GGCL is an end-to-end framework that adaptively integrates unsupervised information into supervised contrastive learning for hyperspectral image classification. Key features:

- **Transformer-based architecture** for spectral-spatial feature extraction
- **Guided contrastive learning** using pseudo-labels from unlabeled data
- **Dynamic weight scheduling** that shifts from contrastive to supervised learning during training
- **Data augmentation** strategies for creating contrastive views

## File Structure

```
GGCL-houston/
├── run_kaggle.py        # ⭐ Standalone script for Kaggle (run this!)
├── main.py              # Main training script (command line)
├── trainer.py           # GGCL and baseline trainers
├── dataloader.py        # Houston dataset loading + few-shot
├── evaluation.py        # Metrics (OA, AA, Kappa)
├── augment.py           # Data augmentation for contrastive learning
├── func.py              # Weight scheduling functions
├── utils.py             # Utility functions
├── requirements.txt     # Package dependencies
├── models/
│   ├── __init__.py
│   └── cross_transformer.py  # HSINet (Transformer model)
├── data/
│   └── houston.mat      # Dataset file
└── results/
    └── (training outputs)
```

## Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| `seed` | 42 | Random seed for reproducibility |
| `patch_size` | 15×15 | Spatial patch size |
| `epochs` | 100 | Training epochs |
| `batch_size` | 64 | Batch size |
| `lr` | 0.001 | Learning rate |
| `temperature` | 0.5 | Contrastive loss temperature |

## Data Protocol

```
TRAINING:
├── Contrastive Learning: Uses pixels from TRAIN region only
│   (2,832 unlabeled patches from TR area - NO test data leakage)
│
└── Supervised (CE) Loss: Few-shot samples from TR
    ├── 1-shot:  15 labeled pixels (1 per class)
    ├── 5-shot:  75 labeled pixels (5 per class)
    └── 10-shot: 150 labeled pixels (10 per class)

TESTING:
└── Full test set (TE): 12,197 pixels
    (ONLY used for evaluation, never during training)
```

## Usage

### Kaggle (Recommended)

1. Upload the entire `GGCL-houston` folder to Kaggle
2. Upload `houston.mat` (or place it in the `data/` folder)
3. Run the standalone script:

```bash
cd /kaggle/input/ggcl-houston
python run_kaggle.py
```

This will automatically run all three experiments (10-shot, 5-shot, 1-shot).

### Local Training

```bash
# Single experiment
python main.py --data_path ./data/houston.mat --n_shot 10

# Different shot settings
python main.py --data_path ./data/houston.mat --n_shot 5
python main.py --data_path ./data/houston.mat --n_shot 1

# Full options
python main.py \
    --data_path ./data/houston.mat \
    --n_shot 10 \
    --epochs 100 \
    --batch_size 64 \
    --patch_size 15 \
    --seed 42 \
    --gpu 0
```

## Houston Dataset

- **Size**: 349 × 1905 pixels, 144 spectral bands
- **Classes**: 15 land-cover classes
- **Split**: Pre-defined TR (train) and TE (test) labels
- **Training samples**: 2,832 (used for unlabeled contrastive + few-shot labeled)
- **Test samples**: 12,197

### Class Distribution:
| Class | Name | Train | Test |
|-------|------|-------|------|
| 1 | Healthy grass | 198 | 1053 |
| 2 | Stressed grass | 190 | 1064 |
| 3 | Synthetic grass | 192 | 505 |
| 4 | Tree | 188 | 1056 |
| 5 | Soil | 186 | 1056 |
| 6 | Water | 182 | 143 |
| 7 | Residential | 196 | 1072 |
| 8 | Commercial | 191 | 1053 |
| 9 | Road | 193 | 1059 |
| 10 | Highway | 191 | 1036 |
| 11 | Railway | 181 | 1054 |
| 12 | Parking lot 1 | 192 | 1041 |
| 13 | Parking lot 2 | 184 | 285 |
| 14 | Tennis court | 181 | 247 |
| 15 | Running track | 187 | 473 |

## Training Pipeline

```
1. Load Houston data (HSI cube + TR/TE labels)
2. Create few-shot ground truth (n samples per class from TR)
3. Create data loaders:
   - train_loader: few-shot labeled samples
   - unlabel_loader: all TR region pixels (for contrastive)
   - test_loader: all TE samples (for evaluation only)
4. For each epoch:
   a. Combine labeled + unlabeled batches
   b. Apply augmentation → create left/right views
   c. Forward: model(data, left, right) → logits, features
   d. Compute losses:
      - Guided contrastive loss (weight decreases)
      - Cross-entropy loss (weight increases)
   e. Backpropagation + optimization
5. Evaluate every 10 epochs (OA, AA, Kappa)
6. Save best model based on OA
```

## Output

Results will be saved to the `results/` directory:
- `results_10shot.json` - 10-shot experiment results
- `results_5shot.json` - 5-shot experiment results
- `results_1shot.json` - 1-shot experiment results
- `all_results.json` - Combined summary
- `best_model_Xshot.pth` - Model checkpoints

## Citation

If you use this code, please cite the original paper:

```bibtex
@article{li2024enhancing,
  title={Enhancing Hyperspectral Image Classification: Leveraging Unsupervised Information With Guided Group Contrastive Learning},
  author={Li, Ben and Fang, Leyuan and Chen, Ning and Kang, Jitong and Yue, Jun},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  year={2024},
  publisher={IEEE}
}
```

## License

This implementation is for research and educational purposes.
