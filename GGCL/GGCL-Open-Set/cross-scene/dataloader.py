"""
Data Loading for Cross-Scene Dataset (Houston 2013 / Houston 2018)
Handles HDF5 loading, preprocessing, patch extraction, and dataset creation

KEY CHANGES FROM GGCL-HOUSTON:
1. HDF5 loading with h5py (not scipy.io)
2. 7 classes instead of 15
3. No PCA (only 48 bands)
4. Houston-18 split: 80% Train / 20% Test
5. Pre-training dataloader for H13
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import random


def set_seed(seed=42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def extract_patch(data, row, col, patch_size):
    """
    Extract a patch centered at (row, col) with ZERO-PADDING if needed.
    This ensures ALL labeled pixels can be used, even at boundaries.
    
    Args:
        data: HSI cube (H, W, C)
        row, col: Center pixel coordinates
        patch_size: Size of the patch (odd number)
        
    Returns:
        patch: (C, patch_size, patch_size)
    """
    H, W, C = data.shape
    half = patch_size // 2
    
    # Initialize with zeros
    patch = np.zeros((C, patch_size, patch_size), dtype=np.float32)
    
    # Calculate valid ranges
    r_start = max(0, row - half)
    r_end = min(H, row + half + 1)
    c_start = max(0, col - half)
    c_end = min(W, col + half + 1)
    
    # Calculate patch ranges
    pr_start = half - (row - r_start)
    pr_end = half + (r_end - row)
    pc_start = half - (col - c_start)
    pc_end = half + (c_end - col)
    
    # Copy data (transpose to C, H, W)
    patch[:, pr_start:pr_end, pc_start:pc_end] = data[r_start:r_end, c_start:c_end, :].transpose(2, 0, 1)
    
    return patch


class CrossSceneDataset(Dataset):
    """PyTorch Dataset for Cross-Scene HSI data with zero-padding patch extraction.
    
    Args:
        data: normalized HSI cube (H x W x C)
        indices: list of (row, col) tuples for pixels to include
        labels: list of labels (0-indexed, use -1 for unlabeled)
        patch_size: patch size (default: 15)
    """
    
    def __init__(self, data, indices, labels, patch_size=15):
        self.data = data
        self.indices = indices
        self.labels = labels
        self.patch_size = patch_size
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        row, col = self.indices[idx]
        patch = extract_patch(self.data, row, col, self.patch_size)
        label = self.labels[idx]
        
        return torch.FloatTensor(patch), label


def load_cross_scene_data(h13_path, h13_gt_path, h18_path, h18_gt_path, 
                           test_ratio=0.2, seed=42):
    """
    Load Cross-Scene dataset from HDF5 files.
    
    Args:
        h13_path: path to Houston13.mat
        h13_gt_path: path to Houston13_7gt.mat
        h18_path: path to Houston18.mat
        h18_gt_path: path to Houston18_7gt.mat
        test_ratio: ratio of H18 for testing (default: 0.2 = 20% test, 80% train)
        seed: random seed
        
    Returns:
        h13_data: Houston-13 HSI (H, W, C)
        h13_gt: Houston-13 ground truth (H, W)
        h18_data: Houston-18 HSI (H, W, C)
        h18_gt: Houston-18 full ground truth (H, W)
        h18_train_gt: Houston-18 train labels (H, W)
        h18_test_gt: Houston-18 test labels (H, W)
        info: dataset information dict
    """
    set_seed(seed)
    
    print("=" * 60)
    print("LOADING CROSS-SCENE DATASET")
    print("=" * 60)
    
    # Load Houston-13
    print(f"\nLoading Houston-13 from {h13_path}")
    with h5py.File(h13_path, 'r') as f:
        h13_data = np.array(f['ori_data']).T  # Transpose for correct orientation
    
    with h5py.File(h13_gt_path, 'r') as f:
        h13_gt = np.array(f['map']).T.astype(np.int32)
    
    print(f"  H13 data shape: {h13_data.shape}")
    print(f"  H13 GT shape: {h13_gt.shape}")
    print(f"  H13 data range: [{h13_data.min():.4f}, {h13_data.max():.4f}]")
    print(f"  H13 labeled pixels: {np.count_nonzero(h13_gt)}")
    
    # Load Houston-18
    print(f"\nLoading Houston-18 from {h18_path}")
    with h5py.File(h18_path, 'r') as f:
        h18_data = np.array(f['ori_data']).T
    
    with h5py.File(h18_gt_path, 'r') as f:
        h18_gt = np.array(f['map']).T.astype(np.int32)
    
    print(f"  H18 data shape: {h18_data.shape}")
    print(f"  H18 GT shape: {h18_gt.shape}")
    print(f"  H18 data range: [{h18_data.min():.4f}, {h18_data.max():.4f}]")
    print(f"  H18 labeled pixels: {np.count_nonzero(h18_gt)}")
    
    # Normalize data (Z-score normalization for cross-scene transfer)
    print("\nNormalizing data (Z-score)...")
    h13_data = h13_data.astype(np.float32)
    h13_mean = h13_data.mean()
    h13_std = h13_data.std()
    h13_data = (h13_data - h13_mean) / (h13_std + 1e-8)
    print(f"  H13: mean={h13_mean:.4f}, std={h13_std:.4f}")
    
    h18_data = h18_data.astype(np.float32)
    h18_mean = h18_data.mean()
    h18_std = h18_data.std()
    h18_data = (h18_data - h18_mean) / (h18_std + 1e-8)
    print(f"  H18: mean={h18_mean:.4f}, std={h18_std:.4f}")

    
    # Split H18 into train/test (80/20)
    print(f"\nCreating {int((1-test_ratio)*100)}/{int(test_ratio*100)} train/test split for H18 (seed={seed}):")
    h18_train_gt, h18_test_gt = create_train_test_split(h18_gt, test_ratio=test_ratio, seed=seed)
    
    # Count classes
    num_classes = 7
    print("\nClass distribution:")
    print("  H13:")
    for c in range(1, num_classes + 1):
        n = np.sum(h13_gt == c)
        print(f"    Class {c}: {n}")
    
    print("  H18 (train | test):")
    for c in range(1, num_classes + 1):
        n_train = np.sum(h18_train_gt == c)
        n_test = np.sum(h18_test_gt == c)
        print(f"    Class {c}: {n_train} | {n_test}")
    
    info = {
        'num_classes': num_classes,
        'spectral_size': h13_data.shape[2],  # 48 bands
        'spatial_size_h13': h13_data.shape[:2],
        'spatial_size_h18': h18_data.shape[:2],
        'n_h13_labeled': np.count_nonzero(h13_gt),
        'n_h18_train': np.count_nonzero(h18_train_gt),
        'n_h18_test': np.count_nonzero(h18_test_gt),
        'label_values': [
            'Class 1', 'Class 2', 'Class 3', 'Class 4',
            'Class 5', 'Class 6', 'Class 7'
        ]
    }
    
    return h13_data, h13_gt, h18_data, h18_gt, h18_train_gt, h18_test_gt, info


def create_train_test_split(gt, test_ratio=0.2, seed=42):
    """
    Split ground truth into train/test sets stratified by class.
    
    Args:
        gt: ground truth labels (H, W), 0=background, 1-7=classes
        test_ratio: ratio for test set (default: 0.2 = 20%)
        seed: random seed
        
    Returns:
        train_gt: training labels (H, W)
        test_gt: test labels (H, W)
    """
    np.random.seed(seed)
    
    train_gt = np.zeros_like(gt)
    test_gt = np.zeros_like(gt)
    
    # Split each class
    for c in range(1, 8):  # Classes 1-7
        rows, cols = np.where(gt == c)
        indices = list(zip(rows, cols))
        
        # Shuffle
        np.random.shuffle(indices)
        
        # Split
        n_test = int(len(indices) * test_ratio)
        test_indices = indices[:n_test]
        train_indices = indices[n_test:]
        
        # Assign
        for r, c_idx in train_indices:
            train_gt[r, c_idx] = c
        for r, c_idx in test_indices:
            test_gt[r, c_idx] = c
    
    print(f"  Total: {np.count_nonzero(train_gt)} train, {np.count_nonzero(test_gt)} test")
    
    return train_gt, test_gt


def sample_few_shot(gt, n_shot, num_classes=7, seed=42):
    """
    Sample n_shot labeled samples per class from ground truth.
    Remaining samples become "unlabeled" (for contrastive learning).
    
    Args:
        gt: labels (H x W), 0=background, 1-7=classes
        n_shot: number of samples per class to keep as labeled
        num_classes: number of classes (default: 7)
        seed: random seed
        
    Returns:
        labeled_indices: list of (row, col) for labeled samples
        labeled_labels: corresponding labels (0-indexed)
        unlabeled_indices: list of (row, col) for unlabeled samples
    """
    np.random.seed(seed)
    
    labeled_indices = []
    labeled_labels = []
    unlabeled_indices = []
    
    print(f"\nFew-shot sampling ({n_shot}-shot, seed={seed}):")
    
    for c in range(1, num_classes + 1):
        rows, cols = np.where(gt == c)
        indices = list(zip(rows, cols))
        
        # Shuffle
        np.random.shuffle(indices)
        
        # Take n_shot as labeled (or all if fewer samples)
        n_labeled = min(n_shot, len(indices))
        labeled = indices[:n_labeled]
        unlabeled = indices[n_labeled:]
        
        labeled_indices.extend(labeled)
        labeled_labels.extend([c - 1] * len(labeled))  # 0-indexed
        unlabeled_indices.extend(unlabeled)
        
        print(f"  Class {c}: {len(labeled)} labeled, {len(unlabeled)} unlabeled")
    
    print(f"  Total labeled: {len(labeled_indices)}")
    print(f"  Total unlabeled: {len(unlabeled_indices)}")
    
    return labeled_indices, labeled_labels, unlabeled_indices


def get_test_indices(test_gt):
    """
    Get all test pixel indices from test label map.
    
    Returns:
        test_indices: list of (row, col)
        test_labels: corresponding labels (0-indexed)
    """
    rows, cols = np.where(test_gt > 0)
    test_indices = list(zip(rows, cols))
    test_labels = [test_gt[r, c] - 1 for r, c in test_indices]  # 0-indexed
    
    return test_indices, test_labels


def create_pretrain_dataloader(data, gt, patch_size=15, batch_size=64, num_workers=0):
    """
    Create DataLoader for pre-training on Houston-13 (unsupervised).
    Uses only valid labeled pixels but treats them as unlabeled (label=-1).
    
    Args:
        data: HSI cube
        gt: ground truth (used to get valid pixel indices)
        patch_size: patch size
        batch_size: batch size
        num_workers: number of workers
        
    Returns:
        pretrain_loader: DataLoader
    """
    # Get all valid labeled pixel indices
    rows, cols = np.where(gt > 0)
    indices = list(zip(rows, cols))
    
    # Assign label -1 to all (unsupervised)
    labels = [-1] * len(indices)
    
    print(f"Pre-training on {len(indices)} valid labeled pixels from H13")
    
    dataset = CrossSceneDataset(data, indices, labels, patch_size=patch_size)
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True
    )
    
    print(f"\nPretrain DataLoader:")
    print(f"  Total patches: {len(indices)}")
    print(f"  Batches: {len(loader)}")
    
    return loader


def create_fewshot_dataloaders(data, labeled_gt, test_gt, unlabeled_data,
                               patch_size=15, batch_size=64, num_workers=0,
                               n_shot=10, num_classes=7, seed=42):
    """
    Create PyTorch DataLoaders for few-shot experiments.
    
    Args:
        data: HSI cube for labeled/unlabeled samples
        labeled_gt: ground truth to sample few-shots from
        test_gt: ground truth for test set
        unlabeled_data: HSI cube for unlabeled pool
        patch_size: patch size (default: 15)
        batch_size: batch size
        num_workers: number of workers
        n_shot: number of samples per class
        num_classes: number of classes (default: 7)
        seed: random seed
        
    Returns:
        labeled_loader: DataLoader for labeled samples
        test_loader: DataLoader for test samples
        unlabeled_loader: DataLoader for unlabeled samples
    """
    set_seed(seed)
    
    # Sample few-shot
    labeled_indices, labeled_labels, unlabeled_indices = sample_few_shot(
        labeled_gt, n_shot, num_classes, seed
    )
    
    # Get test indices
    test_indices, test_labels = get_test_indices(test_gt)
    
    # Create datasets
    labeled_dataset = CrossSceneDataset(data, labeled_indices, labeled_labels, patch_size)
    test_dataset = CrossSceneDataset(data, test_indices, test_labels, patch_size)
    unlabeled_dataset = CrossSceneDataset(
        unlabeled_data, unlabeled_indices, [-1] * len(unlabeled_indices), patch_size
    )
    
    # Create DataLoaders
    labeled_loader = DataLoader(
        labeled_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=False  # Critical for few-shot
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    unlabeled_loader = DataLoader(
        unlabeled_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True
    )
    
    print(f"\nDataLoader info:")
    print(f"  Patch size: {patch_size}x{patch_size}")
    print(f"  Labeled samples: {len(labeled_indices)} ({len(labeled_loader)} batches)")
    print(f"  Test samples: {len(test_indices)} ({len(test_loader)} batches)")
    print(f"  Unlabeled samples: {len(unlabeled_indices)} ({len(unlabeled_loader)} batches)")
    
    return labeled_loader, test_loader, unlabeled_loader


class InfiniteDataLoader:
    """Wrapper for infinite iteration over a DataLoader.
    Useful for unlabeled data which may have different size.
    """
    
    def __init__(self, loader):
        self.loader = loader
        self.iterator = iter(loader)
    
    def __next__(self):
        try:
            return next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.loader)
            return next(self.iterator)
    
    def __iter__(self):
        return self
