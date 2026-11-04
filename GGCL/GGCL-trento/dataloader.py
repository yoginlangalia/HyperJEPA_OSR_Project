"""
Data Loading for Trento Dataset
Handles loading, preprocessing, patch extraction, and dataset creation

DIFFERENCES FROM HOUSTON:
1. No predefined train/test split - uses 80/20 stratified split
2. 63 spectral bands (vs 144 for Houston)
3. 6 classes (vs 15 for Houston)
4. Ground truth has classes 1-6 (0 is background)
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy import io
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import random

from pca import apply_pca


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
    
    # Calculate valid ranges in source image
    r_start = max(0, row - half)
    r_end = min(H, row + half + 1)
    c_start = max(0, col - half)
    c_end = min(W, col + half + 1)
    
    # Calculate corresponding ranges in patch
    pr_start = half - (row - r_start)
    pr_end = half + (r_end - row)
    pc_start = half - (col - c_start)
    pc_end = half + (c_end - col)
    
    # Copy data with transpose to (C, H, W)
    patch[:, pr_start:pr_end, pc_start:pc_end] = \
        data[r_start:r_end, c_start:c_end, :].transpose(2, 0, 1)
    
    return patch


class TrentoDataset(Dataset):
    """
    PyTorch Dataset for Trento HSI data with zero-padding patch extraction.
    
    Args:
        data: normalized HSI cube (H x W x C)
        indices: list of (row, col) tuples for pixels to include
        labels: list of labels corresponding to indices
        patch_size: spatial patch size (default: 13)
    """
    def __init__(self, data, indices, labels, patch_size=13):
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


def load_trento_data(data_path, normalize=True, n_components=None, test_ratio=0.2, seed=42):
    """
    Load Trento dataset from .mat file and create 80/20 train/test split
    
    Args:
        data_path: path to Trento.mat file
        normalize: whether to normalize the data
        n_components: number of PCA components (None = no PCA)
        test_ratio: ratio of test samples (default: 0.2 for 80/20 split)
        seed: random seed for reproducibility
        
    Returns:
        data: HSI cube (H x W x C)
        train_gt: training ground truth
        test_gt: test ground truth
        info: dict with dataset info
    """
    print(f"Loading Trento dataset from {data_path}")
    set_seed(seed)
    
    mat = io.loadmat(data_path)
    
    # Load HSI data - just convert to float32 (don't reinterpret bytes!)
    if 'HSI_data' in mat:
        data = mat['HSI_data'].astype(np.float32)
    elif 'HSI' in mat:
        data = mat['HSI'].astype(np.float32)
    else:
        # Try to find the HSI data key
        for key in mat.keys():
            if not key.startswith('__') and len(mat[key].shape) == 3:
                data = mat[key].astype(np.float32)
                print(f"  Found HSI data in key: {key}")
                break
    
    # Load ground truth
    if 'GT' in mat:
        gt = mat['GT'].astype(np.int64)
    elif 'gt' in mat:
        gt = mat['gt'].astype(np.int64)
    elif 'LiDAR_data' in mat:
        # LiDAR_data might be confused with GT, look for actual GT
        for key in mat.keys():
            if not key.startswith('__') and len(mat[key].shape) == 2 and 'lidar' not in key.lower():
                if mat[key].max() <= 10:  # Likely class labels
                    gt = mat[key].astype(np.int64)
                    print(f"  Found GT in key: {key}")
                    break
    
    print(f"Original data shape: {data.shape}")  # Should be (166, 600, 63)
    print(f"Ground truth shape: {gt.shape}")
    
    # Class distribution
    unique_classes = np.unique(gt)
    print(f"Classes found: {unique_classes}")
    
    # Normalize BEFORE PCA - use z-score normalization to match GGCL_trento
    if normalize:
        data = (data - data.mean()) / data.std()
    
    # Apply PCA if requested
    if n_components is not None and n_components < data.shape[2]:
        data = apply_pca(data, n_components=n_components)
    
    # Create 80/20 stratified train/test split
    train_gt, test_gt = create_train_test_split(gt, test_ratio=test_ratio, seed=seed)
    
    # Dataset info
    num_classes = int(gt.max())  # Classes 1-6 (0 is background)
    info = {
        'num_classes': num_classes,
        'spectral_size': data.shape[2],
        'spatial_size': data.shape[:2],
        'n_train': np.count_nonzero(train_gt),
        'n_test': np.count_nonzero(test_gt),
        'label_values': [
            'Apple trees', 'Buildings', 'Ground', 
            'Wood', 'Vineyard', 'Roads'
        ]
    }
    
    # Print class distribution
    print("\nClass distribution (after 80/20 split):")
    for i in range(1, num_classes + 1):
        n_train = np.sum(train_gt == i)
        n_test = np.sum(test_gt == i)
        print(f"  Class {i}: Train={n_train}, Test={n_test}")
    
    return data, train_gt, test_gt, info


def create_train_test_split(gt, test_ratio=0.2, seed=42):
    """
    Create stratified 80/20 train/test split from ground truth
    
    Args:
        gt: ground truth labels (H x W), 0=background, 1-6=classes
        test_ratio: ratio of test samples (default: 0.2)
        seed: random seed
        
    Returns:
        train_gt: training labels (H x W)
        test_gt: test labels (H x W)
    """
    set_seed(seed)
    
    train_gt = np.zeros_like(gt)
    test_gt = np.zeros_like(gt)
    
    num_classes = int(gt.max())
    
    print(f"\nCreating 80/20 train/test split (seed={seed}):")
    
    for c in range(1, num_classes + 1):
        # Find all pixels of this class
        rows, cols = np.where(gt == c)
        indices = list(range(len(rows)))
        
        if len(indices) == 0:
            continue
        
        # Stratified split
        train_idx, test_idx = train_test_split(
            indices, test_size=test_ratio, random_state=seed
        )
        
        # Assign to train/test ground truth
        for idx in train_idx:
            train_gt[rows[idx], cols[idx]] = c
        for idx in test_idx:
            test_gt[rows[idx], cols[idx]] = c
        
        print(f"  Class {c}: {len(train_idx)} train, {len(test_idx)} test")
    
    print(f"  Total: {np.count_nonzero(train_gt)} train, {np.count_nonzero(test_gt)} test")
    
    return train_gt, test_gt


def sample_few_shot(train_gt, n_shot, num_classes=6, seed=42):
    """
    Sample n_shot labeled samples per class from training set.
    Remaining train samples become "unlabeled" (for contrastive learning).
    
    Args:
        train_gt: training labels (H x W), 0=background, 1-6=classes
        n_shot: number of samples per class
        num_classes: number of classes (6 for Trento)
        seed: random seed for reproducibility
        
    Returns:
        labeled_indices: list of (row, col) for labeled samples
        labeled_labels: corresponding labels (0-indexed)
        unlabeled_indices: list of (row, col) for unlabeled samples
    """
    set_seed(seed)
    
    labeled_indices = []
    labeled_labels = []
    unlabeled_indices = []
    
    print(f"\nFew-shot sampling ({n_shot}-shot, seed={seed}):")
    
    for c in range(1, num_classes + 1):
        # Find all pixels of this class in training set
        rows, cols = np.where(train_gt == c)
        class_indices = list(zip(rows.tolist(), cols.tolist()))
        
        # Shuffle
        random.shuffle(class_indices)
        
        # Split: first n_shot as labeled, rest as unlabeled
        if len(class_indices) < n_shot:
            print(f"  Warning: Class {c} has only {len(class_indices)} samples")
            labeled = class_indices
            unlabeled = []
        else:
            labeled = class_indices[:n_shot]
            unlabeled = class_indices[n_shot:]
        
        # Add to lists (convert labels to 0-indexed)
        labeled_indices.extend(labeled)
        labeled_labels.extend([c - 1] * len(labeled))
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
    test_indices = list(zip(rows.tolist(), cols.tolist()))
    test_labels = [test_gt[r, c] - 1 for r, c in test_indices]  # 0-indexed
    
    return test_indices, test_labels


def create_dataloaders(data, train_gt, test_gt, 
                       patch_size=13, batch_size=64, 
                       num_workers=0, use_unlabeled=True,
                       n_shot=10, seed=42):
    """
    Create PyTorch DataLoaders for training and testing.
    
    FIXES:
    - drop_last=False for training (critical for few-shot)
    - Zero-padding for all pixels
    - Proper few-shot sampling
    
    Returns:
        labeled_loader: DataLoader for few-shot labeled samples
        test_loader: DataLoader for test samples
        unlabeled_loader: DataLoader for unlabeled samples (rest of train)
    """
    set_seed(seed)
    
    num_classes = int(train_gt.max())  # 6 for Trento
    
    # Sample few-shot from training set
    labeled_indices, labeled_labels, unlabeled_indices = sample_few_shot(
        train_gt, n_shot, num_classes=num_classes, seed=seed
    )
    
    # Get test indices
    test_indices, test_labels = get_test_indices(test_gt)
    
    # Create datasets
    labeled_dataset = TrentoDataset(
        data, labeled_indices, labeled_labels, patch_size=patch_size
    )
    
    test_dataset = TrentoDataset(
        data, test_indices, test_labels, patch_size=patch_size
    )
    
    # CRITICAL FIX: drop_last=False for training
    labeled_loader = DataLoader(
        labeled_dataset,
        batch_size=min(batch_size, len(labeled_dataset)),
        shuffle=True,
        num_workers=num_workers,
        drop_last=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers
    )
    
    unlabeled_loader = None
    if use_unlabeled and len(unlabeled_indices) > 0:
        unlabeled_dataset = TrentoDataset(
            data, unlabeled_indices,
            [-1] * len(unlabeled_indices),
            patch_size=patch_size
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
    print(f"  Labeled samples: {len(labeled_dataset)} ({len(labeled_loader)} batches)")
    print(f"  Test samples: {len(test_dataset)} ({len(test_loader)} batches)")
    if unlabeled_loader:
        print(f"  Unlabeled samples: {len(unlabeled_loader.dataset)} ({len(unlabeled_loader)} batches)")
    
    return labeled_loader, test_loader, unlabeled_loader


class InfiniteDataLoader:
    """
    Wrapper for infinite iteration over a DataLoader.
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
