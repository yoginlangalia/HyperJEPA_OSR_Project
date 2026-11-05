#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data loading and preprocessing for SpectralDINO Houston Cross-Scene (Houston 2013 → 2018)
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import h5py
import scipy.io as sio
import os
import random


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def padWithZeros(X, margin=7):
    newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2 * margin, X.shape[2]), dtype=X.dtype)
    newX[margin:X.shape[0] + margin, margin:X.shape[1] + margin, :] = X
    return newX


def createImageCubes(X, y, windowSize=15, removeZeroLabels=True):
    """
    Extract spatial patches around each pixel.
    Returns:
        patchesData: (N, windowSize, windowSize, C)
        patchesLabels: (N,) 0-indexed
    """
    margin = int((windowSize - 1) / 2)
    zeroPaddedX = padWithZeros(X, margin=margin)
    H, W, C = X.shape
    
    patchesData = []
    patchesLabels = []
    
    labeled_r, labeled_c = np.where(y > 0) if removeZeroLabels else np.where(np.ones_like(y, dtype=bool))
    for r, c in zip(labeled_r, labeled_c):
        patch = zeroPaddedX[r:r + windowSize, c:c + windowSize]
        patchesData.append(patch)
        patchesLabels.append(y[r, c] - 1 if removeZeroLabels else y[r, c])
        
    patchesData = np.array(patchesData, dtype=np.float32)
    patchesLabels = np.array(patchesLabels, dtype=np.int64)
    return patchesData, patchesLabels


def load_mat_flexible(fpath, candidates_data=('ori_data', 'data', 'input', 'HSI_data'),
                      candidates_gt=('map', 'gt', 'ground', 'Houston13_7gt', 'Houston18_7gt')):
    """Load MATLAB file supporting both classic and v7.3 HDF5 formats."""
    try:
        data = sio.loadmat(fpath)
        arr = None
        for k in candidates_data:
            if k in data:
                arr = data[k]
                break
        if arr is None:
            for k in candidates_gt:
                if k in data:
                    arr = data[k]
                    break
        if arr is not None:
            return arr.astype(np.float32) if arr.dtype in [np.float32, np.float64] else arr.astype(np.int32)
    except Exception:
        pass
        
    # Fallback to h5py
    with h5py.File(fpath, 'r') as f:
        for k in candidates_data:
            if k in f:
                return np.transpose(f[k][()]).astype(np.float32)
        for k in candidates_gt:
            if k in f:
                return np.transpose(f[k][()]).astype(np.int32)
    raise KeyError(f"Could not find valid keys in {fpath}")


def load_cross_scene_data(h13_path, h13_gt_path, h18_path, h18_gt_path,
                          patch_size=15, test_ratio=0.2, seed=42):
    """
    Load Houston-13 and Houston-18, apply Z-score normalization, extract 15x15 patches,
    and split Houston-18 into 80% train and 20% test.
    """
    set_seed(seed)
    
    print("Loading Houston-13...")
    X13 = load_mat_flexible(h13_path, candidates_data=('ori_data', 'data', 'Houston13'))
    y13 = load_mat_flexible(h13_gt_path, candidates_gt=('map', 'gt', 'Houston13_7gt'))
    if y13.ndim > 2:
        y13 = y13.squeeze()
        
    print("Loading Houston-18...")
    X18 = load_mat_flexible(h18_path, candidates_data=('ori_data', 'data', 'Houston18'))
    y18 = load_mat_flexible(h18_gt_path, candidates_gt=('map', 'gt', 'Houston18_7gt'))
    if y18.ndim > 2:
        y18 = y18.squeeze()
        
    print(f"H13 shape: {X13.shape}, labeled: {np.sum(y13 > 0)}")
    print(f"H18 shape: {X18.shape}, labeled: {np.sum(y18 > 0)}")
    
    # Z-score normalization per image
    mean13, std13 = np.mean(X13), np.std(X13)
    X13 = (X13 - mean13) / (std13 + 1e-8)
    
    mean18, std18 = np.mean(X18), np.std(X18)
    X18 = (X18 - mean18) / (std18 + 1e-8)
    
    print("Extracting 15x15 patches...")
    X13_patches, Y13_patches = createImageCubes(X13, y13, windowSize=patch_size, removeZeroLabels=True)
    X18_patches, Y18_patches = createImageCubes(X18, y18, windowSize=patch_size, removeZeroLabels=True)
    
    # Transpose to (N, C, H, W) for PyTorch
    X13_patches = np.transpose(X13_patches, (0, 3, 1, 2))
    X18_patches = np.transpose(X18_patches, (0, 3, 1, 2))
    
    # Stratified 80/20 train/test split on Houston-18
    X18_train, X18_test, Y18_train, Y18_test = train_test_split(
        X18_patches, Y18_patches, test_size=test_ratio, random_state=seed, stratify=Y18_patches
    )
    
    print(f"H13 Total Patches       : {len(X13_patches)}")
    print(f"H18 Train Patches (80%) : {len(X18_train)}")
    print(f"H18 Test Patches  (20%) : {len(X18_test)}")
    
    return X13_patches, Y13_patches, X18_train, Y18_train, X18_test, Y18_test


class SpectralDataset(Dataset):
    """PyTorch Dataset for Hyperspectral image patches."""
    def __init__(self, data, labels):
        self.data = torch.from_numpy(data).float()
        self.labels = torch.from_numpy(labels).long()
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
