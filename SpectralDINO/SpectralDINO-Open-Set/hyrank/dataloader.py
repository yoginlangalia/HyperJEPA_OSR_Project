#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data loading and preprocessing for SpectralDINO HyRank Cross-Scene (Dioni → Loukia)
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


def load_mat_flexible(fpath, candidates_data=('ori_data', 'data', 'Dioni', 'Loukia'),
                      candidates_gt=('map', 'gt', 'ground', 'Dioni_gt', 'Loukia_gt')):
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
            return arr.astype(np.float32) if arr.dtype in [np.float32, np.float64, np.uint16] else arr.astype(np.int32)
    except Exception:
        pass
        
    with h5py.File(fpath, 'r') as f:
        for k in candidates_data:
            if k in f:
                return np.transpose(f[k][()]).astype(np.float32)
        for k in candidates_gt:
            if k in f:
                return np.transpose(f[k][()]).astype(np.int32)
    raise KeyError(f"Could not find valid keys in {fpath}")


def load_cross_scene_data(dioni_path, dioni_gt_path=None, loukia_path=None, loukia_gt_path=None,
                          patch_size=15, test_ratio=0.2, seed=42):
    """
    Load Dioni and Loukia, apply MinMax + Z-score normalization, extract 15x15 patches,
    and split Loukia into 80% train and 20% test.
    Supports both separate ground truth files and embedded labels.
    """
    set_seed(seed)
    
    print("Loading Dioni (Source)...")
    X_dioni = load_mat_flexible(dioni_path, candidates_data=('ori_data', 'data', 'input', 'HSI_data', 'Dioni'))
    
    y_dioni = None
    if dioni_gt_path and os.path.isfile(dioni_gt_path):
        try:
            y_dioni = load_mat_flexible(dioni_gt_path, candidates_gt=('map', 'gt', 'ground', 'Dioni_gt', 'labels'))
        except Exception:
            pass
    if y_dioni is None:
        y_dioni = load_mat_flexible(dioni_path, candidates_gt=('map', 'gt', 'ground', 'Dioni_gt', 'labels'))
        
    if y_dioni.ndim > 2:
        y_dioni = y_dioni.squeeze()
        
    print("Loading Loukia (Target)...")
    X_loukia = load_mat_flexible(loukia_path, candidates_data=('ori_data', 'data', 'input', 'HSI_data', 'Loukia'))
    
    y_loukia = None
    if loukia_gt_path and os.path.isfile(loukia_gt_path):
        try:
            y_loukia = load_mat_flexible(loukia_gt_path, candidates_gt=('map', 'gt', 'ground', 'Loukia_gt', 'labels'))
        except Exception:
            pass
    if y_loukia is None:
        y_loukia = load_mat_flexible(loukia_path, candidates_gt=('map', 'gt', 'ground', 'Loukia_gt', 'labels'))
        
    if y_loukia.ndim > 2:
        y_loukia = y_loukia.squeeze()
        
    print(f"Dioni shape : {X_dioni.shape}, labeled: {np.sum(y_dioni > 0)}")
    print(f"Loukia shape: {X_loukia.shape}, labeled: {np.sum(y_loukia > 0)}")
    
    # Step 1: MinMax scaling to [0, 1]
    X_dioni = (X_dioni - np.min(X_dioni)) / (np.max(X_dioni) - np.min(X_dioni) + 1e-8)
    X_loukia = (X_loukia - np.min(X_loukia)) / (np.max(X_loukia) - np.min(X_loukia) + 1e-8)
    
    # Step 2: Z-score standardization
    mean_d, std_d = np.mean(X_dioni), np.std(X_dioni)
    X_dioni = (X_dioni - mean_d) / (std_d + 1e-8)
    
    mean_l, std_l = np.mean(X_loukia), np.std(X_loukia)
    X_loukia = (X_loukia - mean_l) / (std_l + 1e-8)
    
    print("Extracting 15x15 patches...")
    X_dioni_patches, Y_dioni_patches   = createImageCubes(X_dioni, y_dioni, windowSize=patch_size, removeZeroLabels=True)
    X_loukia_patches, Y_loukia_patches = createImageCubes(X_loukia, y_loukia, windowSize=patch_size, removeZeroLabels=True)
    
    # Transpose to (N, C, H, W) for PyTorch
    X_dioni_patches  = np.transpose(X_dioni_patches, (0, 3, 1, 2))
    X_loukia_patches = np.transpose(X_loukia_patches, (0, 3, 1, 2))
    
    # Stratified 80/20 train/test split on Loukia
    X_loukia_train, X_loukia_test, Y_loukia_train, Y_loukia_test = train_test_split(
        X_loukia_patches, Y_loukia_patches, test_size=test_ratio, random_state=seed, stratify=Y_loukia_patches
    )
    
    print(f"Dioni Total Patches       : {len(X_dioni_patches)}")
    print(f"Loukia Train Patches (80%): {len(X_loukia_train)}")
    print(f"Loukia Test Patches  (20%): {len(X_loukia_test)}")
    
    return X_dioni_patches, Y_dioni_patches, X_loukia_train, Y_loukia_train, X_loukia_test, Y_loukia_test


class SpectralDataset(Dataset):
    """PyTorch Dataset for Hyperspectral image patches."""
    def __init__(self, data, labels):
        self.data = torch.from_numpy(data).float()
        self.labels = torch.from_numpy(labels).long()
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
