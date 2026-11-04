#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data loading and preprocessing for SSLSM Houston Cross-Scene (Houston 2013 → 2018)
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import h5py
import os
import random


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def applyPCA(X, numComponents=15):
    """Apply PCA with whitening for dimensionality reduction and normalization."""
    H, W, B = X.shape
    newX = np.reshape(X, (-1, B))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    return np.reshape(newX, (H, W, numComponents))


def padWithZeros(X, margin=2):
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
    
    patchesData = np.zeros((H * W, windowSize, windowSize, C), dtype=np.float32)
    patchesLabels = np.zeros((H * W), dtype=np.int64)
    patchIndex = 0
    
    for r in range(margin, zeroPaddedX.shape[0] - margin):
        for c in range(margin, zeroPaddedX.shape[1] - margin):
            patch = zeroPaddedX[r - margin:r + margin + 1, c - margin:c + margin + 1]
            patchesData[patchIndex] = patch
            patchesLabels[patchIndex] = y[r - margin, c - margin]
            patchIndex += 1
            
    if removeZeroLabels:
        mask = patchesLabels > 0
        patchesData = patchesData[mask]
        patchesLabels = patchesLabels[mask] - 1  # 0-indexed
        
    return patchesData, patchesLabels


def load_cross_scene_data(h13_path, h13_gt_path, h18_path, h18_gt_path,
                           pca_components=15, patch_size=15, test_ratio=0.2, seed=42):
    """
    Load Houston-13 and Houston-18, apply PCA, extract patches, split H18.
    """
    set_seed(seed)
    
    print("Loading Houston-13...")
    with h5py.File(h13_path, 'r') as f:
        X13 = np.transpose(f['ori_data'][()]).astype(np.float32)
    with h5py.File(h13_gt_path, 'r') as f:
        y13 = np.transpose(f['map'][()]).astype(np.int32)
        
    print("Loading Houston-18...")
    with h5py.File(h18_path, 'r') as f:
        X18 = np.transpose(f['ori_data'][()]).astype(np.float32)
    with h5py.File(h18_gt_path, 'r') as f:
        y18 = np.transpose(f['map'][()]).astype(np.int32)
        
    print(f"H13 raw shape: {X13.shape}, labeled pixels: {np.sum(y13 > 0)}")
    print(f"H18 raw shape: {X18.shape}, labeled pixels: {np.sum(y18 > 0)}")
    
    print(f"Applying PCA (components={pca_components}, whiten=True)...")
    X13_pca = applyPCA(X13, numComponents=pca_components)
    X18_pca = applyPCA(X18, numComponents=pca_components)
    
    print("Extracting spatial patches...")
    X13_patches, Y13_patches = createImageCubes(X13_pca, y13, windowSize=patch_size, removeZeroLabels=True)
    X18_patches, Y18_patches = createImageCubes(X18_pca, y18, windowSize=patch_size, removeZeroLabels=True)
    
    # Transpose patches to PyTorch format: (N, bands, patch_size, patch_size)
    X13_patches = X13_patches.transpose(0, 3, 1, 2)
    X18_patches = X18_patches.transpose(0, 3, 1, 2)
    
    # Split H18 into 80% train / 20% test
    X18_tr, X18_te, Y18_tr, Y18_te = train_test_split(
        X18_patches, Y18_patches, test_size=test_ratio, random_state=seed, stratify=Y18_patches
    )
    
    print(f"H13 patches (Source pool)     : {X13_patches.shape}")
    print(f"H18 Train patches (Adaptation): {X18_tr.shape}")
    print(f"H18 Test patches (Evaluation) : {X18_te.shape}")
    
    return X13_patches, Y13_patches, X18_tr, Y18_tr, X18_te, Y18_te


class SSLSMDataset(Dataset):
    def __init__(self, x_data, y_data):
        self.x_data = torch.FloatTensor(x_data)
        self.y_data = torch.LongTensor(y_data)
        self.len = len(x_data)
        
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
        
    def __len__(self):
        return self.len
