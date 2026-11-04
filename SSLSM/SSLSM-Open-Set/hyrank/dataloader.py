#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data loading and preprocessing for SSLSM HyRank Cross-Scene (Dioni → Loukia)
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import scipy.io as sio
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


def load_cross_scene_data(dioni_path, dioni_gt_path, loukia_path, loukia_gt_path,
                           pca_components=15, patch_size=15, test_ratio=0.2, seed=42):
    """
    Load Dioni and Loukia, apply PCA, extract patches, split Loukia.
    """
    set_seed(seed)
    
    print("Loading Dioni (Source)...")
    dioni_mat = sio.loadmat(dioni_path)
    dioni_gt_mat = sio.loadmat(dioni_gt_path)
    X_dioni = dioni_mat['ori_data'].astype(np.float32)
    y_dioni = dioni_gt_mat['map'].astype(np.int32)
    
    print("Loading Loukia (Target)...")
    loukia_mat = sio.loadmat(loukia_path)
    loukia_gt_mat = sio.loadmat(loukia_gt_path)
    X_loukia = loukia_mat['ori_data'].astype(np.float32)
    y_loukia = loukia_gt_mat['map'].astype(np.int32)
    
    print(f"Dioni raw shape : {X_dioni.shape}, labeled pixels: {np.sum(y_dioni > 0)}")
    print(f"Loukia raw shape: {X_loukia.shape}, labeled pixels: {np.sum(y_loukia > 0)}")
    
    print(f"Applying PCA (components={pca_components}, whiten=True)...")
    X_dioni_pca  = applyPCA(X_dioni, numComponents=pca_components)
    X_loukia_pca = applyPCA(X_loukia, numComponents=pca_components)
    
    print("Extracting spatial patches...")
    X_dioni_patches, Y_dioni_patches   = createImageCubes(X_dioni_pca, y_dioni, windowSize=patch_size, removeZeroLabels=True)
    X_loukia_patches, Y_loukia_patches = createImageCubes(X_loukia_pca, y_loukia, windowSize=patch_size, removeZeroLabels=True)
    
    # Transpose patches to PyTorch format: (N, bands, patch_size, patch_size)
    X_dioni_patches  = X_dioni_patches.transpose(0, 3, 1, 2)
    X_loukia_patches = X_loukia_patches.transpose(0, 3, 1, 2)
    
    # Split Loukia into 80% train / 20% test
    X_loukia_tr, X_loukia_te, Y_loukia_tr, Y_loukia_te = train_test_split(
        X_loukia_patches, Y_loukia_patches, test_size=test_ratio, random_state=seed, stratify=Y_loukia_patches
    )
    
    print(f"Dioni patches (Source pool)     : {X_dioni_patches.shape}")
    print(f"Loukia Train patches (Adaptation): {X_loukia_tr.shape}")
    print(f"Loukia Test patches (Evaluation) : {X_loukia_te.shape}")
    
    return X_dioni_patches, Y_dioni_patches, X_loukia_tr, Y_loukia_tr, X_loukia_te, Y_loukia_te


class SSLSMDataset(Dataset):
    def __init__(self, x_data, y_data):
        self.x_data = torch.FloatTensor(x_data)
        self.y_data = torch.LongTensor(y_data)
        self.len = len(x_data)
        
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
        
    def __len__(self):
        return self.len
