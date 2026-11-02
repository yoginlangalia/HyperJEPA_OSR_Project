#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DMVL Data Generation for Trento Dataset
Adapted from Houston implementation

Changes:
- Trento dataset: 63 bands, 6 classes
- No predefined TR/TE split - create 80/20 stratified split
- 63 bands split into 31+32
- No data leakage: pretrain on TR only
"""

import scipy.io as sio
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import cv2
import os
import h5py

# Set seed for reproducibility
SEED = 42
np.random.seed(SEED)

# Trento dataset path (relative path for Kaggle)
DATA_PATH = './data/trento.mat'

# Load Trento data
print("Loading Trento dataset...")
mat_data = sio.loadmat(DATA_PATH)
img = mat_data['HSI_data']  # (166, 600, 63)
gt = mat_data['ground']     # Ground truth (166, 600)

m, n = gt.shape
n_bands = img.shape[2]
print(f"Image shape: {img.shape}")
print(f"Ground truth shape: {gt.shape}")
print(f"Number of bands: {n_bands}")
print(f"Unique classes: {np.unique(gt)}")
print(f"Total labeled pixels: {np.sum(gt > 0)}")

num_classes = 7  # Trento has 6 classes + background

def Patch(data, height_index, width_index, PATCH_SIZE):
    height_slice = slice(height_index - PATCH_SIZE, height_index + PATCH_SIZE)
    width_slice = slice(width_index - PATCH_SIZE, width_index + PATCH_SIZE)
    patch = data[height_slice, width_slice, :]
    patch = patch.reshape(-1, patch.shape[0] * patch.shape[1] * patch.shape[2])
    return patch

# Split 63 bands into two views: 31 + 32
split_point = n_bands // 2  # 31
img1 = img[:, :, :split_point]   # First 31 bands
img2 = img[:, :, split_point:]   # Last 32 bands

print(f"View 1 bands: 0-{split_point-1} ({img1.shape[2]} bands)")
print(f"View 2 bands: {split_point}-{n_bands-1} ({img2.shape[2]} bands)")

img1 = img1.reshape(-1, split_point)
img2 = img2.reshape(-1, n_bands - split_point)

pca = PCA(n_components=3)

reduced_img1 = pca.fit_transform(img1)
img_max, img_min = reduced_img1.max(), reduced_img1.min()
img1 = (reduced_img1 - img_min) / (img_max - img_min)
img1 = img1.reshape(m, n, 3)

reduced_img2 = pca.fit_transform(img2)
img_max, img_min = reduced_img2.max(), reduced_img2.min()
img2 = (reduced_img2 - img_min) / (img_max - img_min)
img2 = img2.reshape(m, n, 3)

print(f"PCA completed. View1 shape: {img1.shape}, View2 shape: {img2.shape}")

# Create 80/20 train/test split (Trento has no predefined split)
print("\nCreating 80/20 stratified train/test split...")

# Get all labeled pixel coordinates and labels
labeled_coords = []
labeled_labels = []
for i in range(m):
    for j in range(n):
        if gt[i, j] > 0:
            labeled_coords.append((i, j))
            labeled_labels.append(gt[i, j])

labeled_coords = np.array(labeled_coords)
labeled_labels = np.array(labeled_labels)

# Stratified split
train_idx, test_idx = train_test_split(
    np.arange(len(labeled_labels)),
    test_size=0.2,
    random_state=SEED,
    stratify=labeled_labels
)

# Create TR and TE masks
TR = np.zeros((m, n), dtype=np.int32)
TE = np.zeros((m, n), dtype=np.int32)

for idx in train_idx:
    i, j = labeled_coords[idx]
    TR[i, j] = labeled_labels[idx]

for idx in test_idx:
    i, j = labeled_coords[idx]
    TE[i, j] = labeled_labels[idx]

print(f"Training samples: {np.sum(TR > 0)} (80%)")
print(f"Test samples: {np.sum(TE > 0)} (20%)")

PATCH_SIZE = 14

# Pad images and labels
img1 = cv2.copyMakeBorder(img1, PATCH_SIZE, PATCH_SIZE, PATCH_SIZE, PATCH_SIZE, cv2.BORDER_CONSTANT, value=(0,0,0))
img2 = cv2.copyMakeBorder(img2, PATCH_SIZE, PATCH_SIZE, PATCH_SIZE, PATCH_SIZE, cv2.BORDER_CONSTANT, value=(0,0,0))
TR_padded = cv2.copyMakeBorder(TR, PATCH_SIZE, PATCH_SIZE, PATCH_SIZE, PATCH_SIZE, cv2.BORDER_CONSTANT, value=(0))
TE_padded = cv2.copyMakeBorder(TE, PATCH_SIZE, PATCH_SIZE, PATCH_SIZE, PATCH_SIZE, cv2.BORDER_CONSTANT, value=(0))

[mm, nn, bb] = img1.shape

def extract_patches(img1, img2, labels_padded, PATCH_SIZE, mm, nn, n):
    """Extract patches for pixels where label > 0"""
    data = []
    label = []
    gt_index = []
    
    for i in range(PATCH_SIZE, mm - PATCH_SIZE):
        for j in range(PATCH_SIZE, nn - PATCH_SIZE):
            if labels_padded[i, j] == 0:
                continue
            else:
                temp_data1 = Patch(img1, i, j, PATCH_SIZE)
                temp_data2 = Patch(img2, i, j, PATCH_SIZE)
                temp_data1 = temp_data1.reshape(28, 28, 3)
                temp_data2 = temp_data2.reshape(28, 28, 3)
                temp_data = np.concatenate((temp_data1, temp_data2), 2)
                temp_data = temp_data.reshape(-1)
                data.append(temp_data)
                label.append(labels_padded[i, j] - 1)  # 0-indexed labels
                gt_index.append((i - PATCH_SIZE) * n + j - PATCH_SIZE)
    
    data = np.array(data)
    data = np.squeeze(data)
    data = np.uint8(data * 255)
    label = np.array(label)
    label = np.squeeze(label)
    
    return data, label, gt_index

# Extract patches for TRAINING set (TR) - used for pretraining
print("\nExtracting training patches (TR)...")
train_data, train_label, train_index = extract_patches(img1, img2, TR_padded, PATCH_SIZE, mm, nn, n)
print(f"Training data shape: {train_data.shape}")
print(f"Training labels shape: {train_label.shape}")
print(f"Training label range: {train_label.min()} to {train_label.max()}")

# Extract patches for TEST set (TE) - used only for evaluation
print("\nExtracting test patches (TE)...")
test_data, test_label, test_index = extract_patches(img1, img2, TE_padded, PATCH_SIZE, mm, nn, n)
print(f"Test data shape: {test_data.shape}")
print(f"Test labels shape: {test_label.shape}")
print(f"Test label range: {test_label.min()} to {test_label.max()}")

# Save to h5 files
# Training data (for pretraining and few-shot sampling)
f = h5py.File('data/Trento28-28-6_train.h5', 'w')
f['data'] = train_data
f['label'] = train_label
f.close()
print(f"\nSaved training data to data/Trento28-28-6_train.h5")

# Test data (for evaluation only)
f = h5py.File('data/Trento28-28-6_test.h5', 'w')
f['data'] = test_data
f['label'] = test_label
f.close()
print(f"Saved test data to data/Trento28-28-6_test.h5")

print(f"\n=== Data Generation Complete ===")
print(f"Training samples: {len(train_label)} (80% split)")
print(f"Test samples: {len(test_label)} (20% split)")
print(f"Number of classes: {len(np.unique(train_label))}")
print(f"Patch size: {PATCH_SIZE*2}x{PATCH_SIZE*2}x6")