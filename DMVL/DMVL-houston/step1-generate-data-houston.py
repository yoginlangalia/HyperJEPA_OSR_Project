#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DMVL Data Generation for Houston Dataset
Adapted from original Indian Pines implementation

Changes:
- Houston dataset with TR/TE split
- 144 bands split into 72+72
- No data leakage: pretrain on TR only
"""

import scipy.io as sio
import numpy as np
from sklearn.decomposition import PCA
import cv2
import os

# Set seed for reproducibility
SEED = 42
np.random.seed(SEED)

# Houston dataset path (relative path for Kaggle)
DATA_PATH = './data/houston.mat'

# Load Houston data
mat_data = sio.loadmat(DATA_PATH)
img = mat_data['input']  # (349, 1905, 144)
TR = mat_data['TR']      # Training labels (349, 1905)
TE = mat_data['TE']      # Test labels (349, 1905)

m, n = TR.shape
print(f"Image shape: {img.shape}")
print(f"TR shape: {TR.shape}, TE shape: {TE.shape}")
print(f"TR labeled pixels: {np.sum(TR > 0)}, TE labeled pixels: {np.sum(TE > 0)}")

num_classes = 16  # Houston has 15 classes + background

def Patch(data,height_index,width_index,PATCH_SIZE):
    height_slice = slice(height_index-PATCH_SIZE, height_index+PATCH_SIZE)
    width_slice = slice(width_index-PATCH_SIZE, width_index+PATCH_SIZE)
    patch = data[height_slice, width_slice,:]
    patch = patch.reshape(-1,patch.shape[0]*patch.shape[1]*patch.shape[2])
    return patch
#xxx = img.max()
#img = img/xxx

# Split 144 bands into two views: 72 + 72
img1 = img[:,:,:72]
img2 = img[:,:,72:]

img1 = img1.reshape(-1, 72)
img2 = img2.reshape(-1, 72)

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
import h5py

# Training data (for pretraining and few-shot sampling)
f = h5py.File('data/Houston28-28-6_train.h5', 'w')
f['data'] = train_data
f['label'] = train_label
f.close()
print(f"\nSaved training data to data/Houston28-28-6_train.h5")

# Test data (for evaluation only)
f = h5py.File('data/Houston28-28-6_test.h5', 'w')
f['data'] = test_data
f['label'] = test_label
f.close()
print(f"Saved test data to data/Houston28-28-6_test.h5")

print(f"\n=== Data Generation Complete ===")
print(f"Training samples: {len(train_label)} (from TR)")
print(f"Test samples: {len(test_label)} (from TE)")
print(f"Number of classes: {len(np.unique(train_label))}")
print(f"Patch size: {PATCH_SIZE*2}x{PATCH_SIZE*2}x6")