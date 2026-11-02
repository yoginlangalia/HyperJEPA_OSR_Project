#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DMVL Data Generation for HyRank Cross-Scene Dataset
Dioni (Source) and Loukia (Target)

Pre-training: All Dioni patches
Experiment 1: Few-shots from Dioni, test on Loukia test
Experiment 2: Few-shots from Loukia train, test on Loukia test
"""

import numpy as np
import scipy.io as sio
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import cv2
import os
import h5py

# Set seed for reproducibility
SEED = 42
np.random.seed(SEED)

# Dataset paths
DIONI_DATA_PATH = './data/Dioni.mat'
DIONI_GT_PATH = './data/Dioni_gt_out68.mat'
LOUKIA_DATA_PATH = './data/Loukia.mat'
LOUKIA_GT_PATH = './data/Loukia_gt_out68.mat'

# =====================================================
# Load Dioni (Source Domain)
# =====================================================
print("=== Loading Dioni (Source Domain) ===")
dioni_img = sio.loadmat(DIONI_DATA_PATH)['ori_data']  # (250, 1376, 176)
dioni_gt = sio.loadmat(DIONI_GT_PATH)['map'].astype(np.int32)  # (250, 1376)

# Normalize raw uint16 to float [0, 1]
dioni_img = dioni_img.astype(np.float32) / dioni_img.max()

print(f"Dioni Image shape: {dioni_img.shape}")
print(f"Dioni GT shape: {dioni_gt.shape}")
print(f"Dioni Classes: {np.unique(dioni_gt)}")
print(f"Dioni Labeled pixels: {np.sum(dioni_gt > 0)}")
print(f"Dioni Value range: [{dioni_img.min():.4f}, {dioni_img.max():.4f}]")

# =====================================================
# Load Loukia (Target Domain)
# =====================================================
print("\n=== Loading Loukia (Target Domain) ===")
loukia_img = sio.loadmat(LOUKIA_DATA_PATH)['ori_data']  # (249, 945, 176)
loukia_gt = sio.loadmat(LOUKIA_GT_PATH)['map'].astype(np.int32)  # (249, 945)

# Normalize raw uint16 to float [0, 1]
loukia_img = loukia_img.astype(np.float32) / loukia_img.max()

print(f"Loukia Image shape: {loukia_img.shape}")
print(f"Loukia GT shape: {loukia_gt.shape}")
print(f"Loukia Classes: {np.unique(loukia_gt)}")
print(f"Loukia Labeled pixels: {np.sum(loukia_gt > 0)}")
print(f"Loukia Value range: [{loukia_img.min():.4f}, {loukia_img.max():.4f}]")

# Get dimensions (different for Dioni and Loukia!)
m_dioni, n_dioni, n_bands = dioni_img.shape
m_loukia, n_loukia = loukia_gt.shape
NUM_CLASSES = 13  # 12 classes + background

def Patch(data, height_index, width_index, PATCH_SIZE):
    height_slice = slice(height_index - PATCH_SIZE, height_index + PATCH_SIZE)
    width_slice = slice(width_index - PATCH_SIZE, width_index + PATCH_SIZE)
    patch = data[height_slice, width_slice, :]
    patch = patch.reshape(-1, patch.shape[0] * patch.shape[1] * patch.shape[2])
    return patch

def process_image(img, m, n, n_bands):
    """Apply PCA to split bands and normalize"""
    # Split 176 bands into two views: 88 + 88
    split_point = n_bands // 2  # 88
    img1 = img[:, :, :split_point]
    img2 = img[:, :, split_point:]
    
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
    
    return img1, img2

def extract_patches(img1, img2, labels_padded, PATCH_SIZE, mm, nn):
    """Extract patches for pixels where label > 0"""
    data = []
    label = []
    
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
    
    data = np.array(data)
    data = np.squeeze(data)
    data = np.uint8(data * 255)
    label = np.array(label)
    label = np.squeeze(label)
    
    return data, label

PATCH_SIZE = 14

# =====================================================
# Process Dioni (ALL labeled for pretraining)
# =====================================================
print("\n=== Processing Dioni ===")
print(f"Band split: {n_bands} → {n_bands//2}+{n_bands - n_bands//2}")

dioni_img1, dioni_img2 = process_image(dioni_img, m_dioni, n_dioni, n_bands)
print(f"PCA completed. View1: {dioni_img1.shape}, View2: {dioni_img2.shape}")

# Pad images and labels
dioni_img1_pad = cv2.copyMakeBorder(dioni_img1, PATCH_SIZE, PATCH_SIZE, PATCH_SIZE, PATCH_SIZE, cv2.BORDER_CONSTANT, value=(0,0,0))
dioni_img2_pad = cv2.copyMakeBorder(dioni_img2, PATCH_SIZE, PATCH_SIZE, PATCH_SIZE, PATCH_SIZE, cv2.BORDER_CONSTANT, value=(0,0,0))
dioni_gt_pad = cv2.copyMakeBorder(dioni_gt, PATCH_SIZE, PATCH_SIZE, PATCH_SIZE, PATCH_SIZE, cv2.BORDER_CONSTANT, value=(0))

mm_dioni, nn_dioni, _ = dioni_img1_pad.shape

print("Extracting Dioni patches (ALL labeled for pretraining)...")
dioni_data, dioni_label = extract_patches(dioni_img1_pad, dioni_img2_pad, dioni_gt_pad, PATCH_SIZE, mm_dioni, nn_dioni)
print(f"Dioni pretrain data: {dioni_data.shape}")
print(f"Dioni labels: {dioni_label.shape}, range: {dioni_label.min()}-{dioni_label.max()}")

# =====================================================
# Process Loukia (80/20 train/test split)
# =====================================================
print("\n=== Processing Loukia ===")

loukia_img1, loukia_img2 = process_image(loukia_img, m_loukia, n_loukia, n_bands)
print(f"PCA completed. View1: {loukia_img1.shape}, View2: {loukia_img2.shape}")

# Create 80/20 stratified split
print("\nCreating 80/20 stratified train/test split...")

labeled_coords = []
labeled_labels = []
for i in range(m_loukia):
    for j in range(n_loukia):
        if loukia_gt[i, j] > 0:
            labeled_coords.append((i, j))
            labeled_labels.append(loukia_gt[i, j])

labeled_coords = np.array(labeled_coords)
labeled_labels = np.array(labeled_labels)

train_idx, test_idx = train_test_split(
    np.arange(len(labeled_labels)),
    test_size=0.2,
    random_state=SEED,
    stratify=labeled_labels
)

# Create TR and TE masks
loukia_TR = np.zeros((m_loukia, n_loukia), dtype=np.int32)
loukia_TE = np.zeros((m_loukia, n_loukia), dtype=np.int32)

for idx in train_idx:
    i, j = labeled_coords[idx]
    loukia_TR[i, j] = labeled_labels[idx]

for idx in test_idx:
    i, j = labeled_coords[idx]
    loukia_TE[i, j] = labeled_labels[idx]

print(f"Loukia Train samples: {np.sum(loukia_TR > 0)} (80%)")
print(f"Loukia Test samples: {np.sum(loukia_TE > 0)} (20%)")

# Print per-class distribution
print("\nLoukia class distribution:")
for c in range(1, 13):
    train_count = np.sum(loukia_TR == c)
    test_count = np.sum(loukia_TE == c)
    print(f"  Class {c}: Train={train_count}, Test={test_count}")

# Pad Loukia images and labels
loukia_img1_pad = cv2.copyMakeBorder(loukia_img1, PATCH_SIZE, PATCH_SIZE, PATCH_SIZE, PATCH_SIZE, cv2.BORDER_CONSTANT, value=(0,0,0))
loukia_img2_pad = cv2.copyMakeBorder(loukia_img2, PATCH_SIZE, PATCH_SIZE, PATCH_SIZE, PATCH_SIZE, cv2.BORDER_CONSTANT, value=(0,0,0))
loukia_TR_pad = cv2.copyMakeBorder(loukia_TR, PATCH_SIZE, PATCH_SIZE, PATCH_SIZE, PATCH_SIZE, cv2.BORDER_CONSTANT, value=(0))
loukia_TE_pad = cv2.copyMakeBorder(loukia_TE, PATCH_SIZE, PATCH_SIZE, PATCH_SIZE, PATCH_SIZE, cv2.BORDER_CONSTANT, value=(0))

mm_loukia, nn_loukia, _ = loukia_img1_pad.shape

print("\nExtracting Loukia TRAIN patches...")
loukia_train_data, loukia_train_label = extract_patches(loukia_img1_pad, loukia_img2_pad, loukia_TR_pad, PATCH_SIZE, mm_loukia, nn_loukia)
print(f"Loukia train data: {loukia_train_data.shape}")
print(f"Loukia train labels: {loukia_train_label.shape}, range: {loukia_train_label.min()}-{loukia_train_label.max()}")

print("\nExtracting Loukia TEST patches...")
loukia_test_data, loukia_test_label = extract_patches(loukia_img1_pad, loukia_img2_pad, loukia_TE_pad, PATCH_SIZE, mm_loukia, nn_loukia)
print(f"Loukia test data: {loukia_test_data.shape}")
print(f"Loukia test labels: {loukia_test_label.shape}, range: {loukia_test_label.min()}-{loukia_test_label.max()}")

# =====================================================
# Save to HDF5 files
# =====================================================
print("\n=== Saving Data ===")
os.makedirs('data', exist_ok=True)

# Dioni pretrain (ALL labeled)
f = h5py.File('data/Dioni_pretrain.h5', 'w')
f['data'] = dioni_data
f['label'] = dioni_label
f.close()
print(f"Saved: data/Dioni_pretrain.h5 ({len(dioni_label)} samples)")

# Loukia train (80%)
f = h5py.File('data/Loukia_train.h5', 'w')
f['data'] = loukia_train_data
f['label'] = loukia_train_label
f.close()
print(f"Saved: data/Loukia_train.h5 ({len(loukia_train_label)} samples)")

# Loukia test (20%)
f = h5py.File('data/Loukia_test.h5', 'w')
f['data'] = loukia_test_data
f['label'] = loukia_test_label
f.close()
print(f"Saved: data/Loukia_test.h5 ({len(loukia_test_label)} samples)")

print(f"\n=== Data Generation Complete ===")
print(f"Dioni pretrain: {len(dioni_label)} samples (for pretraining)")
print(f"Loukia train: {len(loukia_train_label)} samples (for Exp2 few-shot)")
print(f"Loukia test: {len(loukia_test_label)} samples (for evaluation)")
print(f"Number of classes: 12")
print(f"Patch size: {PATCH_SIZE*2}x{PATCH_SIZE*2}x6")