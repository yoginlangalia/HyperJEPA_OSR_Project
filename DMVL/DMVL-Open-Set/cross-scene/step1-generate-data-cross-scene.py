#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DMVL Data Generation for Cross-Scene Dataset
Houston-13 (Source) and Houston-18 (Target)

Pre-training: All Houston-13 patches
Experiment 1: Few-shots from H13, test on H18 test
Experiment 2: Few-shots from H18 train, test on H18 test
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import cv2
import os
import sys
import h5py

# Set seed for reproducibility
SEED = 42
np.random.seed(SEED)

def find_file(rel_name, fallback_rel):
    fname = os.path.basename(rel_name)
    if os.path.isfile(rel_name):
        return rel_name
    script_dir = os.path.dirname(os.path.abspath(__file__))
    p1 = os.path.join(script_dir, rel_name)
    if os.path.isfile(p1):
        return p1
    p_local_data = os.path.join(script_dir, 'data', fname)
    if os.path.isfile(p_local_data):
        return p_local_data
    p2 = os.path.normpath(os.path.join(script_dir, fallback_rel))
    if os.path.isfile(p2):
        return p2
    # Search /kaggle/input automatically
    if os.path.isdir('/kaggle/input'):
        for root, _, files in os.walk('/kaggle/input'):
            if fname in files:
                found = os.path.join(root, fname)
                print(f"Auto-detected in Kaggle input: {found}")
                return found
    # Search current directory recursively
    for root, _, files in os.walk('.'):
        if fname in files:
            return os.path.join(root, fname)
    return rel_name

# Dataset paths
H13_DATA_PATH = find_file('./data/Houston13.mat', '../../DMVL-cross-scene/data/Houston13.mat')
H13_GT_PATH   = find_file('./data/Houston13_7gt.mat', '../../DMVL-cross-scene/data/Houston13_7gt.mat')
H18_DATA_PATH = find_file('./data/Houston18.mat', '../../DMVL-cross-scene/data/Houston18.mat')
H18_GT_PATH   = find_file('./data/Houston18_7gt.mat', '../../DMVL-cross-scene/data/Houston18_7gt.mat')

print("=== Loading Houston-13 (Source Domain) ===")
print(f"Loading from: {H13_DATA_PATH}")
with h5py.File(H13_DATA_PATH, 'r') as f:
    h13_img = np.transpose(f['ori_data'][()])  # (210, 954, 48)
with h5py.File(H13_GT_PATH, 'r') as f:
    h13_gt = np.transpose(f['map'][()]).astype(np.int32)  # (210, 954)

print(f"Houston-13 Image shape: {h13_img.shape}")
print(f"Houston-13 GT shape: {h13_gt.shape}")
print(f"Houston-13 Classes: {np.unique(h13_gt)}")
print(f"Houston-13 Labeled pixels: {np.sum(h13_gt > 0)}")

print("\n=== Loading Houston-18 (Target Domain) ===")
print(f"Loading from: {H18_DATA_PATH}")
with h5py.File(H18_DATA_PATH, 'r') as f:
    h18_img = np.transpose(f['ori_data'][()])  # (210, 954, 48)
with h5py.File(H18_GT_PATH, 'r') as f:
    h18_gt = np.transpose(f['map'][()]).astype(np.int32)  # (210, 954)

print(f"Houston-18 Image shape: {h18_img.shape}")
print(f"Houston-18 GT shape: {h18_gt.shape}")
print(f"Houston-18 Classes: {np.unique(h18_gt)}")
print(f"Houston-18 Labeled pixels: {np.sum(h18_gt > 0)}")

# Get dimensions
m, n, n_bands = h13_img.shape
NUM_CLASSES = 8  # 7 classes + background

def Patch(data, height_index, width_index, PATCH_SIZE):
    height_slice = slice(height_index - PATCH_SIZE, height_index + PATCH_SIZE)
    width_slice = slice(width_index - PATCH_SIZE, width_index + PATCH_SIZE)
    patch = data[height_slice, width_slice, :]
    patch = patch.reshape(-1, patch.shape[0] * patch.shape[1] * patch.shape[2])
    return patch

def process_image(img, m, n):
    """Apply PCA to split bands and normalize"""
    split_point = n_bands // 2  # 24
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

def extract_patches(img1, img2, labels_padded, PATCH_SIZE, mm, nn, n):
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
# Process Houston-13 (ALL labeled for pretraining)
# =====================================================
print("\n=== Processing Houston-13 ===")
print(f"Band split: {n_bands} → {n_bands//2}+{n_bands - n_bands//2}")

h13_img1, h13_img2 = process_image(h13_img, m, n)
print(f"PCA completed. View1: {h13_img1.shape}, View2: {h13_img2.shape}")

# Pad images and labels
h13_img1_pad = cv2.copyMakeBorder(h13_img1, PATCH_SIZE, PATCH_SIZE, PATCH_SIZE, PATCH_SIZE, cv2.BORDER_CONSTANT, value=(0,0,0))
h13_img2_pad = cv2.copyMakeBorder(h13_img2, PATCH_SIZE, PATCH_SIZE, PATCH_SIZE, PATCH_SIZE, cv2.BORDER_CONSTANT, value=(0,0,0))
h13_gt_pad = cv2.copyMakeBorder(h13_gt, PATCH_SIZE, PATCH_SIZE, PATCH_SIZE, PATCH_SIZE, cv2.BORDER_CONSTANT, value=(0))

mm, nn, _ = h13_img1_pad.shape

print("Extracting Houston-13 patches (ALL labeled for pretraining)...")
h13_data, h13_label = extract_patches(h13_img1_pad, h13_img2_pad, h13_gt_pad, PATCH_SIZE, mm, nn, n)
print(f"Houston-13 pretrain data: {h13_data.shape}")
print(f"Houston-13 labels: {h13_label.shape}, range: {h13_label.min()}-{h13_label.max()}")

# =====================================================
# Process Houston-18 (80/20 train/test split)
# =====================================================
print("\n=== Processing Houston-18 ===")

h18_img1, h18_img2 = process_image(h18_img, m, n)
print(f"PCA completed. View1: {h18_img1.shape}, View2: {h18_img2.shape}")

# Create 80/20 stratified split
print("\nCreating 80/20 stratified train/test split...")

labeled_coords = []
labeled_labels = []
for i in range(m):
    for j in range(n):
        if h18_gt[i, j] > 0:
            labeled_coords.append((i, j))
            labeled_labels.append(h18_gt[i, j])

labeled_coords = np.array(labeled_coords)
labeled_labels = np.array(labeled_labels)

train_idx, test_idx = train_test_split(
    np.arange(len(labeled_labels)),
    test_size=0.2,
    random_state=SEED,
    stratify=labeled_labels
)

# Create TR and TE masks
h18_TR = np.zeros((m, n), dtype=np.int32)
h18_TE = np.zeros((m, n), dtype=np.int32)

for idx in train_idx:
    i, j = labeled_coords[idx]
    h18_TR[i, j] = labeled_labels[idx]

for idx in test_idx:
    i, j = labeled_coords[idx]
    h18_TE[i, j] = labeled_labels[idx]

print(f"Houston-18 Train samples: {np.sum(h18_TR > 0)} (80%)")
print(f"Houston-18 Test samples: {np.sum(h18_TE > 0)} (20%)")

# Print per-class distribution
print("\nHouston-18 class distribution:")
for c in range(1, 8):
    train_count = np.sum(h18_TR == c)
    test_count = np.sum(h18_TE == c)
    print(f"  Class {c}: Train={train_count}, Test={test_count}")

# Pad Houston-18 images and labels
h18_img1_pad = cv2.copyMakeBorder(h18_img1, PATCH_SIZE, PATCH_SIZE, PATCH_SIZE, PATCH_SIZE, cv2.BORDER_CONSTANT, value=(0,0,0))
h18_img2_pad = cv2.copyMakeBorder(h18_img2, PATCH_SIZE, PATCH_SIZE, PATCH_SIZE, PATCH_SIZE, cv2.BORDER_CONSTANT, value=(0,0,0))
h18_TR_pad = cv2.copyMakeBorder(h18_TR, PATCH_SIZE, PATCH_SIZE, PATCH_SIZE, PATCH_SIZE, cv2.BORDER_CONSTANT, value=(0))
h18_TE_pad = cv2.copyMakeBorder(h18_TE, PATCH_SIZE, PATCH_SIZE, PATCH_SIZE, PATCH_SIZE, cv2.BORDER_CONSTANT, value=(0))

print("\nExtracting Houston-18 TRAIN patches...")
h18_train_data, h18_train_label = extract_patches(h18_img1_pad, h18_img2_pad, h18_TR_pad, PATCH_SIZE, mm, nn, n)
print(f"Houston-18 train data: {h18_train_data.shape}")
print(f"Houston-18 train labels: {h18_train_label.shape}, range: {h18_train_label.min()}-{h18_train_label.max()}")

print("\nExtracting Houston-18 TEST patches...")
h18_test_data, h18_test_label = extract_patches(h18_img1_pad, h18_img2_pad, h18_TE_pad, PATCH_SIZE, mm, nn, n)
print(f"Houston-18 test data: {h18_test_data.shape}")
print(f"Houston-18 test labels: {h18_test_label.shape}, range: {h18_test_label.min()}-{h18_test_label.max()}")

# =====================================================
# Save to HDF5 files
# =====================================================
print("\n=== Saving Data ===")
script_dir = os.path.dirname(os.path.abspath(__file__))
out_data_dir = os.path.join(script_dir, 'data')
os.makedirs(out_data_dir, exist_ok=True)

# Houston-13 pretrain (ALL labeled)
f = h5py.File(os.path.join(out_data_dir, 'Houston13_pretrain.h5'), 'w')
f['data'] = h13_data
f['label'] = h13_label
f.close()
print(f"Saved: {os.path.join(out_data_dir, 'Houston13_pretrain.h5')} ({len(h13_label)} samples)")

# Houston-18 train (80%)
f = h5py.File(os.path.join(out_data_dir, 'Houston18_train.h5'), 'w')
f['data'] = h18_train_data
f['label'] = h18_train_label
f.close()
print(f"Saved: {os.path.join(out_data_dir, 'Houston18_train.h5')} ({len(h18_train_label)} samples)")

# Houston-18 test (20%)
f = h5py.File(os.path.join(out_data_dir, 'Houston18_test.h5'), 'w')
f['data'] = h18_test_data
f['label'] = h18_test_label
f.close()
print(f"Saved: {os.path.join(out_data_dir, 'Houston18_test.h5')} ({len(h18_test_label)} samples)")

print(f"\n=== Data Generation Complete ===")
print(f"Houston-13 pretrain: {len(h13_label)} samples (for pretraining)")
print(f"Houston-18 train: {len(h18_train_label)} samples (for Exp2 few-shot)")
print(f"Houston-18 test: {len(h18_test_label)} samples (for evaluation)")
print(f"Number of classes: 7")
print(f"Patch size: {PATCH_SIZE*2}x{PATCH_SIZE*2}x6")
