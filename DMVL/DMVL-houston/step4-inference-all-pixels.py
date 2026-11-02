#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DMVL Inference for All Houston Pixels (Memory-Optimized)
Step 4: Run inference on the ENTIRE Houston image (all 664,845 pixels)

This script uses CHUNKED PROCESSING to avoid memory overflow:
- Processes pixels row by row
- Extracts features in batches
- Never loads all data into memory at once

Output files:
- Houston/Houston_all_pixel_predictions.npz (predictions + class names)
- Houston/Houston_all_pixel_predictions.csv (human-readable)
- Houston/class_index_mapping.txt (class index to name mapping)
"""

import scipy.io as sio
import numpy as np
from sklearn.decomposition import PCA
import cv2
import os
import torch
from torch.utils.data import DataLoader, TensorDataset
import joblib
from tqdm import tqdm
import gc

# Import local modules
import utils
from model import Model

# Set seed for reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# Houston class names (15 classes, 0-indexed)
CLASS_NAMES = {
    0: "Healthy Grass",
    1: "Stressed Grass",
    2: "Synthetic Grass",
    3: "Trees",
    4: "Soil",
    5: "Water",
    6: "Residential",
    7: "Commercial",
    8: "Road",
    9: "Highway",
    10: "Railway",
    11: "Parking Lot 1",
    12: "Parking Lot 2",
    13: "Tennis Court",
    14: "Running Track"
}

# Houston dataset path
DATA_PATH = './data/houston.mat'

# Model path (pre-trained ResNet from step2)
MODEL_PATH = 'results/128_0.5_200_128_50_houston_model.pth'

# SVM model path (5-shot classifier from step3)
SVM_MODEL_PATH = 'results/5shot_svm_model.joblib'

# Output directory
OUTPUT_DIR = 'Houston'

# Processing parameters - adjust if still hitting memory limits
CHUNK_SIZE = 50  # Process this many rows at a time
BATCH_SIZE = 256  # Batch size for feature extraction

print("=" * 70)
print("  DMVL Houston - Inference on All Pixels (Memory-Optimized)")
print("=" * 70)

# ============================================================================
# Step 1: Load Houston Dataset
# ============================================================================
print("\n=== Step 1: Loading Houston Dataset ===")
mat_data = sio.loadmat(DATA_PATH)
img = mat_data['input']  # (349, 1905, 144)

m, n = img.shape[:2]
total_pixels = m * n
print(f"Image shape: {img.shape}")
print(f"Total pixels: {total_pixels} ({m} x {n})")

# Free memory
del mat_data
gc.collect()

# ============================================================================
# Step 2: Apply PCA (same as step1-generate-data-houston.py)
# ============================================================================
print("\n=== Step 2: Applying PCA ===")

# Split 144 bands into two views: 72 + 72
img1 = img[:,:,:72]
img2 = img[:,:,72:]

# Free original image
del img
gc.collect()

img1 = img1.reshape(-1, 72)
img2 = img2.reshape(-1, 72)

pca = PCA(n_components=3)

print("Applying PCA to view 1...")
reduced_img1 = pca.fit_transform(img1)
img_max, img_min = reduced_img1.max(), reduced_img1.min()
img1_norm = (reduced_img1 - img_min) / (img_max - img_min)
img1_norm = img1_norm.reshape(m, n, 3)
del img1, reduced_img1
gc.collect()

print("Applying PCA to view 2...")
reduced_img2 = pca.fit_transform(img2)
img_max, img_min = reduced_img2.max(), reduced_img2.min()
img2_norm = (reduced_img2 - img_min) / (img_max - img_min)
img2_norm = img2_norm.reshape(m, n, 3)
del img2, reduced_img2
gc.collect()

print(f"PCA completed. View1 shape: {img1_norm.shape}, View2 shape: {img2_norm.shape}")

PATCH_SIZE = 14

# Pad images
img1_padded = cv2.copyMakeBorder(img1_norm, PATCH_SIZE, PATCH_SIZE, PATCH_SIZE, PATCH_SIZE, cv2.BORDER_CONSTANT, value=(0,0,0))
img2_padded = cv2.copyMakeBorder(img2_norm, PATCH_SIZE, PATCH_SIZE, PATCH_SIZE, PATCH_SIZE, cv2.BORDER_CONSTANT, value=(0,0,0))
del img1_norm, img2_norm
gc.collect()

print(f"Padded image shape: {img1_padded.shape}")

# ============================================================================
# Step 3: Load Models
# ============================================================================
print("\n=== Step 3: Loading Models ===")

# Load pre-trained ResNet model
model = Model(feature_dim=128).cuda()
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()
print(f"Loaded pre-trained model from {MODEL_PATH}")

# Load SVM classifier
if not os.path.exists(SVM_MODEL_PATH):
    raise FileNotFoundError(
        f"SVM model not found at {SVM_MODEL_PATH}. "
        "Please run the pipeline first (python run.py) to train and save the 5-shot SVM model."
    )
clf = joblib.load(SVM_MODEL_PATH)
print(f"Loaded 5-shot SVM classifier from {SVM_MODEL_PATH}")

# ============================================================================
# Step 4: Process Pixels in Chunks (Extract Features + Predict)
# ============================================================================
print("\n=== Step 4: Processing Pixels in Chunks ===")

# Prepare output array
all_predictions = np.zeros((m, n), dtype=np.int32)

def extract_row_patches(img1_padded, img2_padded, row_start, row_end, PATCH_SIZE, n_cols):
    """Extract patches for a range of rows"""
    patches = []
    for i in range(row_start + PATCH_SIZE, row_end + PATCH_SIZE):
        for j in range(PATCH_SIZE, n_cols + PATCH_SIZE):
            # Extract patch from view 1
            patch1 = img1_padded[i-PATCH_SIZE:i+PATCH_SIZE, j-PATCH_SIZE:j+PATCH_SIZE, :]
            patch1 = patch1.reshape(28, 28, 3)
            
            # Extract patch from view 2
            patch2 = img2_padded[i-PATCH_SIZE:i+PATCH_SIZE, j-PATCH_SIZE:j+PATCH_SIZE, :]
            patch2 = patch2.reshape(28, 28, 3)
            
            # Combine views
            combined = np.concatenate((patch1, patch2), axis=2)
            patches.append(combined)
    
    patches = np.array(patches)
    patches = np.uint8(patches * 255)
    return patches

def extract_features_batch(model, patches, batch_size):
    """Extract features for a batch of patches"""
    # Create dummy labels
    dummy_labels = np.zeros(len(patches), dtype=np.int64)
    
    # Use the test dataset class
    dataset = utils.HsiDataset_test(patches.reshape(-1, 28*28*6), dummy_labels, transform=utils.test_transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    
    features = []
    with torch.no_grad():
        for data_batch, _, _ in loader:
            _, out = model(data_batch.cuda(non_blocking=True))
            features.append(out.cpu().numpy())
    
    return np.concatenate(features, axis=0)

# Process in chunks
num_chunks = (m + CHUNK_SIZE - 1) // CHUNK_SIZE
print(f"Processing {m} rows in {num_chunks} chunks of {CHUNK_SIZE} rows each...")

for chunk_idx in tqdm(range(num_chunks), desc="Processing chunks"):
    row_start = chunk_idx * CHUNK_SIZE
    row_end = min((chunk_idx + 1) * CHUNK_SIZE, m)
    num_rows = row_end - row_start
    
    # Extract patches for this chunk
    patches = extract_row_patches(img1_padded, img2_padded, row_start, row_end, PATCH_SIZE, n)
    
    # Extract features
    features = extract_features_batch(model, patches, BATCH_SIZE)
    
    # Predict
    chunk_predictions = clf.predict(features)
    
    # Store predictions
    all_predictions[row_start:row_end, :] = chunk_predictions.reshape(num_rows, n)
    
    # Free memory
    del patches, features, chunk_predictions
    gc.collect()
    torch.cuda.empty_cache()

print(f"Predictions complete. Shape: {all_predictions.shape}")
print(f"Unique predicted classes: {np.unique(all_predictions)}")

# Free padded images
del img1_padded, img2_padded
gc.collect()

# ============================================================================
# Step 5: Save Outputs
# ============================================================================
print("\n=== Step 5: Saving Outputs ===")

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Save NPZ file
npz_path = os.path.join(OUTPUT_DIR, 'Houston_all_pixel_predictions.npz')
np.savez(npz_path,
         predictions=all_predictions,
         class_names=CLASS_NAMES,
         shape=(m, n),
         num_classes=15)
print(f"Saved predictions to {npz_path}")

# Save CSV file (human-readable)
csv_path = os.path.join(OUTPUT_DIR, 'Houston_all_pixel_predictions.csv')
print(f"Writing CSV file ({total_pixels} rows)...")

with open(csv_path, 'w') as f:
    # Write header
    f.write("row,col,predicted_class_index,predicted_class_name\n")
    
    # Write data row by row to avoid memory issues
    for row in tqdm(range(m), desc="Writing CSV"):
        for col in range(n):
            pred_class = all_predictions[row, col]
            class_name = CLASS_NAMES[pred_class]
            f.write(f"{row},{col},{pred_class},{class_name}\n")

print(f"Saved CSV to {csv_path}")

# Save class name mapping separately for reference
class_mapping_path = os.path.join(OUTPUT_DIR, 'class_index_mapping.txt')
with open(class_mapping_path, 'w') as f:
    f.write("Class Index to Class Name Mapping\n")
    f.write("=" * 40 + "\n")
    for idx, name in CLASS_NAMES.items():
        f.write(f"{idx}: {name}\n")
print(f"Saved class mapping to {class_mapping_path}")

# Print summary statistics
print("\n" + "=" * 70)
print("  INFERENCE COMPLETE!")
print("=" * 70)
print(f"\n📊 Prediction Statistics:")
for class_idx in range(15):
    count = np.sum(all_predictions == class_idx)
    percentage = count / total_pixels * 100
    print(f"   Class {class_idx:2d} ({CLASS_NAMES[class_idx]:15s}): {count:7d} pixels ({percentage:5.2f}%)")

print(f"\n📁 Output Files:")
print(f"   - {npz_path}")
print(f"   - {csv_path}")
print(f"   - {class_mapping_path}")
print("\n" + "=" * 70)
