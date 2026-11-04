#!/usr/bin/env python3
"""
GGCL Houston - All-Pixel Inference Script
Predicts class labels for ALL 664,845 pixels in the Houston 2013 dataset.

Usage:
    1. First run: python run_kaggle.py  (trains and saves 5-shot model)
    2. Then run: python step_inference_all_pixels.py  (runs inference on all pixels)

Output:
    - Houston/Houston_all_pixel_predictions.npz (binary file with predictions)
    - Houston/Houston_all_pixel_predictions.csv (human-readable CSV)
    - Houston/class_index_mapping.txt (class name reference)
"""

import os
import sys
import numpy as np
import torch
from scipy import io
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

# ============================================
# Configuration
# ============================================

SEED = 42
BATCH_SIZE = 128
PATCH_SIZE = 13

# Class names for Houston dataset (0-indexed)
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

# Paths
DATA_PATH = './data/houston.mat'
MODEL_PATH = './results/best_model_5shot.pth'
OUTPUT_DIR = './Houston'

# Handle Kaggle paths
if not os.path.exists(DATA_PATH):
    DATA_PATH = '/kaggle/input/houston-dataset/houston.mat'
if not os.path.exists(DATA_PATH):
    DATA_PATH = '/kaggle/input/houston/houston.mat'
if not os.path.exists(DATA_PATH):
    DATA_PATH = '/kaggle/input/houston-dataset/Houston.mat'
    
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = '/kaggle/working/results/best_model_5shot.pth'

if '/kaggle/' in os.getcwd() or os.path.exists('/kaggle/working'):
    OUTPUT_DIR = '/kaggle/working/Houston'

# ============================================
# Set seed for reproducibility
# ============================================

def set_seed(seed):
    """Set random seed for reproducibility"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

# ============================================
# Patch Extraction (matching dataloader.py)
# ============================================

def extract_patch(data, row, col, patch_size):
    """
    Extract a patch centered at (row, col) with ZERO-PADDING if needed.
    This ensures ALL pixels can be processed, even at boundaries.
    
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

# ============================================
# Load Data
# ============================================

def load_houston_data(data_path):
    """
    Load Houston dataset from .mat file and normalize.
    
    Args:
        data_path: path to houston.mat file
        
    Returns:
        data: HSI cube (H x W x C) after normalization
        shape_info: tuple (height, width, bands)
    """
    print(f"Loading Houston dataset from {data_path}")
    
    mat = io.loadmat(data_path)
    
    # Load HSI data - shape: (349, 1905, 144)
    data = mat['input'].astype(np.float32)
    
    print(f"Original data shape: {data.shape}")
    print(f"Total pixels: {data.shape[0] * data.shape[1]:,}")
    
    # Normalize using MinMaxScaler (same as training)
    h, w, c = data.shape
    data_reshaped = data.reshape(-1, c)
    scaler = MinMaxScaler()
    data_reshaped = scaler.fit_transform(data_reshaped)
    data = data_reshaped.reshape(h, w, c)
    
    print(f"Data normalized with MinMaxScaler")
    
    return data, (h, w, c)

# ============================================
# Load Model
# ============================================

def load_model(model_path, device):
    """
    Load the trained 5-shot GGCL model.
    
    Args:
        model_path: path to saved model weights
        device: torch device
        
    Returns:
        model: loaded HSINet model in eval mode
    """
    print(f"Loading model from {model_path}")
    
    # Import model
    try:
        from models.cross_transformer import HSINet
    except ImportError:
        # Add current directory to path
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from models.cross_transformer import HSINet
    
    # Model parameters (matching run_kaggle.py CONFIG)
    params = {
        'data': {
            'num_classes': 15,
            'spectral_size': 144,  # All 144 bands, no PCA
            'patch_size': PATCH_SIZE,
        },
        'net': {
            'depth': 1,
            'heads': 8,
            'dim': 64,
            'mlp_dim': 8,
            'dropout': 0,
        }
    }
    
    # Create model
    model = HSINet(params).to(device)
    
    # Load weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    print(f"Model loaded successfully")
    print(f"  - Spectral bands: {params['data']['spectral_size']}")
    print(f"  - Patch size: {params['data']['patch_size']}x{params['data']['patch_size']}")
    print(f"  - Num classes: {params['data']['num_classes']}")
    
    return model

# ============================================
# Run Inference
# ============================================

def run_inference(model, data, device, batch_size=128):
    """
    Run inference on ALL pixels in the dataset.
    
    Args:
        model: trained HSINet model
        data: HSI cube (H, W, C)
        device: torch device
        batch_size: batch size for inference
        
    Returns:
        predictions: (H, W) array with predicted class indices (0-14)
    """
    H, W, C = data.shape
    total_pixels = H * W
    
    print(f"\nRunning inference on {total_pixels:,} pixels...")
    print(f"  - Image size: {H} x {W}")
    print(f"  - Batch size: {batch_size}")
    
    # Pre-allocate predictions array
    predictions = np.zeros((H, W), dtype=np.int64)
    
    # Generate all (row, col) coordinates
    all_coords = [(r, c) for r in range(H) for c in range(W)]
    
    # Process in batches
    num_batches = (total_pixels + batch_size - 1) // batch_size
    
    with torch.no_grad():
        for batch_idx in tqdm(range(num_batches), desc="Predicting"):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, total_pixels)
            
            batch_coords = all_coords[start_idx:end_idx]
            
            # Extract patches for this batch
            batch_patches = []
            for row, col in batch_coords:
                patch = extract_patch(data, row, col, PATCH_SIZE)
                batch_patches.append(patch)
            
            # Stack into tensor: (batch, C, H, W)
            patches_tensor = torch.FloatTensor(np.array(batch_patches)).to(device)
            
            # Forward pass
            logits, _, _ = model(patches_tensor, None, None)
            
            # Get predictions
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            
            # Store predictions
            for i, (row, col) in enumerate(batch_coords):
                predictions[row, col] = preds[i]
    
    return predictions

# ============================================
# Save Outputs
# ============================================

def save_outputs(predictions, output_dir):
    """
    Save predictions to NPZ, CSV, and TXT files.
    
    Args:
        predictions: (H, W) array with predicted class indices
        output_dir: directory to save outputs
    """
    os.makedirs(output_dir, exist_ok=True)
    
    H, W = predictions.shape
    
    # 1. Save NPZ file
    npz_path = os.path.join(output_dir, 'Houston_all_pixel_predictions.npz')
    np.savez(npz_path, 
             predictions=predictions,
             class_names=CLASS_NAMES)
    print(f"\nSaved NPZ file: {npz_path}")
    print(f"  - predictions shape: {predictions.shape}")
    
    # 2. Save CSV file
    csv_path = os.path.join(output_dir, 'Houston_all_pixel_predictions.csv')
    print(f"\nGenerating CSV file...")
    
    with open(csv_path, 'w') as f:
        f.write('row,col,predicted_class_index,predicted_class_name\n')
        for row in tqdm(range(H), desc="Writing CSV"):
            for col in range(W):
                class_idx = predictions[row, col]
                class_name = CLASS_NAMES[class_idx]
                f.write(f'{row},{col},{class_idx},{class_name}\n')
    
    print(f"Saved CSV file: {csv_path}")
    print(f"  - Total rows: {H * W:,}")
    
    # 3. Save class mapping file
    txt_path = os.path.join(output_dir, 'class_index_mapping.txt')
    with open(txt_path, 'w') as f:
        f.write("Houston 2013 Dataset - Class Index to Name Mapping\n")
        f.write("=" * 50 + "\n\n")
        for idx, name in CLASS_NAMES.items():
            f.write(f"Class {idx:2d}: {name}\n")
    
    print(f"Saved class mapping: {txt_path}")
    
    return npz_path, csv_path, txt_path

# ============================================
# Print Summary Statistics
# ============================================

def print_summary(predictions):
    """Print per-class pixel counts"""
    print("\n" + "=" * 60)
    print("PREDICTION SUMMARY")
    print("=" * 60)
    
    total_pixels = predictions.size
    
    print(f"\nPer-class pixel distribution:")
    print(f"{'Class':5s} {'Name':20s} {'Count':>12s} {'Percentage':>10s}")
    print("-" * 50)
    
    for class_idx in range(15):
        count = np.sum(predictions == class_idx)
        percentage = (count / total_pixels) * 100
        class_name = CLASS_NAMES[class_idx]
        print(f"{class_idx:5d} {class_name:20s} {count:12,d} {percentage:9.2f}%")
    
    print("-" * 50)
    print(f"{'Total':5s} {'':20s} {total_pixels:12,d} {100:9.2f}%")
    print()

# ============================================
# Main
# ============================================

def main():
    print("=" * 60)
    print("GGCL Houston - All-Pixel Inference")
    print("=" * 60)
    
    # Set seed
    set_seed(SEED)
    print(f"\nSeed: {SEED}")
    
    # Check paths
    if not os.path.exists(DATA_PATH):
        print(f"\nERROR: Data file not found at {DATA_PATH}")
        print("Please update DATA_PATH or ensure the dataset is available.")
        sys.exit(1)
    
    if not os.path.exists(MODEL_PATH):
        print(f"\nERROR: Model file not found at {MODEL_PATH}")
        print("Please run 'python run_kaggle.py' first to train and save the 5-shot model.")
        sys.exit(1)
    
    # Get device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    
    # Load data
    print("\n" + "-" * 60)
    data, shape_info = load_houston_data(DATA_PATH)
    
    # Load model
    print("\n" + "-" * 60)
    model = load_model(MODEL_PATH, device)
    
    # Run inference
    print("\n" + "-" * 60)
    predictions = run_inference(model, data, device, batch_size=BATCH_SIZE)
    
    # Save outputs
    print("\n" + "-" * 60)
    npz_path, csv_path, txt_path = save_outputs(predictions, OUTPUT_DIR)
    
    # Print summary
    print_summary(predictions)
    
    print("=" * 60)
    print("INFERENCE COMPLETE!")
    print("=" * 60)
    print(f"\nOutput files:")
    print(f"  1. {npz_path}")
    print(f"  2. {csv_path}")
    print(f"  3. {txt_path}")
    print()
    
    return predictions

if __name__ == '__main__':
    main()
