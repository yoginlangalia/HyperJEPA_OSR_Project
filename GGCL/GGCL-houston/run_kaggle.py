#!/usr/bin/env python3
"""
GGCL Houston - Standalone Kaggle Script
Run this script directly in Kaggle:
    python run_kaggle.py

This will run all three few-shot experiments (1-shot, 5-shot, 10-shot)
and save the results.

FIXES APPLIED:
1. PCA: 144 bands → 15 components
2. Zero-padding patch extraction (uses ALL labeled pixels)
3. drop_last=False for training (critical for few-shot)
4. Odd-even spectral augmentation (HSI-appropriate)
5. Fixed AA calculation (includes all classes)
"""

import os
import sys
import time
import json
import numpy as np

# ============================================
# Configuration
# ============================================

# For Kaggle: adjust these paths based on your dataset names
DATA_PATH = './data/houston.mat'
if not os.path.exists(DATA_PATH):
    DATA_PATH = '/kaggle/input/houston-dataset/houston.mat'
if not os.path.exists(DATA_PATH):
    DATA_PATH = '/kaggle/input/houston/houston.mat'
if not os.path.exists(DATA_PATH):
    DATA_PATH = '/kaggle/input/houston-dataset/Houston.mat'
if not os.path.exists(DATA_PATH):
    # Try to find it
    for root, dirs, files in os.walk('/kaggle/input'):
        for f in files:
            if f.lower() == 'houston.mat':
                DATA_PATH = os.path.join(root, f)
                break

# Results path
RESULTS_PATH = './results'
if '/kaggle/' in os.getcwd() or os.path.exists('/kaggle/working'):
    RESULTS_PATH = '/kaggle/working/results'

os.makedirs(RESULTS_PATH, exist_ok=True)

# ============================================
# Configuration
# ============================================

CONFIG = {
    'seed': 42,
    
    # Data
    'patch_size': 13,
    'batch_size': 64,
    'n_components': None,  # No PCA - use all 144 bands
    
    # Model
    'depth': 1,
    'heads': 8,
    'dim': 64,
    'mlp_dim': 8,
    'dropout': 0,
    
    # Training
    'epochs': 200,
    'lr': 0.001,
    'weight_decay': 5e-3,
    'temperature': 0.5,
    'lambda_contra': 0.1,
    
    # GGCL specific
    'use_unlabel': True,
    'eval_interval': 10,
}

# Few-shot settings to run
SHOT_SETTINGS = [5]  # Only run 5-shot for inference

# ============================================
# Imports
# ============================================

print("Importing modules...")

try:
    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
except ImportError:
    print("PyTorch not found!")
    sys.exit(1)

# Try to import einops (required for model)
try:
    import einops
except ImportError:
    print("Installing einops...")
    os.system('pip install einops -q')
    import einops

from utils import set_seed, get_device
from dataloader import load_houston_data, create_dataloaders
from trainer import GGCLTrainer

# ============================================
# Main Training Function
# ============================================

def run_experiment(n_shot, data, train_gt, test_gt, info, device):
    """Run a single few-shot experiment"""
    
    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {n_shot}-SHOT")
    print(f"{'='*60}")
    
    # Set seed for reproducibility
    set_seed(CONFIG['seed'])
    
    # Create data loaders with few-shot sampling
    print(f"\nCreating {n_shot}-shot data loaders...")
    labeled_loader, test_loader, unlabeled_loader = create_dataloaders(
        data, train_gt, test_gt,
        patch_size=CONFIG['patch_size'],
        batch_size=CONFIG['batch_size'],
        num_workers=2,
        use_unlabeled=CONFIG['use_unlabel'],
        n_shot=n_shot,
        seed=CONFIG['seed']
    )
    
    # Setup parameters
    params = {
        'data': {
            'num_classes': info['num_classes'],
            'spectral_size': info['spectral_size'],  # 15 after PCA
            'patch_size': CONFIG['patch_size'],
        },
        'net': {
            'depth': CONFIG['depth'],
            'heads': CONFIG['heads'],
            'dim': CONFIG['dim'],
            'mlp_dim': CONFIG['mlp_dim'],
            'dropout': CONFIG['dropout'],
        },
        'train': {
            'epochs': CONFIG['epochs'],
            'lr': CONFIG['lr'],
            'weight_decay': CONFIG['weight_decay'],
            'temp': CONFIG['temperature'],
            'lambda_contra': CONFIG['lambda_contra'],
            'eval_interval': CONFIG['eval_interval'],
            'use_unlabel': CONFIG['use_unlabel'],
        }
    }
    
    # Create trainer
    print("\nInitializing GGCL trainer...")
    trainer = GGCLTrainer(params, device=device)
    
    # Train
    print("\nStarting training...")
    start_time = time.time()
    
    model_save_path = os.path.join(RESULTS_PATH, f'best_model_{n_shot}shot.pth')
    results = trainer.train(
        labeled_loader, test_loader, unlabeled_loader,
        save_path=model_save_path
    )
    
    training_time = time.time() - start_time
    
    # Save results
    results_dict = {
        'n_shot': n_shot,
        'OA': float(results['OA']),
        'AA': float(results['AA']),
        'Kappa': float(results['Kappa']),
        'F1': float(results['F1']),
        'per_class': results['per_class'].tolist(),
        'training_time': training_time,
        'config': CONFIG
    }
    
    results_file = os.path.join(RESULTS_PATH, f'results_{n_shot}shot.json')
    with open(results_file, 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    print(f"\nResults saved to {results_file}")
    print(f"Model saved to {model_save_path}")
    
    return results_dict


def main():
    """Main function to run all experiments"""
    
    print("=" * 60)
    print("GGCL: Guided Group Contrastive Learning")
    print("Houston Dataset - Few-Shot Classification")
    print("=" * 60)
    
    # Print configuration
    print(f"\nConfiguration:")
    print(f"  Data path: {DATA_PATH}")
    print(f"  Results path: {RESULTS_PATH}")
    print(f"  Seed: {CONFIG['seed']}")
    print(f"  Patch size: {CONFIG['patch_size']}x{CONFIG['patch_size']}")
    print(f"  PCA components: {CONFIG['n_components']}")
    print(f"  Epochs: {CONFIG['epochs']}")
    print(f"  Shot settings: {SHOT_SETTINGS}")
    
    # Check if data file exists
    if not os.path.exists(DATA_PATH):
        print(f"\nERROR: Data file not found at {DATA_PATH}")
        print("Please update DATA_PATH in this script to point to houston.mat")
        sys.exit(1)
    
    # Set seed
    set_seed(CONFIG['seed'])
    
    # Get device
    device = get_device(0 if torch.cuda.is_available() else -1)
    
    # Load data once (with PCA)
    print("\n" + "=" * 60)
    print("LOADING DATA")
    print("=" * 60)
    data, train_gt, test_gt, info = load_houston_data(
        DATA_PATH, 
        normalize=True,
        n_components=CONFIG['n_components']
    )
    
    # Run all experiments
    all_results = {}
    
    for n_shot in SHOT_SETTINGS:
        result = run_experiment(n_shot, data, train_gt, test_gt, info, device)
        all_results[f'{n_shot}-shot'] = result
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY OF ALL EXPERIMENTS")
    print("=" * 60)
    print(f"\n{'Shot':<10} {'OA':<12} {'AA':<12} {'Kappa':<12} {'F1':<12}")
    print("-" * 60)
    
    for n_shot in SHOT_SETTINGS:
        r = all_results[f'{n_shot}-shot']
        print(f"{n_shot:<10} {r['OA']:.2f}%      {r['AA']:.2f}%      {r['Kappa']:.2f}      {r['F1']:.2f}%")
    
    # Save combined results
    combined_file = os.path.join(RESULTS_PATH, 'all_results.json')
    with open(combined_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nAll results saved to {combined_file}")
    
    # Print per-class accuracy for best (10-shot)
    print("\n" + "=" * 60)
    print("PER-CLASS ACCURACY (10-shot)")
    print("=" * 60)
    for i, (label, acc) in enumerate(zip(info['label_values'], all_results['10-shot']['per_class'])):
        print(f"  {i+1:2d}. {label:<20}: {acc:.2f}%")
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    
    return all_results


if __name__ == '__main__':
    main()
