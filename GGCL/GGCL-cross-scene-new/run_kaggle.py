#!/usr/bin/env python3
"""
GGCL Cross-Scene - Standalone Kaggle Script
Run this script directly in Kaggle:
    python run_kaggle.py

CROSS-SCENE EXPERIMENTS:
- Pre-training: ONCE on entire Houston-13 (contrastive learning, no labels)
- Experiment 1 (Zero-shot): Few-shots from H13, test on H18 test set
- Experiment 2 (Few-shot): Few-shots from H18 train, test on H18 test set

DATASET:
- Houston-13: 210x954, 48 bands, 7 classes, 2530 labeled pixels
- Houston-18: 210x954, 48 bands, 7 classes, 53200 labeled pixels
- Split H18: 80% Train (~42560), 20% Test (~10640)
"""

import os
import sys
import time
import json
import numpy as np

# ============================================
# Configuration
# ============================================

# Data paths - adjust for Kaggle
DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    DATA_DIR = '/kaggle/input/cross-scene'
if not os.path.exists(DATA_DIR):
    for root, dirs, files in os.walk('/kaggle/input'):
        if 'Houston13.mat' in files or 'houston13.mat' in files:
            DATA_DIR = root
            break

H13_PATH = os.path.join(DATA_DIR, 'Houston13.mat')
H13_GT_PATH = os.path.join(DATA_DIR, 'Houston13_7gt.mat')
H18_PATH = os.path.join(DATA_DIR, 'Houston18.mat')
H18_GT_PATH = os.path.join(DATA_DIR, 'Houston18_7gt.mat')

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
    'patch_size': 15,
    'batch_size': 64,
    'test_ratio': 0.2,  # 20% of H18 for testing, 80% for training
    
   # Model (same as GGCL-Houston)
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
    
    # Pre-training
    'pretrain_epochs': 100,
    
    # Cross-scene specific
    'num_classes': 7,  # 7 classes (not 15 like Houston benchmark)
}

# Few-shot settings to run
SHOT_SETTINGS = [10, 5, 1]

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

# Try to import h5py (required for HDF5 files)
try:
    import h5py
except ImportError:
    print("Installing h5py...")
    os.system('pip install h5py -q')
    import h5py

from utils import set_seed, get_device
from dataloader import (
    load_cross_scene_data, 
    create_fewshot_dataloaders,
    create_pretrain_dataloader
)
from trainer import GGCLTrainer

# ============================================
# Experiment Functions
# ============================================

def pretrain_phase(h13_data, h13_gt, device):
    """
    Pre-train model on Houston-13 dataset (unsupervised).
    Uses all valid labeled pixels but treats them as unlabeled.
    
    Args:
        h13_data: H13 HSI data
        h13_gt: H13 ground truth (used for filtering valid pixels)
        device: torch device
        
    Returns:
        pretrained_state_dict: weights of pre-trained model
    """
    print(f"\n{'='*60}")
    print(f"PRE-TRAINING PHASE")
    print(f"Contrastive Learning on Houston-13 (Valid samples, No labels)")
    print(f"{'='*60}")
    
    set_seed(CONFIG['seed'])
    
    # Create unsupervised loader
    pretrain_loader = create_pretrain_dataloader(
        h13_data, 
        gt=h13_gt,
        patch_size=CONFIG['patch_size'],
        batch_size=CONFIG['batch_size'],
        num_workers=2
    )
    
    # Setup params
    params = {
        'data': {
            'num_classes': CONFIG['num_classes'],
            'spectral_size': h13_data.shape[2], 
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
            'epochs': CONFIG['pretrain_epochs'],
            'lr': CONFIG['lr'],
            'weight_decay': CONFIG['weight_decay'],
            'temp': CONFIG['temperature'],
            'lambda_contra': 1.0,  # Pure contrastive
            'eval_interval': 10,
            'use_unlabel': True,
            'fixed_weight': True,  # No ramp-up during pre-training
        }
    }
    
    # Initialize trainer
    trainer = GGCLTrainer(params, device=device)
    
    # TRAIN LOOP for Pre-training
    # Pass pretrain_loader as 'labeled_loader'
    # Labels are -1, so CE loss will be 0, only Contrastive Loss
    
    trainer.net.train()
    print("Pre-training...")
    
    for epoch in range(params['train']['epochs']):
        loss_avg = trainer.train_epoch(labeled_loader=pretrain_loader, unlabeled_loader=None, epoch=epoch)
        print(f"Pre-train Epoch {epoch+1}/{params['train']['epochs']} - Loss: {loss_avg:.4f}")
        
    save_path = os.path.join(RESULTS_PATH, 'pretrained_h13_model.pth')
    torch.save(trainer.net.state_dict(), save_path)
    print(f"Pre-trained model saved to {save_path}")
    
    return trainer.net.state_dict()


def run_experiment(exp_name, n_shot, labeled_data, labeled_gt, 
                   test_data, test_gt, unlabeled_data, info, device, pretrained_weights=None):
    """
    Run a single few-shot experiment using pre-trained weights.
    """
    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {exp_name.upper()} - {n_shot}-SHOT")
    print(f"{'='*60}")
    
    set_seed(CONFIG['seed'])
    
    # Create data loaders
    labeled_loader, test_loader, unlabeled_loader = create_fewshot_dataloaders(
        data=labeled_data,
        labeled_gt=labeled_gt,
        test_gt=test_gt,
        unlabeled_data=unlabeled_data,
        patch_size=CONFIG['patch_size'],
        batch_size=CONFIG['batch_size'],
        num_workers=2,
        n_shot=n_shot,
        num_classes=CONFIG['num_classes'],
        seed=CONFIG['seed']
    )
    
    # Setup parameters
    params = {
        'data': {
            'num_classes': CONFIG['num_classes'],
            'spectral_size': info['spectral_size'], 
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
    trainer = GGCLTrainer(params, device=device)
    
    # LOAD PRE-TRAINED WEIGHTS
    if pretrained_weights is not None:
        print("Loading pre-trained weights from Houston-13...")
        trainer.net.load_state_dict(pretrained_weights, strict=False)
        
    # Train (Fine-tuning)
    print("\nStarting Fine-tuning...")
    start_time = time.time()
    
    model_save_path = os.path.join(RESULTS_PATH, f'best_model_{exp_name}_{n_shot}shot.pth')
    
    results = trainer.train(
        labeled_loader, test_loader, unlabeled_loader,
        save_path=model_save_path
    )
    
    training_time = time.time() - start_time
    
    # Save results
    results_dict = {
        'experiment': exp_name,
        'n_shot': n_shot,
        'OA': float(results['OA']),
        'AA': float(results['AA']),
        'Kappa': float(results['Kappa']),
        'F1': float(results['F1']),
        'per_class': results['per_class'].tolist(),
        'training_time': training_time,
        'config': CONFIG
    }
    
    results_file = os.path.join(RESULTS_PATH, f'results_{exp_name}_{n_shot}shot.json')
    with open(results_file, 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    return results_dict


def main():
    """Main function to run all cross-scene experiments."""
    print("=" * 60)
    print("GGCL: Guided Group Contrastive Learning")
    print("Cross-Scene Dataset (Houston 2013 / Houston 2018)")
    print("=" * 60)
    
    # Check if data files exist
    for path in [H13_PATH, H13_GT_PATH, H18_PATH, H18_GT_PATH]:
        if not os.path.exists(path):
            print(f"\nERROR: Data file not found at {path}")
            print("Please update DATA_DIR in this script.")
            sys.exit(1)
    
    set_seed(CONFIG['seed'])
    device = get_device(0 if torch.cuda.is_available() else -1)
    
    # Load data
    h13_data, h13_gt, h18_data, h18_gt, h18_train_gt, h18_test_gt, info = load_cross_scene_data(
        H13_PATH, H13_GT_PATH, H18_PATH, H18_GT_PATH,
        test_ratio=CONFIG['test_ratio'],
        seed=CONFIG['seed']
    )
    
    # 1. PRE-TRAIN ON HOUSTON-13 (Entire dataset, Unlabeled)
    pretrained_weights = pretrain_phase(h13_data, h13_gt, device)
    
    all_results = {'exp1_zeroshot': {}, 'exp2_fewshot': {}}
    
    # 2. EXPERIMENT 1: Zero-shot (H13 -> H18)
    print("\n" + "=" * 60)
    print("EXPERIMENT 1: ZERO-SHOT TRANSFER (H13 -> H18)")
    print("Fine-tune on H13 few-shots, Test on H18")
    print("=" * 60)
    
    for n_shot in SHOT_SETTINGS:
        result = run_experiment(
            exp_name='exp1_zeroshot',
            n_shot=n_shot,
            labeled_data=h13_data,
            labeled_gt=h13_gt,
            test_data=h18_data,
            test_gt=h18_test_gt,
            unlabeled_data=h13_data,
            info=info,
            device=device,
            pretrained_weights=pretrained_weights 
        )
        all_results['exp1_zeroshot'][f'{n_shot}-shot'] = result
    
    # 3. EXPERIMENT 2: Few-shot (H18 -> H18)
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: FEW-SHOT TRANSFER (H18 -> H18)")
    print("Fine-tune on H18 few-shots, Test on H18")
    print("=" * 60)
    
    for n_shot in SHOT_SETTINGS:
        result = run_experiment(
            exp_name='exp2_fewshot',
            n_shot=n_shot,
            labeled_data=h18_data,
            labeled_gt=h18_train_gt,
            test_data=h18_data,
            test_gt=h18_test_gt,
            unlabeled_data=h18_data,
            info=info,
            device=device,
            pretrained_weights=pretrained_weights 
        )
        all_results['exp2_fewshot'][f'{n_shot}-shot'] = result
    
    # ============================================
    # Print Summary
    # ============================================
    print("\n" + "=" * 60)
    print("SUMMARY OF ALL EXPERIMENTS")
    print("=" * 60)
    
    print("\n--- EXPERIMENT 1: Zero-Shot (H13 -> H18) ---")
    print(f"{'Shot':<10} {'OA':<12} {'AA':<12} {'Kappa':<12} {'F1':<12}")
    print("-" * 60)
    for n_shot in SHOT_SETTINGS:
        if f'{n_shot}-shot' in all_results['exp1_zeroshot']:
            r = all_results['exp1_zeroshot'][f'{n_shot}-shot']
            print(f"{n_shot:<10} {r['OA']:.2f}%      {r['AA']:.2f}%      {r['Kappa']:.2f}      {r['F1']:.2f}%")
    
    print("\n--- EXPERIMENT 2: Few-Shot (H18 train -> H18 test) ---")
    print(f"{'Shot':<10} {'OA':<12} {'AA':<12} {'Kappa':<12} {'F1':<12}")
    print("-" * 60)
    for n_shot in SHOT_SETTINGS:
        if f'{n_shot}-shot' in all_results['exp2_fewshot']:
            r = all_results['exp2_fewshot'][f'{n_shot}-shot']
            print(f"{n_shot:<10} {r['OA']:.2f}%      {r['AA']:.2f}%      {r['Kappa']:.2f}      {r['F1']:.2f}%")
    
    # Save combined results
    combined_file = os.path.join(RESULTS_PATH, 'all_results.json')
    with open(combined_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nAll results saved to {combined_file}")
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    
    return all_results


if __name__ == '__main__':
    main()
