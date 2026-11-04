#!/usr/bin/env python3
"""
GGCL Cross-Scene - Standalone Kaggle Script
Run this script directly in Kaggle:
    python run_kaggle.py

CROSS-SCENE EXPERIMENTS:
- Pre-training: ONCE on entire Dioni (contrastive learning, no labels)
- Experiment 1 (Zero-shot): Few-shots from Dioni, test on Loukia test set
- Experiment 2 (Few-shot): Few-shots from Loukia train, test on Loukia test set

DATASET:
- Dioni (source): 250x1376, 176 bands, 12 classes, 20,024 labeled pixels
- Loukia (target): 249x945, 176 bands, 12 classes, 10,317 labeled pixels
- Split Loukia: 80% Train (~8,254), 20% Test (~2,063)
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
    DATA_DIR = '/kaggle/input/hyrank-cross-scene'
if not os.path.exists(DATA_DIR):
    for root, dirs, files in os.walk('/kaggle/input'):
        if 'Dioni.mat' in files or 'dioni.mat' in files:
            DATA_DIR = root
            break

DIONI_PATH = os.path.join(DATA_DIR, 'Dioni.mat')
DIONI_GT_PATH = os.path.join(DATA_DIR, 'Dioni_gt_out68.mat')
LOUKIA_PATH = os.path.join(DATA_DIR, 'Loukia.mat')
LOUKIA_GT_PATH = os.path.join(DATA_DIR, 'Loukia_gt_out68.mat')

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
    'num_classes': 12,  # 12 classes for HyRank
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

# Try to import scipy.io (required for MATLAB files)
try:
    import scipy.io as sio
except ImportError:
    print("Installing scipy...")
    os.system('pip install scipy -q')
    import scipy.io as sio

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

def pretrain_phase(dioni_data, dioni_gt, device):
    """
    Pre-train model on Dioni dataset (unsupervised).
    Uses all valid labeled pixels but treats them as unlabeled.
    
    Args:
        dioni_data: Dioni HSI data
        dioni_gt: Dioni ground truth (used for filtering valid pixels)
        device: torch device
        
    Returns:
        pretrained_state_dict: weights of pre-trained model
    """
    print(f"\n{'='*60}")
    print(f"PRE-TRAINING PHASE")
    print(f"Contrastive Learning on Dioni (Valid samples, No labels)")
    print(f"{'='*60}")
    
    set_seed(CONFIG['seed'])
    
    # Create unsupervised loader
    pretrain_loader = create_pretrain_dataloader(
        dioni_data, 
        gt=dioni_gt,
        patch_size=CONFIG['patch_size'],
        batch_size=CONFIG['batch_size'],
        num_workers=2
    )
    
    # Setup params
    params = {
        'data': {
            'num_classes': CONFIG['num_classes'],
            'spectral_size': dioni_data.shape[2], 
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
        
    save_path = os.path.join(RESULTS_PATH, 'pretrained_dioni_model.pth')
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
        print("Loading pre-trained weights from Dioni...")
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
    print("Cross-Scene Dataset (HyRank: Dioni / Loukia)")
    print("=" * 60)
    
    # Check if data files exist
    for path in [DIONI_PATH, DIONI_GT_PATH, LOUKIA_PATH, LOUKIA_GT_PATH]:
        if not os.path.exists(path):
            print(f"\nERROR: Data file not found at {path}")
            print("Please update DATA_DIR in this script.")
            sys.exit(1)
    
    set_seed(CONFIG['seed'])
    device = get_device(0 if torch.cuda.is_available() else -1)
    
    # Load data
    dioni_data, dioni_gt, loukia_data, loukia_gt, loukia_train_gt, loukia_test_gt, info = load_cross_scene_data(
        DIONI_PATH, DIONI_GT_PATH, LOUKIA_PATH, LOUKIA_GT_PATH,
        test_ratio=CONFIG['test_ratio'],
        seed=CONFIG['seed']
    )
    
    # 1. PRE-TRAIN ON DIONI (Entire dataset, Unlabeled)
    pretrained_weights = pretrain_phase(dioni_data, dioni_gt, device)
    
    all_results = {'exp1_zeroshot': {}, 'exp2_fewshot': {}}
    
    # 2. EXPERIMENT 1: Zero-shot (Dioni -> Loukia)
    print("\n" + "=" * 60)
    print("EXPERIMENT 1: ZERO-SHOT TRANSFER (Dioni -> Loukia)")
    print("Fine-tune on Dioni few-shots, Test on Loukia")
    print("=" * 60)
    
    for n_shot in SHOT_SETTINGS:
        result = run_experiment(
            exp_name='exp1_zeroshot',
            n_shot=n_shot,
            labeled_data=dioni_data,
            labeled_gt=dioni_gt,
            test_data=loukia_data,
            test_gt=loukia_test_gt,
            unlabeled_data=dioni_data,
            info=info,
            device=device,
            pretrained_weights=pretrained_weights 
        )
        all_results['exp1_zeroshot'][f'{n_shot}-shot'] = result
    
    # 3. EXPERIMENT 2: Few-shot (Loukia -> Loukia)
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: FEW-SHOT TRANSFER (Loukia -> Loukia)")
    print("Fine-tune on Loukia few-shots, Test on Loukia")
    print("=" * 60)
    
    for n_shot in SHOT_SETTINGS:
        result = run_experiment(
            exp_name='exp2_fewshot',
            n_shot=n_shot,
            labeled_data=loukia_data,
            labeled_gt=loukia_train_gt,
            test_data=loukia_data,
            test_gt=loukia_test_gt,
            unlabeled_data=loukia_data,
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
    
    print("\n--- EXPERIMENT 1: Zero-Shot (Dioni -> Loukia) ---")
    print(f"{'Shot':<10} {'OA':<12} {'AA':<12} {'Kappa':<12} {'F1':<12}")
    print("-" * 60)
    for n_shot in SHOT_SETTINGS:
        if f'{n_shot}-shot' in all_results['exp1_zeroshot']:
            r = all_results['exp1_zeroshot'][f'{n_shot}-shot']
            print(f"{n_shot:<10} {r['OA']:.2f}%      {r['AA']:.2f}%      {r['Kappa']:.2f}      {r['F1']:.2f}%")
    
    print("\n--- EXPERIMENT 2: Few-Shot (Loukia train -> Loukia test) ---")
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
