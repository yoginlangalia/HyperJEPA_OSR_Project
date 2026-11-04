"""
GGCL Main Script for Houston Dataset
Guided Group Contrastive Learning for Hyperspectral Image Classification

Usage:
    python main.py --data_path ./data/houston.mat --n_shot 10
    python main.py --data_path ./data/houston.mat --n_shot 5
    python main.py --data_path ./data/houston.mat --n_shot 1
"""

import os
import sys
import argparse
import time
import numpy as np
import torch
import json

from utils import set_seed, get_device
from dataloader import load_houston_data, create_dataloaders
from trainer import GGCLTrainer


def parse_args():
    parser = argparse.ArgumentParser(description='GGCL for Houston Dataset')
    
    # Data arguments
    parser.add_argument('--data_path', type=str, default='./data/houston.mat',
                        help='Path to houston.mat file')
    parser.add_argument('--save_path', type=str, default='./results',
                        help='Path to save results')
    parser.add_argument('--n_components', type=int, default=None,
                        help='Number of PCA components (default: None = no PCA)')
    
    # Model arguments
    parser.add_argument('--patch_size', type=int, default=13,
                        help='Spatial patch size (default: 13)')
    parser.add_argument('--depth', type=int, default=1,
                        help='Transformer depth')
    parser.add_argument('--heads', type=int, default=8,
                        help='Number of attention heads')
    parser.add_argument('--dim', type=int, default=64,
                        help='Model dimension')
    parser.add_argument('--mlp_dim', type=int, default=8,
                        help='MLP dimension')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs (default: 100)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-3,
                        help='Weight decay')
    parser.add_argument('--temperature', type=float, default=0.5,
                        help='Temperature for contrastive loss')
    parser.add_argument('--lambda_contra', type=float, default=0.1,
                        help='Weight for contrastive loss')
    
    # Few-shot arguments
    parser.add_argument('--n_shot', type=int, default=10,
                        choices=[1, 5, 10],
                        help='Number of labeled samples per class (default: 10)')
    
    # Contrastive learning arguments
    parser.add_argument('--use_unlabel', action='store_true', default=True,
                        help='Use unlabeled data for contrastive learning')
    parser.add_argument('--no_unlabel', action='store_true',
                        help='Disable unlabeled data')
    
    # Other arguments
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID (-1 for CPU)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of data loading workers')
    parser.add_argument('--eval_interval', type=int, default=10,
                        help='Evaluation interval (epochs)')
    
    args = parser.parse_args()
    
    if args.no_unlabel:
        args.use_unlabel = False
    
    return args


def main():
    args = parse_args()
    
    print("=" * 60)
    print("GGCL: Guided Group Contrastive Learning")
    print("Houston Dataset Classification")
    print("=" * 60)
    
    # Set seed for reproducibility
    set_seed(args.seed)
    print(f"\nRandom seed: {args.seed}")
    
    # Get device
    device = get_device(args.gpu)
    
    # Create save directory
    os.makedirs(args.save_path, exist_ok=True)
    
    # Load data with PCA
    print("\n[1/4] Loading Houston dataset...")
    data, train_gt, test_gt, info = load_houston_data(
        args.data_path, 
        normalize=True,
        n_components=args.n_components
    )
    
    # Create data loaders with few-shot sampling
    print(f"\n[2/4] Creating data loaders ({args.n_shot}-shot)...")
    labeled_loader, test_loader, unlabeled_loader = create_dataloaders(
        data, train_gt, test_gt,
        patch_size=args.patch_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_unlabeled=args.use_unlabel,
        n_shot=args.n_shot,
        seed=args.seed
    )
    
    # Setup parameters
    params = {
        'data': {
            'num_classes': info['num_classes'],
            'spectral_size': info['spectral_size'],  # Now 15 after PCA
            'patch_size': args.patch_size,
        },
        'net': {
            'depth': args.depth,
            'heads': args.heads,
            'dim': args.dim,
            'mlp_dim': args.mlp_dim,
            'dropout': args.dropout,
        },
        'train': {
            'epochs': args.epochs,
            'lr': args.lr,
            'weight_decay': args.weight_decay,
            'temp': args.temperature,
            'lambda_contra': args.lambda_contra,
            'eval_interval': args.eval_interval,
            'use_unlabel': args.use_unlabel,
        }
    }
    
    # Print configuration
    print("\n[3/4] Configuration:")
    print(f"  Dataset: Houston")
    print(f"  Classes: {info['num_classes']}")
    print(f"  Spectral bands (after PCA): {info['spectral_size']}")
    print(f"  Patch size: {args.patch_size}x{args.patch_size}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Lambda contrastive: {args.lambda_contra}")
    print(f"  Use unlabeled: {args.use_unlabel}")
    print(f"  Few-shot: {args.n_shot}-shot")
    
    # Create trainer
    print("\n[4/4] Training...")
    trainer = GGCLTrainer(params, device=device)
    
    # Train
    start_time = time.time()
    
    model_save_path = os.path.join(args.save_path, f'best_model_{args.n_shot}shot.pth')
    results = trainer.train(
        labeled_loader, test_loader, unlabeled_loader,
        save_path=model_save_path
    )
    
    training_time = time.time() - start_time
    
    # Print final results
    print("\n" + "=" * 60)
    print(f"TRAINING COMPLETE ({args.n_shot}-shot)")
    print("=" * 60)
    print(f"Total training time: {training_time/60:.2f} minutes")
    print(f"\nFinal Results:")
    print(f"  Overall Accuracy (OA): {results['OA']:.2f}%")
    print(f"  Average Accuracy (AA): {results['AA']:.2f}%")
    print(f"  Kappa Coefficient: {results['Kappa']:.2f}")
    print(f"  F1 Score (Macro): {results['F1']:.2f}%")
    
    # Print per-class accuracy
    print(f"\nPer-class Accuracy:")
    for i, (label, acc) in enumerate(zip(info['label_values'], results['per_class'])):
        print(f"  {i+1:2d}. {label}: {acc:.2f}%")
    
    # Save results
    results_dict = {
        'n_shot': args.n_shot,
        'OA': float(results['OA']),
        'AA': float(results['AA']),
        'Kappa': float(results['Kappa']),
        'F1': float(results['F1']),
        'per_class': results['per_class'].tolist(),
        'training_time': training_time,
        'config': vars(args)
    }
    
    results_file = os.path.join(args.save_path, f'results_{args.n_shot}shot.json')
    with open(results_file, 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    print(f"\nResults saved to {results_file}")
    print(f"Model saved to {model_save_path}")
    
    return results


if __name__ == '__main__':
    main()
