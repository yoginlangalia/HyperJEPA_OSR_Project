#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GGCL Feature Extraction — Houston Cross-Scene (Houston 2013 → 2018)

1. Pre-trains HSINet (ViT encoder) on Houston-13 using Guided Contrastive Learning.
2. Extracts 64-dimensional [CLS] representations (logit_pixel) for:
   - Houston-13 (Source)       -> data/Houston13_feature.h5
   - Houston-18 Train (Target)  -> data/Houston18_feature_train.h5
   - Houston-18 Test (Target)   -> data/Houston18_feature_test.h5
"""

import argparse
import os
import sys
import time
import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import HSINet
from dataloader import (
    load_cross_scene_data,
    create_pretrain_dataloader,
    extract_patch,
    CrossSceneDataset,
    get_test_indices,
    set_seed
)
from trainer import GGCLTrainer

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(SCRIPT_DIR, 'data')
RESULTS_DIR = os.path.join(SCRIPT_DIR, 'results')
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def resolve_dataset_file(fname, custom_dir=None, fallback_rel=None):
    """
    Intelligently locate dataset file across:
    1. custom_dir if specified
    2. Local DATA_DIR
    3. Auto-unzip any .zip files in /kaggle/input or /kaggle/working containing fname
    4. SCRIPT_DIR and parents
    5. /kaggle/input and subdirectories
    6. /kaggle/working and subdirectories
    7. Current working directory recursively
    """
    base = os.path.basename(fname).lower()

    # 1. Check custom_dir
    if custom_dir and os.path.isdir(custom_dir):
        for f in os.listdir(custom_dir):
            if f.lower() == base:
                return os.path.join(custom_dir, f)

    # 2. Check local DATA_DIR
    if os.path.isdir(DATA_DIR):
        for f in os.listdir(DATA_DIR):
            if f.lower() == base:
                return os.path.join(DATA_DIR, f)

    # 3. Check direct path or fallback_rel
    if os.path.isfile(fname):
        return os.path.abspath(fname)
    if fallback_rel:
        p_fb = os.path.normpath(os.path.join(SCRIPT_DIR, fallback_rel))
        if os.path.isfile(p_fb):
            return p_fb

    # 4. Check for zip files that might contain the dataset and auto-extract them!
    search_dirs = [
        DATA_DIR,
        '/kaggle/working',
        '/kaggle/input',
        '.',
        os.path.normpath(os.path.join(SCRIPT_DIR, '..', '..')),
        os.path.normpath(os.path.join(SCRIPT_DIR, '..', '..', '..'))
    ]
    for sdir in search_dirs:
        if not os.path.isdir(sdir):
            continue
        for root, _, files in os.walk(sdir):
            for f in files:
                if f.lower().endswith('.zip'):
                    zpath = os.path.join(root, f)
                    try:
                        import zipfile
                        with zipfile.ZipFile(zpath, 'r') as zf:
                            z_names = [zn.lower() for zn in zf.namelist()]
                            if any(base == os.path.basename(zn).lower() for zn in z_names):
                                print(f"⚡ Found '{fname}' inside '{zpath}'! Auto-extracting to {DATA_DIR}...")
                                zf.extractall(DATA_DIR)
                                for extracted in os.listdir(DATA_DIR):
                                    if extracted.lower() == base:
                                        return os.path.join(DATA_DIR, extracted)
                    except Exception:
                        pass

    # 5. Search /kaggle/input recursively
    if os.path.isdir('/kaggle/input'):
        for root, _, files in os.walk('/kaggle/input'):
            for f in files:
                if f.lower() == base:
                    found = os.path.join(root, f)
                    print(f"Auto-detected in Kaggle input: {found}")
                    return found

    # 6. Search /kaggle/working recursively
    if os.path.isdir('/kaggle/working'):
        for root, _, files in os.walk('/kaggle/working'):
            for f in files:
                if f.lower() == base:
                    return os.path.join(root, f)

    # 7. Search current workspace recursively
    for root, _, files in os.walk('.'):
        for f in files:
            if f.lower() == base:
                return os.path.join(root, f)

    # If still not found, print detailed diagnostic and exit cleanly
    print("\n" + "=" * 70)
    print(f"❌ ERROR: Required dataset file '{fname}' could not be found anywhere!")
    print("=" * 70)
    print("Searched in:")
    print(f"  - Local data directory: {DATA_DIR}")
    print(f"  - Working directory:    {os.getcwd()}")
    if os.path.isdir('/kaggle/input'):
        print("\nItems in /kaggle/input:")
        count = 0
        for r, d, fl in os.walk('/kaggle/input'):
            for f in fl:
                print(f"   • {os.path.join(r, f)}")
                count += 1
                if count >= 15:
                    break
            if count >= 15:
                print("   ... (truncated)")
                break
        if count == 0:
            print("   (Empty: no datasets attached to this Kaggle notebook)")
    if os.path.isdir('/kaggle/working'):
        print("\nItems in /kaggle/working:")
        count = 0
        for r, d, fl in os.walk('/kaggle/working'):
            for f in fl:
                print(f"   • {os.path.join(r, f)}")
                count += 1
                if count >= 15:
                    break
            if count >= 15:
                print("   ... (truncated)")
                break

    print("\nHOW TO FIX:")
    print("  Option 1 (Add as Kaggle Dataset):")
    print("    In the Kaggle notebook sidebar, click '+ Add Input' -> Upload 'Houston-cross-scene-data.zip'")
    print("  Option 2 (Unzip inside notebook):")
    print("    !unzip -q /path/to/Houston-cross-scene-data.zip -d /kaggle/working/GGCL-Open-Set/cross-scene/data/")
    print("  Option 3 (Specify custom path):")
    print("    python run.py --data_dir /path/to/folder_with_mat_files")
    print("=" * 70 + "\n")
    sys.exit(1)


def extract_features_from_dataset(model, data, indices, labels, patch_size=15, batch_size=128):
    """Pass patches through HSINet encoder to extract 64-dim [CLS] token features."""
    dataset = CrossSceneDataset(data, indices, labels, patch_size=patch_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    model.eval()
    feats = []
    labs = []
    with torch.no_grad():
        for batch_x, batch_y in tqdm(loader, desc="Extracting features"):
            batch_x = batch_x.to(device)
            # encoder_block returns (logit_pixel, reduce_x)
            # logit_pixel is the 64-dim [CLS] token representation passed to mlp_head
            logit_pixel, _ = model.encoder_block(batch_x)
            feats.append(logit_pixel.cpu().numpy())
            labs.append(batch_y.numpy())
            
    features = np.concatenate(feats, axis=0).astype(np.float32)
    labels_arr = np.concatenate(labs, axis=0).astype(np.int64)
    return features, labels_arr


def save_h5(filepath, features, labels):
    """Save features and labels to HDF5."""
    with h5py.File(filepath, 'w') as f:
        f.create_dataset('data', data=features)
        f.create_dataset('label', data=labels)
    print(f"Saved: {filepath} | shape: {features.shape}, labels: {labels.shape}")


def main():
    parser = argparse.ArgumentParser(description="GGCL Houston Cross-Scene Feature Extraction")
    parser.add_argument('--data_dir', type=str, default=None, help='Custom directory containing dataset .mat files')
    parser.add_argument('--pretrain_epochs', type=int, default=50, help='Pre-training epochs on H13 (default: 50)')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training/extraction')
    parser.add_argument('--patch_size', type=int, default=15, help='Spatial patch size (default: 15)')
    parser.add_argument('--force', action='store_true', help='Force re-extraction even if files exist')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    set_seed(args.seed)

    h13_feat_path    = os.path.join(DATA_DIR, 'Houston13_feature.h5')
    h18_tr_feat_path = os.path.join(DATA_DIR, 'Houston18_feature_train.h5')
    h18_te_feat_path = os.path.join(DATA_DIR, 'Houston18_feature_test.h5')

    if not args.force and all(os.path.isfile(p) for p in [h13_feat_path, h18_tr_feat_path, h18_te_feat_path]):
        print("\n" + "=" * 70)
        print("  All Houston Cross-Scene feature files already exist on disk:")
        print(f"   - {h13_feat_path}")
        print(f"   - {h18_tr_feat_path}")
        print(f"   - {h18_te_feat_path}")
        print("  Skipping extraction. (Pass --force to re-generate)")
        print("=" * 70 + "\n")
        return

    # Resolve raw data files
    h13_path    = resolve_dataset_file('Houston13.mat', custom_dir=args.data_dir, fallback_rel='../../GGCL-cross-scene-new/data/Houston13.mat')
    h13_gt_path = resolve_dataset_file('Houston13_7gt.mat', custom_dir=args.data_dir, fallback_rel='../../GGCL-cross-scene-new/data/Houston13_7gt.mat')
    h18_path    = resolve_dataset_file('Houston18.mat', custom_dir=args.data_dir, fallback_rel='../../GGCL-cross-scene-new/data/Houston18.mat')
    h18_gt_path = resolve_dataset_file('Houston18_7gt.mat', custom_dir=args.data_dir, fallback_rel='../../GGCL-cross-scene-new/data/Houston18_7gt.mat')

    print(f"H13 Data: {h13_path}")
    print(f"H13 GT:   {h13_gt_path}")
    print(f"H18 Data: {h18_path}")
    print(f"H18 GT:   {h18_gt_path}")

    # Load Cross-Scene Data (Z-score normalized, 80/20 target split)
    h13_data, h13_gt, h18_data, h18_gt, h18_train_gt, h18_test_gt, info = load_cross_scene_data(
        h13_path, h13_gt_path, h18_path, h18_gt_path,
        test_ratio=0.2, seed=args.seed
    )

    spectral_size = info['spectral_size']
    num_classes   = info['num_classes']

    # Model configuration matching GGCL paper & repo
    params = {
        'data': {
            'num_classes': num_classes,
            'spectral_size': spectral_size,
            'patch_size': args.patch_size,
        },
        'net': {
            'depth': 1,
            'heads': 8,
            'dim': 64,
            'mlp_dim': 8,
            'dropout': 0,
        },
        'train': {
            'epochs': args.pretrain_epochs,
            'lr': 0.001,
            'weight_decay': 5e-3,
            'temp': 0.5,
            'lambda_contra': 1.0,
            'eval_interval': 10,
            'use_unlabel': True,
            'fixed_weight': True,
        }
    }

    # Initialize HSINet
    trainer = GGCLTrainer(params, device=device)

    # Check for existing pretrained weights
    weight_candidates = [
        os.path.join(RESULTS_DIR, 'pretrained_h13_model.pth'),
        os.path.join(SCRIPT_DIR, '..', '..', 'GGCL-cross-scene-new', 'results', 'pretrained_h13_model.pth'),
        os.path.join('/kaggle/working/results', 'pretrained_h13_model.pth'),
    ]
    loaded_weights = False
    for wc in weight_candidates:
        if os.path.isfile(wc):
            print(f"Loading existing pre-trained weights from: {wc}")
            trainer.net.load_state_dict(torch.load(wc, map_location=device), strict=False)
            loaded_weights = True
            break

    if not loaded_weights:
        print(f"\nPre-training HSINet on Houston-13 for {args.pretrain_epochs} epochs...")
        pretrain_loader = create_pretrain_dataloader(
            h13_data, gt=h13_gt, patch_size=args.patch_size,
            batch_size=args.batch_size, num_workers=0
        )
        trainer.net.train()
        for ep in range(args.pretrain_epochs):
            loss = trainer.train_epoch(labeled_loader=pretrain_loader, unlabeled_loader=None, epoch=ep)
            if (ep + 1) % 10 == 0 or ep == args.pretrain_epochs - 1:
                print(f"  Pretrain Epoch [{ep+1}/{args.pretrain_epochs}] — Contrastive Loss: {loss:.4f}")
        
        save_path = os.path.join(RESULTS_DIR, 'pretrained_h13_model.pth')
        torch.save(trainer.net.state_dict(), save_path)
        print(f"Pretrained model saved to: {save_path}")

    # =========================================================================
    # Extract Features
    # =========================================================================
    print("\n" + "=" * 70)
    print("  Extracting 64-dim GGCL Features")
    print("=" * 70)

    # 1. Houston-13 (all valid pixels)
    h13_r, h13_c = np.where(h13_gt > 0)
    h13_indices = list(zip(h13_r, h13_c))
    h13_labels  = [int(h13_gt[r, c] - 1) for r, c in h13_indices]
    print(f"Extracting Houston-13 ({len(h13_indices)} samples)...")
    f_h13, l_h13 = extract_features_from_dataset(trainer.net, h13_data, h13_indices, h13_labels,
                                                 patch_size=args.patch_size, batch_size=args.batch_size * 2)
    save_h5(h13_feat_path, f_h13, l_h13)

    # 2. Houston-18 Train (80% train split)
    h18_tr_r, h18_tr_c = np.where(h18_train_gt > 0)
    h18_tr_indices = list(zip(h18_tr_r, h18_tr_c))
    h18_tr_labels  = [int(h18_train_gt[r, c] - 1) for r, c in h18_tr_indices]
    print(f"Extracting Houston-18 Train ({len(h18_tr_indices)} samples)...")
    f_h18_tr, l_h18_tr = extract_features_from_dataset(trainer.net, h18_data, h18_tr_indices, h18_tr_labels,
                                                       patch_size=args.patch_size, batch_size=args.batch_size * 2)
    save_h5(h18_tr_feat_path, f_h18_tr, l_h18_tr)

    # 3. Houston-18 Test (20% test split)
    h18_te_r, h18_te_c = np.where(h18_test_gt > 0)
    h18_te_indices = list(zip(h18_te_r, h18_te_c))
    h18_te_labels  = [int(h18_test_gt[r, c] - 1) for r, c in h18_te_indices]
    print(f"Extracting Houston-18 Test ({len(h18_te_indices)} samples)...")
    f_h18_te, l_h18_te = extract_features_from_dataset(trainer.net, h18_data, h18_te_indices, h18_te_labels,
                                                       patch_size=args.patch_size, batch_size=args.batch_size * 2)
    save_h5(h18_te_feat_path, f_h18_te, l_h18_te)

    print("\n✅ All Houston Cross-Scene features extracted successfully!")


if __name__ == '__main__':
    main()
