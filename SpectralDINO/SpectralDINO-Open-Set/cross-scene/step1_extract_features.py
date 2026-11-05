#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SpectralDINO Feature Extraction — Houston Cross-Scene (Houston 2013 → 2018)

1. Pre-trains SpectralDINO (Dual-MoS LoRA ViT) on Houston-13 source domain.
2. Extracts 384-dimensional normalized feature representations for:
   - Houston-13 (Source)        -> data/Houston13_feature.h5
   - Houston-18 Train (Target)  -> data/Houston18_feature_train.h5
   - Houston-18 Test (Target)   -> data/Houston18_feature_test.h5
"""

import argparse
import os
import sys
import time
import zipfile
import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from net.spectraldino import SpectralDINO
from dataloader import load_cross_scene_data, SpectralDataset, set_seed

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(SCRIPT_DIR, 'data')
RESULTS_DIR = os.path.join(SCRIPT_DIR, 'results')
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def resolve_dataset_file(fname, custom_dir=None, fallback_rel=None):
    """Locate dataset file supporting fallback paths and auto-unzip."""
    base = os.path.basename(fname).lower()
    
    if custom_dir and os.path.isdir(custom_dir):
        for f in os.listdir(custom_dir):
            if f.lower() == base:
                return os.path.join(custom_dir, f)
                
    if os.path.isdir(DATA_DIR):
        for f in os.listdir(DATA_DIR):
            if f.lower() == base:
                return os.path.join(DATA_DIR, f)
                
    if os.path.isfile(fname):
        return os.path.abspath(fname)
        
    if fallback_rel:
        p_fb = os.path.normpath(os.path.join(SCRIPT_DIR, fallback_rel))
        if os.path.isfile(p_fb):
            return p_fb
            
    # Search common workspace locations
    workspace_candidates = [
        os.path.normpath(os.path.join(SCRIPT_DIR, '..', '..', '..', 'MorphJEPA_DMVL', 'DMVL', 'DMVL-cross-scene', 'data', fname)),
        os.path.normpath(os.path.join(SCRIPT_DIR, '..', '..', '..', 'MorphJEPA_SSLSM', 'SSLSM', 'SSLSM-crossscene', 'data', fname)),
        os.path.normpath(os.path.join(SCRIPT_DIR, '..', '..', '..', 'MorphJEPA_GGCL', 'GGCL', 'GGCL-cross-scene-new', 'data', fname)),
        os.path.join('/kaggle/input', fname),
        os.path.join('/kaggle/working', fname),
    ]
    for c in workspace_candidates:
        if os.path.isfile(c):
            return c
            
    # Auto-extract from zip if present
    zip_candidates = [
        os.path.join(DATA_DIR, 'Houston-cross-scene-data.zip'),
        os.path.normpath(os.path.join(SCRIPT_DIR, '..', '..', '..', 'MorphJEPA_DMVL', 'DMVL', 'Houston-cross-scene-data.zip')),
    ]
    for zpath in zip_candidates:
        if os.path.isfile(zpath):
            try:
                print(f"Extracting {zpath} -> {DATA_DIR}...")
                with zipfile.ZipFile(zpath, 'r') as zf:
                    zf.extractall(DATA_DIR)
                extracted = os.path.join(DATA_DIR, fname)
                if os.path.isfile(extracted):
                    return extracted
            except Exception:
                pass
                
    return os.path.join(DATA_DIR, fname)


def extract_features(model, data, labels, batch_size=64):
    """Extract 384-dimensional L2-normalized features using SpectralDINO backbone."""
    model.eval()
    dataset = SpectralDataset(data, labels)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    all_feats = []
    all_labels = []
    
    with torch.no_grad():
        for bx, by in tqdm(loader, desc="Extracting", leave=False):
            bx = bx.to(device)
            feat = model.extract_features(bx)
            all_feats.append(feat.cpu().numpy())
            all_labels.append(by.numpy())
            
    feats = np.concatenate(all_feats, axis=0)
    labs  = np.concatenate(all_labels, axis=0)
    return feats, labs


def save_h5(filepath, data, labels):
    """Save features and labels to HDF5 format."""
    print(f"Saving {data.shape} to {filepath}...")
    with h5py.File(filepath, 'w') as f:
        f.create_dataset('data', data=data, compression='gzip')
        f.create_dataset('label', data=labels, compression='gzip')
    print(f"Saved: {filepath} ({os.path.getsize(filepath) / 1024:.1f} KB)")


def main():
    parser = argparse.ArgumentParser(description="SpectralDINO Houston Cross-Scene Feature Extraction")
    parser.add_argument('--pretrain_epochs', type=int, default=50, help='Pre-training epochs on H13')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training/extraction')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--patch_size', type=int, default=15, help='Spatial patch size')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--force', action='store_true', help='Force re-extraction even if files exist')
    parser.add_argument('--data_dir', type=str, default=None, help='Custom data directory')
    args = parser.parse_args()

    set_seed(args.seed)

    print("\n" + "=" * 70)
    print("  SpectralDINO Houston Cross-Scene Feature Extraction")
    print("  Houston 2013 (Source) -> Houston 2018 (Target)")
    print(f"  Device: {device}")
    print("=" * 70)

    h13_feat_path    = os.path.join(DATA_DIR, 'Houston13_feature.h5')
    h18_tr_feat_path = os.path.join(DATA_DIR, 'Houston18_feature_train.h5')
    h18_te_feat_path = os.path.join(DATA_DIR, 'Houston18_feature_test.h5')

    if not args.force and all(os.path.isfile(p) for p in [h13_feat_path, h18_tr_feat_path, h18_te_feat_path]):
        print("\n" + "=" * 70)
        print("  All feature files already exist:")
        print(f"    - {h13_feat_path}")
        print(f"    - {h18_tr_feat_path}")
        print(f"    - {h18_te_feat_path}")
        print("  Skipping extraction. (Pass --force to re-generate)")
        print("=" * 70 + "\n")
        return

    # Resolve data paths
    h13_path    = resolve_dataset_file('Houston13.mat', custom_dir=args.data_dir, fallback_rel='../../../../MorphJEPA_DMVL/DMVL/DMVL-cross-scene/data/Houston13.mat')
    h13_gt_path = resolve_dataset_file('Houston13_7gt.mat', custom_dir=args.data_dir, fallback_rel='../../../../MorphJEPA_DMVL/DMVL/DMVL-cross-scene/data/Houston13_7gt.mat')
    h18_path    = resolve_dataset_file('Houston18.mat', custom_dir=args.data_dir, fallback_rel='../../../../MorphJEPA_DMVL/DMVL/DMVL-cross-scene/data/Houston18.mat')
    h18_gt_path = resolve_dataset_file('Houston18_7gt.mat', custom_dir=args.data_dir, fallback_rel='../../../../MorphJEPA_DMVL/DMVL/DMVL-cross-scene/data/Houston18_7gt.mat')

    print(f"H13 Data: {h13_path}")
    print(f"H13 GT:   {h13_gt_path}")
    print(f"H18 Data: {h18_path}")
    print(f"H18 GT:   {h18_gt_path}")

    # Load and preprocess data
    X13, Y13, X18_tr, Y18_tr, X18_te, Y18_te = load_cross_scene_data(
        h13_path, h13_gt_path, h18_path, h18_gt_path,
        patch_size=args.patch_size, test_ratio=0.2, seed=args.seed
    )

    in_bands = X13.shape[1] # 48
    num_classes = 7

    print(f"\nInitializing SpectralDINO (bands={in_bands}, classes={num_classes}, LoRA rank=4, tokens=16)...")
    model = SpectralDINO(
        in_bands=in_bands,
        num_classes=num_classes,
        embed_dim=384,
        depth=12,
        num_heads=6,
        rank=4,
        num_experts=2,
        num_tokens=16,
        patch_size=args.patch_size,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"SpectralDINO Trainable Parameters: {total_params:,}")

    # Pre-train / train on Houston-13 source domain
    weights_path = os.path.join(RESULTS_DIR, 'pretrained_h13_spectraldino.pth')
    if os.path.isfile(weights_path) and not args.force:
        print(f"Loading existing weights from: {weights_path}")
        model.load_state_dict(torch.load(weights_path, map_location=device), strict=False)
    else:
        print(f"\nTraining SpectralDINO on Houston-13 for {args.pretrain_epochs} epochs...")
        train_loader = DataLoader(
            SpectralDataset(X13, Y13),
            batch_size=args.batch_size, shuffle=True, num_workers=0
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        
        t0 = time.time()
        for ep in range(args.pretrain_epochs):
            model.train()
            total_loss = 0.0
            correct = 0
            total_samples = 0
            
            for bx, by in train_loader:
                bx, by = bx.to(device), by.to(device)
                logits, dino_proj, _ = model(bx, return_features=True)
                loss_ce = criterion(logits, by)
                loss = loss_ce
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item() * bx.size(0)
                preds = logits.argmax(dim=-1)
                correct += (preds == by).sum().item()
                total_samples += bx.size(0)
                
            avg_loss = total_loss / total_samples
            acc = 100.0 * correct / total_samples
            if (ep + 1) % 10 == 0 or ep == args.pretrain_epochs - 1:
                print(f"  Epoch [{ep+1:2d}/{args.pretrain_epochs}] — Loss: {avg_loss:.4f} | Source Acc: {acc:.2f}% ({time.time()-t0:.1f}s)")
                
        torch.save(model.state_dict(), weights_path)
        print(f"Saved SpectralDINO model weights to: {weights_path}")

    # Extract 384-dim features
    print("\n" + "=" * 70)
    print("  Extracting 384-dimensional SpectralDINO Features")
    print("=" * 70)

    print(f"Extracting Houston-13 ({len(X13)} samples)...")
    f_h13, l_h13 = extract_features(model, X13, Y13, batch_size=args.batch_size)
    save_h5(h13_feat_path, f_h13, l_h13)

    print(f"Extracting Houston-18 Train ({len(X18_tr)} samples)...")
    f_h18_tr, l_h18_tr = extract_features(model, X18_tr, Y18_tr, batch_size=args.batch_size)
    save_h5(h18_tr_feat_path, f_h18_tr, l_h18_tr)

    print(f"Extracting Houston-18 Test ({len(X18_te)} samples)...")
    f_h18_te, l_h18_te = extract_features(model, X18_te, Y18_te, batch_size=args.batch_size)
    save_h5(h18_te_feat_path, f_h18_te, l_h18_te)

    print("\n✅ All Houston Cross-Scene SpectralDINO features extracted successfully!")


if __name__ == '__main__':
    main()
