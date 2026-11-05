#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SpectralDINO Feature Extraction — HyRank Cross-Scene (Dioni → Loukia)

1. Pre-trains SpectralDINO (Dual-MoS LoRA ViT) on Dioni source domain.
2. Extracts 384-dimensional normalized feature representations for:
   - Dioni (Source)            -> data/Dioni_feature.h5
   - Loukia Train (Target)     -> data/Loukia_feature_train.h5
   - Loukia Test (Target)      -> data/Loukia_feature_test.h5
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
        os.path.normpath(os.path.join(SCRIPT_DIR, '..', '..', '..', 'MorphJEPA_DMVL', 'DMVL', 'DMVL-hyrank', 'data', fname)),
        os.path.normpath(os.path.join(SCRIPT_DIR, '..', '..', '..', 'MorphJEPA_GGCL', 'GGCL', 'GGCL-hyrank', 'data', fname)),
        os.path.join('/kaggle/input', fname),
        os.path.join('/kaggle/working', fname),
    ]
    for c in workspace_candidates:
        if os.path.isfile(c):
            return c
            
    # Auto-extract from zip if present
    zip_candidates = [
        os.path.join(DATA_DIR, 'HyRank-data.zip'),
        os.path.normpath(os.path.join(SCRIPT_DIR, '..', '..', '..', 'MorphJEPA_DMVL', 'DMVL', 'HyRank-data.zip')),
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
    parser = argparse.ArgumentParser(description="SpectralDINO HyRank Cross-Scene Feature Extraction")
    parser.add_argument('--pretrain_epochs', type=int, default=50, help='Pre-training epochs on Dioni')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training/extraction')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--patch_size', type=int, default=15, help='Spatial patch size')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--force', action='store_true', help='Force re-extraction even if files exist')
    parser.add_argument('--data_dir', type=str, default=None, help='Custom data directory')
    args = parser.parse_args()

    set_seed(args.seed)

    print("\n" + "=" * 70)
    print("  SpectralDINO HyRank Cross-Scene Feature Extraction")
    print("  Dioni (Source) -> Loukia (Target)")
    print(f"  Device: {device}")
    print("=" * 70)

    dioni_feat_path     = os.path.join(DATA_DIR, 'Dioni_feature.h5')
    loukia_tr_feat_path = os.path.join(DATA_DIR, 'Loukia_feature_train.h5')
    loukia_te_feat_path = os.path.join(DATA_DIR, 'Loukia_feature_test.h5')

    if not args.force and all(os.path.isfile(p) for p in [dioni_feat_path, loukia_tr_feat_path, loukia_te_feat_path]):
        print("\n" + "=" * 70)
        print("  All feature files already exist:")
        print(f"    - {dioni_feat_path}")
        print(f"    - {loukia_tr_feat_path}")
        print(f"    - {loukia_te_feat_path}")
        print("  Skipping extraction. (Pass --force to re-generate)")
        print("=" * 70 + "\n")
        return

    # Resolve data paths
    dioni_path    = resolve_dataset_file('Dioni.mat', custom_dir=args.data_dir, fallback_rel='../../../../MorphJEPA_DMVL/DMVL/DMVL-hyrank/data/Dioni.mat')
    dioni_gt_path = resolve_dataset_file('Dioni_gt_out68.mat', custom_dir=args.data_dir, fallback_rel='../../../../MorphJEPA_DMVL/DMVL/DMVL-hyrank/data/Dioni_gt_out68.mat')
    if not os.path.isfile(dioni_gt_path):
        dioni_gt_path = resolve_dataset_file('Dioni_gt.mat', custom_dir=args.data_dir, fallback_rel='../../../../MorphJEPA_DMVL/DMVL/DMVL-hyrank/data/Dioni_gt.mat')

    loukia_path    = resolve_dataset_file('Loukia.mat', custom_dir=args.data_dir, fallback_rel='../../../../MorphJEPA_DMVL/DMVL/DMVL-hyrank/data/Loukia.mat')
    loukia_gt_path = resolve_dataset_file('Loukia_gt_out68.mat', custom_dir=args.data_dir, fallback_rel='../../../../MorphJEPA_DMVL/DMVL/DMVL-hyrank/data/Loukia_gt_out68.mat')
    if not os.path.isfile(loukia_gt_path):
        loukia_gt_path = resolve_dataset_file('Loukia_gt.mat', custom_dir=args.data_dir, fallback_rel='../../../../MorphJEPA_DMVL/DMVL/DMVL-hyrank/data/Loukia_gt.mat')

    print(f"Dioni Data:  {dioni_path}")
    print(f"Dioni GT:    {dioni_gt_path}")
    print(f"Loukia Data: {loukia_path}")
    print(f"Loukia GT:   {loukia_gt_path}")

    # Load and preprocess data
    X_dioni, Y_dioni, X_loukia_tr, Y_loukia_tr, X_loukia_te, Y_loukia_te = load_cross_scene_data(
        dioni_path, dioni_gt_path, loukia_path, loukia_gt_path,
        patch_size=args.patch_size, test_ratio=0.2, seed=args.seed
    )

    in_bands = X_dioni.shape[1] # 176
    num_classes = 12

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

    # Pre-train / train on Dioni source domain
    weights_path = os.path.join(RESULTS_DIR, 'pretrained_dioni_spectraldino.pth')
    if os.path.isfile(weights_path) and not args.force:
        print(f"Loading existing weights from: {weights_path}")
        model.load_state_dict(torch.load(weights_path, map_location=device), strict=False)
    else:
        print(f"\nTraining SpectralDINO on Dioni for {args.pretrain_epochs} epochs...")
        train_loader = DataLoader(
            SpectralDataset(X_dioni, Y_dioni),
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

    print(f"Extracting Dioni ({len(X_dioni)} samples)...")
    f_dioni, l_dioni = extract_features(model, X_dioni, Y_dioni, batch_size=args.batch_size)
    save_h5(dioni_feat_path, f_dioni, l_dioni)

    print(f"Extracting Loukia Train ({len(X_loukia_tr)} samples)...")
    f_loukia_tr, l_loukia_tr = extract_features(model, X_loukia_tr, Y_loukia_tr, batch_size=args.batch_size)
    save_h5(loukia_tr_feat_path, f_loukia_tr, l_loukia_tr)

    print(f"Extracting Loukia Test ({len(X_loukia_te)} samples)...")
    f_loukia_te, l_loukia_te = extract_features(model, X_loukia_te, Y_loukia_te, batch_size=args.batch_size)
    save_h5(loukia_te_feat_path, f_loukia_te, l_loukia_te)

    print("\n✅ All HyRank Cross-Scene SpectralDINO features extracted successfully!")


if __name__ == '__main__':
    main()
