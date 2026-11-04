#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SSLSM Feature Extraction — HyRank Cross-Scene (Dioni → Loukia)

1. Pre-trains SSLSM (ViT encoder + Masked Spectral Reconstruction decoder) on Dioni.
2. Extracts 1024-dimensional feature representations for:
   - Dioni (Source)            -> data/Dioni_feature.h5
   - Loukia Train (Target)      -> data/Loukia_feature_train.h5
   - Loukia Test (Target)       -> data/Loukia_feature_test.h5
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

from net import vit, sslsm
from dataloader import load_cross_scene_data, SSLSMDataset, set_seed

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(SCRIPT_DIR, 'data')
RESULTS_DIR = os.path.join(SCRIPT_DIR, 'results')
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def resolve_dataset_file(fname, custom_dir=None, fallback_rel=None):
    """
    Locates dataset file with auto-unzip support across Kaggle and local directories.
    """
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

    if os.path.isdir('/kaggle/input'):
        for root, _, files in os.walk('/kaggle/input'):
            for f in files:
                if f.lower() == base:
                    found = os.path.join(root, f)
                    print(f"Auto-detected in Kaggle input: {found}")
                    return found

    if os.path.isdir('/kaggle/working'):
        for root, _, files in os.walk('/kaggle/working'):
            for f in files:
                if f.lower() == base:
                    return os.path.join(root, f)

    for root, _, files in os.walk('.'):
        for f in files:
            if f.lower() == base:
                return os.path.join(root, f)

    print("\n" + "=" * 70)
    print(f"❌ ERROR: Dataset file '{fname}' could not be found anywhere!")
    print("=" * 70)
    print("Searched locations:")
    print(f"  - Local data directory: {DATA_DIR}")
    print(f"  - Working directory:    {os.getcwd()}")
    print("\nHOW TO FIX:")
    print("  Option 1 (Add Kaggle Dataset):")
    print("    In Kaggle notebook, click '+ Add Input' -> Upload 'HyRank-data.zip'")
    print("  Option 2 (Unzip inside notebook):")
    print("    !unzip -q /path/to/HyRank-data.zip -d /kaggle/working/SSLSM-Open-Set/hyrank/data/")
    print("  Option 3 (Specify custom path):")
    print("    python run.py --data_dir /path/to/folder_with_mat_files")
    print("=" * 70 + "\n")
    sys.exit(1)


def extract_features(model, x_data, y_data, batch_size=64):
    """
    Extract 1024-dimensional feature representations from trained SSLSM.
    """
    model.eval()
    model.masking_ratio = 0.00001
    
    loader = DataLoader(SSLSMDataset(x_data, y_data), batch_size=batch_size, shuffle=False, num_workers=0)
    feats = []
    labs  = []
    
    with torch.no_grad():
        for bx, by in tqdm(loader, desc="Extracting 1024-dim representations"):
            bx = bx.to(device)
            spe_img = model.to_spe_patch(bx)
            batch, n, _ = spe_img.shape
            spe_num_masked = int(model.masking_ratio * n)
            
            spe_rand_indices = torch.rand(batch, n, device=device).argsort(dim=-1)
            spe_img_pos = spe_img + model.encoder.pos_embedding[:, 1:(n + 1)]
            spe_masked_indices, spe_unmasked_indices = spe_rand_indices[:, :spe_num_masked], spe_rand_indices[:, spe_num_masked:]
            spe_batch_range = torch.arange(batch, device=device)[:, None]
            spe_tokens = spe_img_pos[spe_batch_range, spe_unmasked_indices]
            
            spe_encoded_tokens = model.encoder.transformer(spe_tokens)
            decoder_tokens = model.enc_to_dec(spe_encoded_tokens)
            decoder_tokens = decoder_tokens + model.spe_decoder_pos_emb(spe_unmasked_indices)
            decoded_tokens = model.decoder(decoder_tokens)
            
            # Pass through first 6 layers of fc:
            # Flatten -> BatchNorm1d -> Dropout(0) -> Linear(7680, 1024) -> ReLU -> BatchNorm1d(1024)
            h = model.fc[:6](decoded_tokens)
            feats.append(h.cpu().numpy())
            labs.append(by.numpy())
            
    features   = np.concatenate(feats, axis=0).astype(np.float32)
    labels_arr = np.concatenate(labs, axis=0).astype(np.int64)
    return features, labels_arr


def save_h5(filepath, features, labels):
    with h5py.File(filepath, 'w') as f:
        f.create_dataset('data', data=features)
        f.create_dataset('label', data=labels)
    print(f"Saved: {filepath} | shape: {features.shape}, labels: {labels.shape}")


def main():
    parser = argparse.ArgumentParser(description="SSLSM HyRank Cross-Scene Feature Extraction")
    parser.add_argument('--data_dir', type=str, default=None, help='Custom directory containing dataset .mat files')
    parser.add_argument('--pretrain_epochs', type=int, default=50, help='Pre-training epochs on Dioni (default: 50)')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training/extraction')
    parser.add_argument('--patch_size', type=int, default=15, help='Spatial patch size (default: 15)')
    parser.add_argument('--pca_components', type=int, default=15, help='Number of PCA components (default: 15)')
    parser.add_argument('--force', action='store_true', help='Force re-extraction even if files exist')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    set_seed(args.seed)

    dioni_feat_path     = os.path.join(DATA_DIR, 'Dioni_feature.h5')
    loukia_tr_feat_path = os.path.join(DATA_DIR, 'Loukia_feature_train.h5')
    loukia_te_feat_path = os.path.join(DATA_DIR, 'Loukia_feature_test.h5')

    if not args.force and all(os.path.isfile(p) for p in [dioni_feat_path, loukia_tr_feat_path, loukia_te_feat_path]):
        print("\n" + "=" * 70)
        print("  All HyRank Cross-Scene feature files already exist on disk:")
        print(f"   - {dioni_feat_path}")
        print(f"   - {loukia_tr_feat_path}")
        print(f"   - {loukia_te_feat_path}")
        print("  Skipping extraction. (Pass --force to re-generate)")
        print("=" * 70 + "\n")
        return

    # Resolve raw data files
    dioni_path     = resolve_dataset_file('Dioni.mat', custom_dir=args.data_dir, fallback_rel='../../GGCL-hyrank/data/Dioni.mat')
    dioni_gt_path  = resolve_dataset_file('Dioni_gt_out68.mat', custom_dir=args.data_dir, fallback_rel='../../GGCL-hyrank/data/Dioni_gt_out68.mat')
    loukia_path    = resolve_dataset_file('Loukia.mat', custom_dir=args.data_dir, fallback_rel='../../GGCL-hyrank/data/Loukia.mat')
    loukia_gt_path = resolve_dataset_file('Loukia_gt_out68.mat', custom_dir=args.data_dir, fallback_rel='../../GGCL-hyrank/data/Loukia_gt_out68.mat')

    print(f"Dioni Data:  {dioni_path}")
    print(f"Dioni GT:    {dioni_gt_path}")
    print(f"Loukia Data: {loukia_path}")
    print(f"Loukia GT:   {loukia_gt_path}")

    # Load and preprocess data
    X_dioni, Y_dioni, X_loukia_tr, Y_loukia_tr, X_loukia_te, Y_loukia_te = load_cross_scene_data(
        dioni_path, dioni_gt_path, loukia_path, loukia_gt_path,
        pca_components=args.pca_components, patch_size=args.patch_size,
        test_ratio=0.2, seed=args.seed
    )

    num_classes = 12
    dim = args.patch_size ** 2

    # Initialize ViT and SSLSM
    v = vit.ViT(
        image_size=9,
        patch_size=3,
        num_classes=num_classes,
        dim=dim,
        depth=8,
        heads=8,
        mlp_dim=2048,
        bands=args.pca_components,
    )
    model = sslsm.SSLSM(
        encoder=v,
        num_classes=num_classes,
        masking_ratio=0.5,
        decoder_dim=512,
        decoder_depth=2,
        bands=args.pca_components,
        dim=dim,
    ).to(device)

    # Check for existing pretrained weights
    weight_candidates = [
        os.path.join(RESULTS_DIR, 'pretrained_dioni_sslsm.pth'),
        os.path.join('/kaggle/working', 'pretrained_dioni_sslsm.pth'),
    ]
    loaded_weights = False
    for wc in weight_candidates:
        if os.path.isfile(wc):
            print(f"Loading existing pre-trained weights from: {wc}")
            model.load_state_dict(torch.load(wc, map_location=device), strict=False)
            loaded_weights = True
            break

    if not loaded_weights:
        print(f"\nPre-training SSLSM on Dioni for {args.pretrain_epochs} epochs (Masked Reconstruction)...")
        pretrain_loader = DataLoader(
            SSLSMDataset(X_dioni, np.zeros(len(X_dioni))),
            batch_size=args.batch_size, shuffle=True, num_workers=0
        )
        opt = torch.optim.Adam(model.parameters(), lr=3e-4)
        model.train()
        model.masking_ratio = 0.5
        
        t0 = time.time()
        for ep in range(args.pretrain_epochs):
            total_loss = 0.0
            for bx, _ in pretrain_loader:
                bx = bx.to(device)
                loss, _ = model(bx)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total_loss += loss.item() * bx.size(0)
            avg_loss = total_loss / len(X_dioni)
            if (ep + 1) % 10 == 0 or ep == args.pretrain_epochs - 1:
                print(f"  Pretrain Epoch [{ep+1}/{args.pretrain_epochs}] — Reconstruction MSE: {avg_loss:.4f} ({time.time()-t0:.1f}s)")
                
        save_path = os.path.join(RESULTS_DIR, 'pretrained_dioni_sslsm.pth')
        torch.save(model.state_dict(), save_path)
        print(f"Pretrained SSLSM model saved to: {save_path}")

    # =========================================================================
    # Extract Features
    # =========================================================================
    print("\n" + "=" * 70)
    print("  Extracting 1024-dim SSLSM Features")
    print("=" * 70)

    # 1. Dioni (Source pool)
    print(f"Extracting Dioni ({len(X_dioni)} samples)...")
    f_dioni, l_dioni = extract_features(model, X_dioni, Y_dioni, batch_size=args.batch_size)
    save_h5(dioni_feat_path, f_dioni, l_dioni)

    # 2. Loukia Train (Adaptation pool)
    print(f"Extracting Loukia Train ({len(X_loukia_tr)} samples)...")
    f_loukia_tr, l_loukia_tr = extract_features(model, X_loukia_tr, Y_loukia_tr, batch_size=args.batch_size)
    save_h5(loukia_tr_feat_path, f_loukia_tr, l_loukia_tr)

    # 3. Loukia Test (Evaluation set)
    print(f"Extracting Loukia Test ({len(X_loukia_te)} samples)...")
    f_loukia_te, l_loukia_te = extract_features(model, X_loukia_te, Y_loukia_te, batch_size=args.batch_size)
    save_h5(loukia_te_feat_path, f_loukia_te, l_loukia_te)

    print("\n✅ All HyRank Cross-Scene SSLSM features extracted successfully!")


if __name__ == '__main__':
    main()
