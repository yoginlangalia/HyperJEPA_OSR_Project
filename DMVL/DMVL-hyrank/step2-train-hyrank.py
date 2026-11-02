# -*- coding: utf-8 -*-
"""
DMVL Training for HyRank Cross-Scene Dataset
Pre-train on Dioni, extract features for all sets

Outputs:
- Dioni_feature.h5 (for Exp1 few-shot)
- Loukia_feature_train.h5 (for Exp2 few-shot)
- Loukia_feature_test.h5 (for evaluation)
"""

import argparse
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import h5py
import utils
from model import Model

# Set seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def train(net, data_loader, train_optimizer, epoch, epochs, batch_size, temperature):
    net.train()
    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)
    for pos_1, pos_2, target in train_bar:
        pos_1, pos_2 = pos_1.cuda(non_blocking=True), pos_2.cuda(non_blocking=True)
        feature_1, out_1 = net(pos_1)
        feature_2, out_2 = net(pos_2)
        out = torch.cat([out_1, out_2], dim=0)
        sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
        mask = (torch.ones_like(sim_matrix) - torch.eye(2 * batch_size, device=sim_matrix.device)).bool()
        sim_matrix = sim_matrix.masked_select(mask).view(2 * batch_size, -1)

        pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
        pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
        loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()
        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()

        total_num += batch_size
        total_loss += loss.item() * batch_size
        train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f}'.format(epoch, epochs, total_loss / total_num))

    return total_loss / total_num


def extract_features(model, data, label, batch_size):
    """Extract features from trained model"""
    dataset = utils.HsiDataset_test(data, label, transform=utils.test_transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    
    model.eval()
    feature_bank = []
    with torch.no_grad():
        for data_batch, _, target in loader:
            feature, out = model(data_batch.cuda(non_blocking=True))
            feature_bank.append(out)
    
    feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
    feature_labels = torch.tensor(loader.dataset.label, device=feature_bank.device)
    x = feature_bank.cpu().numpy().T
    y = feature_labels.cpu().numpy()
    return x, y


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DMVL HyRank Training')
    parser.add_argument('--feature_dim', default=128, type=int, help='Feature dim for latent vector')
    parser.add_argument('--temperature', default=0.5, type=float, help='Temperature used in softmax')
    parser.add_argument('--k', default=200, type=int, help='Top k most similar images')
    parser.add_argument('--batch_size', default=128, type=int, help='Number of images in each mini-batch')
    parser.add_argument('--epochs', default=50, type=int, help='Number of sweeps over the dataset to train')

    args = parser.parse_args()
    feature_dim, temperature, k = args.feature_dim, args.temperature, args.k
    batch_size, epochs = args.batch_size, args.epochs
    
    # ===========================================
    # Load Dioni for PRETRAINING
    # ===========================================
    print("=== Loading Dioni for Pretraining ===")
    f = h5py.File('data/Dioni_pretrain.h5', 'r')
    pretrain_data = f['data'][:]
    pretrain_label = f['label'][:]
    f.close()

    print(f"Pretrain data: {pretrain_data.shape}, labels: {pretrain_label.shape}")
    print(f"Label range: {pretrain_label.min()} to {pretrain_label.max()}, unique classes: {len(np.unique(pretrain_label))}")
    
    # Data preparation
    train_data = utils.HsiDataset(pretrain_data, pretrain_label, transform=utils.train_transform)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True,
                              drop_last=True)

    # Model setup
    model = Model(feature_dim).cuda()
    print(f'# Model loaded with feature_dim={feature_dim}')
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)

    # Training
    results = {'train_loss': []}
    save_name_pre = '{}_{}_{}_{}_{}_hyrank'.format(feature_dim, temperature, k, batch_size, epochs)
    if not os.path.exists('results'):
        os.mkdir('results')

    print(f"\n=== Starting Pretraining on Dioni ===")
    print(f"Epochs: {epochs}, Batch size: {batch_size}")
    for epoch in range(1, epochs + 1):
        train_loss = train(model, train_loader, optimizer, epoch, epochs, batch_size, temperature)
        results['train_loss'].append(train_loss)
    
    # Save trained model
    model_save_path = 'results/{}_model.pth'.format(save_name_pre)
    torch.save(model.state_dict(), model_save_path)
    print(f"\nModel saved to {model_save_path}")

    # ===========================================
    # Extract Features for ALL sets
    # ===========================================
    
    # 1. Dioni features (for Exp1 zero-shot few-shot)
    print(f"\n=== Extracting Dioni Features ===")
    f = h5py.File('data/Dioni_pretrain.h5', 'r')
    dioni_data = f['data'][:]
    dioni_label = f['label'][:]
    f.close()
    
    dioni_features, dioni_labels = extract_features(model, dioni_data, dioni_label, batch_size)
    print(f"Dioni features: {dioni_features.shape}")
    
    f = h5py.File('data/Dioni_feature.h5', 'w')
    f['data'] = dioni_features
    f['label'] = dioni_labels
    f.close()
    print(f"Saved: data/Dioni_feature.h5")

    # 2. Loukia TRAIN features (for Exp2 few-shot)
    print(f"\n=== Extracting Loukia Train Features ===")
    f = h5py.File('data/Loukia_train.h5', 'r')
    loukia_train_data = f['data'][:]
    loukia_train_label = f['label'][:]
    f.close()
    
    loukia_train_features, loukia_train_labels = extract_features(model, loukia_train_data, loukia_train_label, batch_size)
    print(f"Loukia train features: {loukia_train_features.shape}")
    
    f = h5py.File('data/Loukia_feature_train.h5', 'w')
    f['data'] = loukia_train_features
    f['label'] = loukia_train_labels
    f.close()
    print(f"Saved: data/Loukia_feature_train.h5")

    # 3. Loukia TEST features (for evaluation)
    print(f"\n=== Extracting Loukia Test Features ===")
    f = h5py.File('data/Loukia_test.h5', 'r')
    loukia_test_data = f['data'][:]
    loukia_test_label = f['label'][:]
    f.close()
    
    loukia_test_features, loukia_test_labels = extract_features(model, loukia_test_data, loukia_test_label, batch_size)
    print(f"Loukia test features: {loukia_test_features.shape}")
    
    f = h5py.File('data/Loukia_feature_test.h5', 'w')
    f['data'] = loukia_test_features
    f['label'] = loukia_test_labels
    f.close()
    print(f"Saved: data/Loukia_feature_test.h5")
    
    print(f"\n=== Training Complete ===")
    print(f"Pretrained on: Dioni ({len(dioni_labels)} samples)")
    print(f"Features extracted for:")
    print(f"  - Dioni: {len(dioni_labels)} (Exp1 few-shot source)")
    print(f"  - Loukia Train: {len(loukia_train_labels)} (Exp2 few-shot source)")
    print(f"  - Loukia Test: {len(loukia_test_labels)} (evaluation)")