# -*- coding: utf-8 -*-
"""
DMVL Training for Trento Dataset
Adapted from Houston implementation

Changes:
- Trento dataset with 80/20 split
- seed=42 for reproducibility
- epochs=50
- Pretrain on TR only (no data leakage)
"""

import argparse
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
# thop removed - not essential, causes import errors on some systems
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


# train for one epoch to learn unique features
def train(net, data_loader, train_optimizer):
    net.train()
    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)
    for pos_1, pos_2, target in train_bar:
        #print(pos_1.shape, pos_2.shape, target.shape) # torch.Size([128, 3, 28, 28]) torch.Size([128, 3, 28, 28]) torch.Size([128])
        pos_1, pos_2 = pos_1.cuda(non_blocking=True), pos_2.cuda(non_blocking=True)
        feature_1, out_1 = net(pos_1)
        feature_2, out_2 = net(pos_2)
        # [2*B, D]
        out = torch.cat([out_1, out_2], dim=0)
        # [2*B, 2*B]
        sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
        mask = (torch.ones_like(sim_matrix) - torch.eye(2 * batch_size, device=sim_matrix.device)).bool()
        # [2*B, 2*B-1]
        sim_matrix = sim_matrix.masked_select(mask).view(2 * batch_size, -1)

        # compute loss
        pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
        # [2*B]
        pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
        loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()
        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()

        total_num += batch_size
        total_loss += loss.item() * batch_size
        train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f}'.format(epoch, epochs, total_loss / total_num))

    return total_loss / total_num



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Deep Multiview Learning for HSI classification')
    parser.add_argument('--feature_dim', default=128, type=int, help='Feature dim for latent vector')
    parser.add_argument('--temperature', default=0.5, type=float, help='Temperature used in softmax')
    parser.add_argument('--k', default=200, type=int, help='Top k most similar images used to predict the label')
    parser.add_argument('--batch_size', default=128, type=int, help='Number of images in each mini-batch')
    parser.add_argument('--epochs', default=50, type=int, help='Number of sweeps over the dataset to train')

    # args parse
    args = parser.parse_args()
    feature_dim, temperature, k = args.feature_dim, args.temperature, args.k
    batch_size, epochs = args.batch_size, args.epochs
    
    # Load TRAINING data only for pretraining (no data leakage)
    f = h5py.File('data/Trento28-28-6_train.h5', 'r')
    data = f['data'][:]
    label = f['label'][:]
    f.close()

    print(f"Training data: {data.shape}, labels: {label.shape}")
    print(f"Label range: {label.min()} to {label.max()}, unique classes: {len(np.unique(label))}")
    
    # data prepare
    train_data = utils.HsiDataset(data, label, transform=utils.train_transform)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True,
                              drop_last=True)

    # model setup and optimizer config
    model = Model(feature_dim).cuda()
    print(f'# Model loaded with feature_dim={feature_dim}')
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
    c = train_data.classes

    # training loop
    results = {'train_loss': []}
    save_name_pre = '{}_{}_{}_{}_{}_trento'.format(feature_dim, temperature, k, batch_size, epochs)
    if not os.path.exists('results'):
        os.mkdir('results')

    # Training loop - ENABLED for Trento
    print(f"\n=== Starting Pretraining ===")
    print(f"Epochs: {epochs}, Batch size: {batch_size}")
    for epoch in range(1, epochs + 1):
        train_loss = train(model, train_loader, optimizer)
        results['train_loss'].append(train_loss)
    
    # Save trained model
    model_save_path = 'results/{}_model.pth'.format(save_name_pre)
    torch.save(model.state_dict(), model_save_path)
    print(f"\nModel saved to {model_save_path}")

    # Feature extraction function
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

    # Extract features for TRAINING set
    print(f"\n=== Extracting Training Features ===")
    f = h5py.File('data/Trento28-28-6_train.h5', 'r')
    train_data = f['data'][:]
    train_label = f['label'][:]
    f.close()
    
    train_features, train_labels = extract_features(model, train_data, train_label, batch_size)
    print(f"Training features shape: {train_features.shape}")
    print(f"Training labels shape: {train_labels.shape}")
    
    f = h5py.File('data/Trento_feature_train.h5', 'w')
    f['data'] = train_features
    f['label'] = train_labels
    f.close()
    print(f"Saved training features to data/Trento_feature_train.h5")

    # Extract features for TEST set
    print(f"\n=== Extracting Test Features ===")
    f = h5py.File('data/Trento28-28-6_test.h5', 'r')
    test_data = f['data'][:]
    test_label = f['label'][:]
    f.close()
    
    test_features, test_labels = extract_features(model, test_data, test_label, batch_size)
    print(f"Test features shape: {test_features.shape}")
    print(f"Test labels shape: {test_labels.shape}")
    
    f = h5py.File('data/Trento_feature_test.h5', 'w')
    f['data'] = test_features
    f['label'] = test_labels
    f.close()
    print(f"Saved test features to data/Trento_feature_test.h5")
    
    print(f"\n=== Training Complete ===")