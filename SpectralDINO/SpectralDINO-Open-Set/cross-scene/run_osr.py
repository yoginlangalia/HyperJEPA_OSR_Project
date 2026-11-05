#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SpectralDINO Open Set Recognition — Cross-Scene (Houston 2013 → 2018)

Prerequisites:
    Run step1_extract_features.py first (or run.py).
    Features must exist in: ./data/
        - Houston13_feature.h5
        - Houston18_feature_train.h5
        - Houston18_feature_test.h5

Methodology:
    - MSP (Maximum Softmax Probability) based open-set detection
    - N_UNKNOWN=3 classes hidden per split
    - RobustProbe (2-layer MLP) trained on known classes only
    - Evaluated via class-balanced AUROC across 20 splits and 5 seeds
    - Also records and reports Known Accuracy (multi-class accuracy on known classes)
"""

import itertools
import os
import random
import sys
import time

import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score

# =============================================================================
# Configuration
# =============================================================================
NUM_CLASSES    = 7              # 7 classes (0-indexed: 0–6)
OS_SHOTS       = (1, 5, 10)
SEEDS          = [42, 123, 456, 789, 1024]
N_UNKNOWN      = 3
MIN_TEST_PX    = 50             # class must have >= this many H18 test pixels to be an unknown class
ALWAYS_KNOWN   = [3]            # class 4 (0-based index 3) has ~4 test pixels — forced known

SCRIPT_DIR         = os.path.dirname(os.path.abspath(__file__))
LOCAL_FEATURE_DIR  = os.path.join(SCRIPT_DIR, 'data')

FEATURE_DIR = LOCAL_FEATURE_DIR
OUTPUT_DIR  = os.path.join(SCRIPT_DIR, 'CrossScene')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# =============================================================================
# RobustProbe — lightweight MLP for MSP open-set scoring
# =============================================================================
class RobustProbe(nn.Module):
    """2-layer MLP classifier whose max-softmax confidence serves as the
    closed-set score; 1 − max(softmax) is the novelty / unknown score."""
    def __init__(self, in_dim, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        return self.net(x)


# =============================================================================
# Load pre-extracted features
# =============================================================================
print("\n" + "=" * 70)
print("  SpectralDINO Open Set Recognition — Cross-Scene (Houston 2013 → 2018)")
print("=" * 70)
print(f"\nDevice: {device}")
print(f"Loading features from: {FEATURE_DIR}\n")

required_files = [
    'Houston13_feature.h5',
    'Houston18_feature_train.h5',
    'Houston18_feature_test.h5',
]
for fname in required_files:
    fpath = os.path.join(FEATURE_DIR, fname)
    if not os.path.isfile(fpath):
        print(f"ERROR: Required feature file not found: {fpath}")
        print("       Run step1_extract_features.py first.")
        sys.exit(1)

# Houston-13 features (zero-shot support pool)
with h5py.File(os.path.join(FEATURE_DIR, 'Houston13_feature.h5'), 'r') as f:
    h13_data  = f['data'][:]
    h13_label = f['label'][:].astype(int)
print(f"Houston-13 features : {h13_data.shape}, classes: {np.unique(h13_label)}")

# Houston-18 Train features (adaptation support pool)
with h5py.File(os.path.join(FEATURE_DIR, 'Houston18_feature_train.h5'), 'r') as f:
    h18_train_data  = f['data'][:]
    h18_train_label = f['label'][:].astype(int)
print(f"Houston-18 Train    : {h18_train_data.shape}, classes: {np.unique(h18_train_label)}")

# Houston-18 Test features (evaluation — 20% of target)
with h5py.File(os.path.join(FEATURE_DIR, 'Houston18_feature_test.h5'), 'r') as f:
    te_f = f['data'][:]
    te_l = f['label'][:].astype(int)
print(f"Houston-18 Test     : {te_f.shape}, classes: {np.unique(te_l)}")

FEAT_DIM = te_f.shape[1]
print(f"Feature dim         : {FEAT_DIM}")


# =============================================================================
# Task definitions
# =============================================================================
TASKS = {
    "zero-shot (H13 -> H18)": {
        "data":   h13_data,
        "labels": h13_label,
    },
    "adaptation (H18 -> H18)": {
        "data":   h18_train_data,
        "labels": h18_train_label,
    },
}


# =============================================================================
# Determine eligible unknown classes and generate OSR splits
# =============================================================================
n_test = np.bincount(te_l, minlength=NUM_CLASSES)
eligible = [k for k in range(NUM_CLASSES)
            if n_test[k] >= MIN_TEST_PX and k not in ALWAYS_KNOWN]
SPLITS = [
    dict(unknown=list(u),
         known=[k for k in range(NUM_CLASSES) if k not in u])
    for u in itertools.combinations(eligible, N_UNKNOWN)
]

print(f"\nTest pixel counts per class: {dict(enumerate(n_test))}")
print(f"Eligible unknown classes   : {eligible}  (count: {len(eligible)})")
print(f"Always-known class(es)     : {ALWAYS_KNOWN}  (too few test pixels)")
print(f"Number of OSR splits       : {len(SPLITS)}  (C({len(eligible)}, {N_UNKNOWN}))\n")


# =============================================================================
# Helper: sample k-shot support data for known classes
# =============================================================================
def sample_support(pool_f, pool_l, known_classes, k_shot, seed):
    """Draw k_shot random samples per known class from the support pool."""
    rng = np.random.RandomState(seed)
    idx_list = []
    for c in known_classes:
        c_idxs = np.where(pool_l == c)[0]
        if len(c_idxs) < k_shot:
            chosen = rng.choice(c_idxs, size=k_shot, replace=True)
        else:
            chosen = rng.choice(c_idxs, size=k_shot, replace=False)
        idx_list.extend(chosen)
    return pool_f[idx_list], pool_l[idx_list]


# =============================================================================
# Helper: evaluate one split (AUROC + Known Accuracy)
# =============================================================================
def evaluate_split(split, task_name, k_shot, seed):
    """
    Train RobustProbe on k-shot support from known classes.
    Evaluate:
      1. AUROC: binary novel-vs-known discrimination on full test set
      2. Known Acc: multi-class accuracy on known-class test samples
    """
    known   = split['known']
    unknown = split['unknown']
    n_known = len(known)
    c2i = {c: i for i, c in enumerate(known)}

    pool_f = TASKS[task_name]["data"]
    pool_l = TASKS[task_name]["labels"]

    s_x, s_y_raw = sample_support(pool_f, pool_l, known, k_shot, seed)
    s_y = np.array([c2i[y] for y in s_y_raw], dtype=np.int64)

    # Convert to tensors
    tx_s = torch.from_numpy(s_x).float().to(device)
    ty_s = torch.from_numpy(s_y).long().to(device)
    tx_te = torch.from_numpy(te_f).float().to(device)

    # Train probe
    torch.manual_seed(seed)
    probe = RobustProbe(FEAT_DIM, n_known).to(device)
    optimizer = optim.Adam(probe.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    ds = TensorDataset(tx_s, ty_s)
    loader = DataLoader(ds, batch_size=min(32, len(ds)), shuffle=True)

    probe.train()
    for _ in range(100):
        for bx, by in loader:
            optimizer.zero_grad()
            loss = loss_fn(probe(bx), by)
            loss.backward()
            optimizer.step()

    # Inference on test set
    probe.eval()
    with torch.no_grad():
        logits = probe(tx_te)
        probs = F.softmax(logits, dim=1).cpu().numpy()

    # MSP Novelty score: 1 - max(softmax probability)
    msp_novelty = 1.0 - np.max(probs, axis=1)

    # Binary ground truth: 1 if unknown, 0 if known
    is_unknown = np.isin(te_l, unknown).astype(int)

    # Class-balanced AUROC
    class_aurocs = []
    for u_cls in unknown:
        u_mask = (te_l == u_cls)
        for k_cls in known:
            k_mask = (te_l == k_cls)
            pair_mask = u_mask | k_mask
            if np.sum(u_mask) > 0 and np.sum(k_mask) > 0:
                pair_labels = is_unknown[pair_mask]
                pair_scores = msp_novelty[pair_mask]
                class_aurocs.append(roc_auc_score(pair_labels, pair_scores))
    macro_auroc = np.mean(class_aurocs) if class_aurocs else float('nan')

    # Known Accuracy: multi-class accuracy on known test samples
    known_mask = np.isin(te_l, known)
    if np.sum(known_mask) > 0:
        known_preds_raw = [known[i] for i in np.argmax(probs[known_mask], axis=1)]
        known_acc = np.mean(np.array(known_preds_raw) == te_l[known_mask]) * 100.0
    else:
        known_acc = float('nan')

    return macro_auroc, known_acc


# =============================================================================
# Main Evaluation Loop
# =============================================================================
def main():
    start_time = time.time()
    results = {}

    for task_name in TASKS:
        results[task_name] = {}
        print("=" * 70)
        print(f"  TASK: {task_name.upper()}")
        print("=" * 70)

        for k in OS_SHOTS:
            t0 = time.time()
            all_aurocs = []
            all_known_accs = []

            for seed in SEEDS:
                for split in SPLITS:
                    auroc, k_acc = evaluate_split(split, task_name, k, seed)
                    all_aurocs.append(auroc)
                    all_known_accs.append(k_acc)

            mean_auroc = np.mean(all_aurocs)
            std_auroc  = np.std(all_aurocs)
            mean_acc   = np.mean(all_known_accs)
            std_acc    = np.std(all_known_accs)

            results[task_name][k] = {
                'auroc_mean': mean_auroc,
                'auroc_std':  std_auroc,
                'acc_mean':   mean_acc,
                'acc_std':    std_acc,
            }

            elapsed = time.time() - t0
            print(f"  {k:2d}-shot | AUROC: {mean_auroc*100:6.2f}% +/- {std_auroc*100:4.2f}% | "
                  f"Known Acc: {mean_acc:6.2f}% +/- {std_acc:4.2f}% | "
                  f"({elapsed:.1f}s, {len(all_aurocs)} runs)")

    total_time = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"  OSR EVALUATION COMPLETE ({total_time:.1f}s)")
    print("=" * 70 + "\n")

    # Save summary table
    out_path = os.path.join(OUTPUT_DIR, 'CrossScene_OSR_results.txt')
    with open(out_path, 'w') as f:
        f.write("SpectralDINO Houston Cross-Scene Open Set Recognition (OSR) Results\n")
        f.write("Houston-13 -> Houston-18 | MSP Novelty Scoring | 20 Splits x 5 Seeds\n")
        f.write("=" * 75 + "\n\n")

        for task_name, task_res in results.items():
            f.write(f"Task: {task_name}\n")
            f.write("-" * 75 + "\n")
            f.write(f"{'Shot':<8}{'Macro AUROC (%)':<24}{'Known Accuracy (%)':<24}\n")
            f.write("-" * 75 + "\n")
            for k, m in task_res.items():
                f.write(f"{k:<8}"
                        f"{m['auroc_mean']*100:6.2f} +/- {m['auroc_std']*100:4.2f}%      "
                        f"{m['acc_mean']:6.2f} +/- {m['acc_std']:4.2f}%\n")
            f.write("\n")

    print(f"Results saved to: {out_path}\n")


if __name__ == '__main__':
    main()
