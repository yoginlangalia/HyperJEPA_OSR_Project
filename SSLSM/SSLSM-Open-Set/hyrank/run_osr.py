#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SSLSM Open Set Recognition — HyRank Cross-Scene (Dioni → Loukia)

Prerequisites:
    Run step1_extract_features.py first (or run.py).
    Features must exist in: ./data/
        - Dioni_feature.h5
        - Loukia_feature_train.h5
        - Loukia_feature_test.h5

Methodology:
    - MSP (Maximum Softmax Probability) based open-set detection
    - N_UNKNOWN=3 classes hidden per split
    - RobustProbe (2-layer MLP) trained on known classes only
    - Evaluated via class-balanced AUROC across 84 splits and 5 seeds
    - Also records and reports Known Accuracy (multi-class accuracy on known classes)

Usage:
    cd SSLSM-Open-Set/hyrank
    python run_osr.py
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
NUM_CLASSES    = 12             # 12 classes (0-indexed: 0–11)
OS_SHOTS       = (1, 5, 10)
SEEDS          = [42, 123, 456, 789, 1024]
N_UNKNOWN      = 3
MIN_TEST_PX    = 50             # class must have >= this many Loukia test pixels to be an unknown class
ALWAYS_KNOWN   = [0, 1, 3]      # Classes 1, 2, 4 (0-based: 0, 1, 3) have < 50 test pixels (41, 10, 15)

SCRIPT_DIR         = os.path.dirname(os.path.abspath(__file__))
LOCAL_FEATURE_DIR  = os.path.join(SCRIPT_DIR, 'data')

if os.path.isdir(LOCAL_FEATURE_DIR) and os.path.isfile(os.path.join(LOCAL_FEATURE_DIR, 'Dioni_feature.h5')):
    FEATURE_DIR = LOCAL_FEATURE_DIR
else:
    FEATURE_DIR = LOCAL_FEATURE_DIR

OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'HyRank')

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
print("  SSLSM Open Set Recognition — HyRank (Dioni → Loukia)")
print("=" * 70)
print(f"\nDevice: {device}")
print(f"Loading features from: {FEATURE_DIR}\n")

required_files = [
    'Dioni_feature.h5',
    'Loukia_feature_train.h5',
    'Loukia_feature_test.h5',
]
for fname in required_files:
    fpath = os.path.join(FEATURE_DIR, fname)
    if not os.path.isfile(fpath):
        print(f"ERROR: Required feature file not found: {fpath}")
        print("       Run step1_extract_features.py first.")
        sys.exit(1)

# Dioni features (zero-shot support pool)
with h5py.File(os.path.join(FEATURE_DIR, 'Dioni_feature.h5'), 'r') as f:
    dioni_data  = f['data'][:]
    dioni_label = f['label'][:].astype(int)
print(f"Dioni features      : {dioni_data.shape}, classes: {np.unique(dioni_label)}")

# Loukia Train features (adaptation support pool)
with h5py.File(os.path.join(FEATURE_DIR, 'Loukia_feature_train.h5'), 'r') as f:
    loukia_train_data  = f['data'][:]
    loukia_train_label = f['label'][:].astype(int)
print(f"Loukia Train        : {loukia_train_data.shape}, classes: {np.unique(loukia_train_label)}")

# Loukia Test features (evaluation — 20% of target)
with h5py.File(os.path.join(FEATURE_DIR, 'Loukia_feature_test.h5'), 'r') as f:
    te_f = f['data'][:]
    te_l = f['label'][:].astype(int)
print(f"Loukia Test         : {te_f.shape}, classes: {np.unique(te_l)}")

FEAT_DIM = te_f.shape[1]
print(f"Feature dim         : {FEAT_DIM}")


# =============================================================================
# Task definitions
# =============================================================================
TASKS = {
    "zero-shot (Dioni -> Loukia)": {
        "data":   dioni_data,
        "labels": dioni_label,
    },
    "adaptation (Loukia -> Loukia)": {
        "data":   loukia_train_data,
        "labels": loukia_train_label,
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

print(f"\nTest pixels per class (0-based): {dict(enumerate(n_test.tolist()))}")
print(f"Eligible unknown classes       : {eligible}")
print(f"ALWAYS_KNOWN                   : {ALWAYS_KNOWN}")
print(f"Number of OSR splits           : {len(SPLITS)}")


# =============================================================================
# Support sampling
# =============================================================================
def sample_per_class(data, labels, num_per_class):
    """Sample num_per_class samples from each class."""
    indices = np.arange(len(labels))
    selected = []
    for c in range(NUM_CLASSES):
        class_idx = indices[labels == c]
        if len(class_idx) >= num_per_class:
            chosen = np.random.choice(class_idx, num_per_class, replace=False)
        else:
            chosen = class_idx
        selected.extend(chosen)
    return data[selected], labels[selected]


def get_support(task_name, shots, seed):
    """Draw a few-shot support set for the given task / seed."""
    cfg = TASKS[task_name]
    random.seed(seed)
    np.random.seed(seed)
    f, l = sample_per_class(cfg["data"], cfg["labels"], shots)
    return f, l.astype(int)


# =============================================================================
# MSP-based open-set scoring
# =============================================================================
def probe_msp_scores(tr_f, tr_l, known, te_feats, seed, epochs=150):
    """Train a RobustProbe on known-class support, return MSP novelty scores."""
    km = np.isin(tr_l, known)
    remap = {c: i for i, c in enumerate(known)}
    Y = torch.tensor([remap[c] for c in tr_l[km]], dtype=torch.long)
    X = torch.tensor(tr_f[km], dtype=torch.float32)

    torch.manual_seed(seed)
    probe = RobustProbe(X.shape[1], len(known)).to(device)
    opt   = optim.Adam(probe.parameters(), lr=1e-3, weight_decay=1e-4)
    loader = DataLoader(TensorDataset(X, Y), batch_size=64, shuffle=True)

    for _ in range(epochs):
        probe.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = F.cross_entropy(probe(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()

    probe.eval()
    with torch.no_grad():
        logits = probe(torch.tensor(te_feats, dtype=torch.float32).to(device))
        p = F.softmax(logits, dim=1)
    conf, arg = p.max(1)
    return 1.0 - conf.cpu().numpy(), np.asarray(known)[arg.cpu().numpy()]


def macro_auroc_from_score(score, te_labels, split):
    """Per-unknown-class AUROC: distinguishes each unknown class from all known classes."""
    kn = np.isin(te_labels, split["known"])
    out = {}
    for u in split["unknown"]:
        mask = (te_labels == u) | kn
        out[u] = 100.0 * roc_auc_score(te_labels[mask] == u, score[mask])
    return out


# =============================================================================
# Main evaluation loop
# =============================================================================
def run_msp(feat_fn, te_feats, tasks=TASKS, shots_list=OS_SHOTS,
            seeds=SEEDS, tag=""):
    """Run MSP open-set evaluation across tasks x shots x seeds x splits."""
    results = {}
    for tname in tasks:
        for shots in shots_list:
            bal  = np.zeros((len(seeds), len(SPLITS)))
            kacc = np.zeros_like(bal)

            for i, seed in enumerate(seeds):
                tr_f, tr_l = feat_fn(tname, shots, seed)
                for j, sp in enumerate(SPLITS):
                    score, pred = probe_msp_scores(tr_f, tr_l,
                                                   sp["known"], te_feats, seed)
                    # Class-balanced AUROC
                    per_cls = macro_auroc_from_score(score, te_l, sp)
                    bal[i, j] = np.mean(list(per_cls.values()))
                    # Known-class accuracy
                    kn_mask = ~np.isin(te_l, sp["unknown"])
                    kacc[i, j] = 100.0 * (pred[kn_mask] == te_l[kn_mask]).mean()

            results[(tname, shots)] = {"bal": bal, "kacc": kacc}
            print(f"[{tag}] [{tname}] {shots:>2}-shot class-balanced AUROC "
                  f"{bal.mean():.2f} ± {bal.mean(1).std(ddof=1):.2f} (seeds), "
                  f"± {bal.mean(0).std(ddof=1):.2f} (splits) | "
                  f"known acc {kacc.mean():.2f}%")
    return results


# =============================================================================
# Run
# =============================================================================
print(f"\n{'=' * 70}")
print("  Starting MSP Open-Set Evaluation")
print(f"{'=' * 70}\n")

start_time = time.time()
MSP_ENC = run_msp(get_support, te_f, tag="encoder")
elapsed = time.time() - start_time

print(f"\n{'=' * 70}")
print(f"  OSR Evaluation Complete — {elapsed:.1f}s ({elapsed / 60:.1f} min)")
print(f"{'=' * 70}")


# =============================================================================
# Save results to file
# =============================================================================
os.makedirs(OUTPUT_DIR, exist_ok=True)
results_path = os.path.join(OUTPUT_DIR, 'HyRank_OSR_results.txt')

with open(results_path, 'w') as fp:
    fp.write("SSLSM Open Set Recognition — HyRank Cross-Scene (Dioni → Loukia)\n")
    fp.write("=" * 65 + "\n\n")
    fp.write(f"Feature dim      : {FEAT_DIM}\n")
    fp.write(f"N_UNKNOWN        : {N_UNKNOWN}\n")
    fp.write(f"MIN_TEST_PX      : {MIN_TEST_PX}\n")
    fp.write(f"ALWAYS_KNOWN     : {ALWAYS_KNOWN}\n")
    fp.write(f"Seeds            : {SEEDS}\n")
    fp.write(f"Eligible unknowns: {eligible}\n")
    fp.write(f"Num splits       : {len(SPLITS)}\n")
    fp.write(f"Total time       : {elapsed:.1f}s\n\n")

    for tname in TASKS:
        fp.write(f"=== {tname} ===\n")
        for shots in OS_SHOTS:
            bal  = MSP_ENC[(tname, shots)]["bal"]
            kacc = MSP_ENC[(tname, shots)]["kacc"]
            fp.write(f"  {shots:>2}-shot AUROC: {bal.mean():.2f} "
                     f"± {bal.mean(1).std(ddof=1):.2f} (seeds), "
                     f"± {bal.mean(0).std(ddof=1):.2f} (splits) | "
                     f"Known Acc: {kacc.mean():.2f}%\n")
        fp.write("\n")

print(f"\nResults saved to {results_path}")
