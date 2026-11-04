#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SSLSM Closed-Set Evaluation for Cross-Scene Dataset (Houston 2013 → 2018)

Experiment 1 (Zero-shot): Few-shots from Houston-13, test on Houston-18 Test
Experiment 2 (Few-shot):  Few-shots from Houston-18 Train, test on Houston-18 Test
"""

import os
import sys
import time
import h5py
import numpy as np
from sklearn import svm
from sklearn import metrics

from kappa import kappa

SEED = 42
np.random.seed(SEED)

NUM_CLASSES = 7
FEW_SHOT_SAMPLES = [1, 5, 10]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'CrossScene')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def resolve_data_path(fname):
    if os.path.isfile(fname):
        return fname
    p1 = os.path.join(SCRIPT_DIR, fname)
    if os.path.isfile(p1):
        return p1
    p2 = os.path.join(SCRIPT_DIR, 'data', os.path.basename(fname))
    if os.path.isfile(p2):
        return p2
    return p1


print("=== Loading Features ===")
p_h13 = resolve_data_path('data/Houston13_feature.h5')
print(f"Reading Houston-13: {p_h13}")
with h5py.File(p_h13, 'r') as f:
    h13_data = f['data'][:]
    h13_label = f['label'][:]

p_h18_tr = resolve_data_path('data/Houston18_feature_train.h5')
print(f"Reading Houston-18 Train: {p_h18_tr}")
with h5py.File(p_h18_tr, 'r') as f:
    h18_train_data = f['data'][:]
    h18_train_label = f['label'][:]

p_h18_te = resolve_data_path('data/Houston18_feature_test.h5')
print(f"Reading Houston-18 Test: {p_h18_te}")
with h5py.File(p_h18_te, 'r') as f:
    h18_test_data = f['data'][:]
    h18_test_label = f['label'][:]

print(f"Feature dim: {h18_test_data.shape[1]}")


def sample_per_class(data, labels, num_per_class):
    indices = np.arange(len(labels))
    n_classes = int(labels.max() + 1)
    selected_indices = []
    for c in range(n_classes):
        class_indices = indices[labels == c]
        if len(class_indices) >= num_per_class:
            chosen = np.random.choice(class_indices, num_per_class, replace=False)
        else:
            chosen = class_indices
        selected_indices.extend(chosen)
    return data[selected_indices], labels[selected_indices]


def run_experiment(exp_name, train_data, train_label, test_data, test_label, few_shot_samples):
    print(f"\n{'='*70}\n  {exp_name}\n{'='*70}")
    results = {}
    for num_samples in few_shot_samples:
        print(f"\n>>> Running {num_samples}-shot per class")
        tr_d, tr_l = sample_per_class(train_data, train_label, num_samples)
        
        clf = svm.SVC(C=10.0, kernel='rbf', gamma='scale')
        clf.fit(tr_d, tr_l)
        preds = clf.predict(test_data)
        
        oa = metrics.accuracy_score(test_label, preds) * 100
        cm = metrics.confusion_matrix(test_label, preds)
        cohen_kappa, aa, per_cls = kappa(cm, NUM_CLASSES)
        
        print(f"  OA: {oa:.2f}% | AA: {aa:.2f}% | Kappa: {cohen_kappa:.4f}")
        results[num_samples] = {'oa': oa, 'aa': aa, 'kappa': cohen_kappa}
    return results


def main():
    exp1_res = run_experiment("Exp 1: Zero-Shot (H13 -> H18)", h13_data, h13_label,
                              h18_test_data, h18_test_label, FEW_SHOT_SAMPLES)
    exp2_res = run_experiment("Exp 2: Few-Shot Adaptation (H18 Train -> H18 Test)", h18_train_data, h18_train_label,
                              h18_test_data, h18_test_label, FEW_SHOT_SAMPLES)

    res_file = os.path.join(OUTPUT_DIR, 'CrossScene_results.txt')
    with open(res_file, 'w') as f:
        f.write("SSLSM Closed-Set Few-Shot Classification Results (Houston 2013 -> 2018)\n")
        f.write("=" * 70 + "\n\n")
        f.write("=== Exp 1: Zero-shot (H13 -> H18) ===\n")
        for s in FEW_SHOT_SAMPLES:
            r = exp1_res[s]
            f.write(f"  {s:>2}-shot | OA: {r['oa']:.2f}% | AA: {r['aa']:.2f}% | Kappa: {r['kappa']:.4f}\n")
        f.write("\n=== Exp 2: Few-shot Adaptation (H18 -> H18) ===\n")
        for s in FEW_SHOT_SAMPLES:
            r = exp2_res[s]
            f.write(f"  {s:>2}-shot | OA: {r['oa']:.2f}% | AA: {r['aa']:.2f}% | Kappa: {r['kappa']:.4f}\n")
    print(f"\nSaved closed-set results to: {res_file}")


if __name__ == '__main__':
    main()
