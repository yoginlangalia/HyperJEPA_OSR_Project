"""
DMVL Testing/Evaluation for Houston Dataset
Adapted from original Indian Pines implementation

Changes:
- Houston dataset with TR/TE split
- seed=42 for reproducibility (single run)
- Few-shot from TR features, test on TE features
- 15 classes (Houston)
"""

import numpy as np
from sklearn import svm
import pandas as pd
from sklearn.model_selection import GridSearchCV
import joblib
from kappa import kappa
from scipy.io import loadmat
from sklearn import metrics
import time
import matplotlib.pyplot as plt
import matplotlib as mpl
import h5py
import os

# Set seed for reproducibility
SEED = 42
np.random.seed(SEED)

# Houston dataset parameters
NUM_CLASSES = 15  # Houston has 15 classes (excluding background)
# FEW_SHOT_SAMPLES = [1, 5, 10]  # Few-shot settings to evaluate
FEW_SHOT_SAMPLES = [5]  # Only 5-shot for inference (1-shot and 10-shot commented out)

# Create output directory
if not os.path.exists('Houston'):
    os.makedirs('Houston')

# Load TRAINING features (for few-shot sampling)
print("=== Loading Features ===")
f = h5py.File('data/Houston_feature_train.h5', 'r')
train_data = f['data'][:]
train_label = f['label'][:]
f.close()
print(f"Training features: {train_data.shape}")
print(f"Training labels: {train_label.shape}, range: {train_label.min()}-{train_label.max()}")

# Load TEST features (for evaluation)
f = h5py.File('data/Houston_feature_test.h5', 'r')
test_data = f['data'][:]
test_label = f['label'][:]
f.close()
print(f"Test features: {test_data.shape}")
print(f"Test labels: {test_label.shape}, range: {test_label.min()}-{test_label.max()}")

num_test = len(test_label)  # Number of test samples

def sample_per_class(data, labels, num_per_class):
    """Sample num_per_class samples from each class"""
    indices = np.arange(len(labels))
    n_classes = int(labels.max() + 1)
    selected_indices = []
    
    for c in range(n_classes):
        class_indices = indices[labels == c]
        if len(class_indices) >= num_per_class:
            selected = np.random.choice(class_indices, num_per_class, replace=False)
        else:
            print(f"Warning: Not enough samples for class {c}, using all available samples")
            selected = class_indices  # Use all if not enough
        selected_indices.extend(selected)
    
    return data[selected_indices], labels[selected_indices]

# Results storage
all_results = {}

for num_labeled in FEW_SHOT_SAMPLES:
    print(f"\n{'='*60}")
    print(f"=== Few-shot Setting: {num_labeled} samples per class ===")
    print(f"{'='*60}")
    
    # Sample few-shot training data from TRAINING set
    l_images, l_labels = sample_per_class(train_data, train_label, num_labeled)
    print(f"Few-shot training samples: {len(l_labels)}")
    print(f"Classes sampled: {len(np.unique(l_labels))}")
    
    # Train SVM with GridSearchCV
    start = time.time()
    C = np.logspace(-2, 8, 11, base=2)
    gamma = np.logspace(-2, 8, 11, base=2)
    
    parameters = {'C': C, 'gamma': gamma}
    # Adjust CV folds: need at least num_labeled samples per class for stratified split
    # For 1-shot: can't do CV (only 1 sample per class)
    if num_labeled == 1:
        cv_folds = None  # Will use fixed params instead
    else:
        cv_folds = min(3, num_labeled)
    print(f"Using {cv_folds}-fold cross-validation" if cv_folds else "1-shot: Using fixed SVM params - no CV possible")
    clf = svm.SVC(kernel='rbf', C=1.0, gamma='scale') if cv_folds is None else GridSearchCV(svm.SVC(kernel='rbf'), parameters, cv=cv_folds, refit=True)
    clf.fit(l_images, l_labels)
    
    print(f"Training time: {time.time() - start:.2f}s")
    print(f"Best params: {clf.best_params_}" if cv_folds else "Best params: C=1.0, gamma='scale' (fixed)")
    print(f"Best CV score: {clf.best_score_:.4f}" if cv_folds else "Best CV score: N/A (no CV for 1-shot)")
    
    # Save the 5-shot SVM model for inference
    if num_labeled == 5:
        svm_model_path = 'results/5shot_svm_model.joblib'
        joblib.dump(clf, svm_model_path)
        print(f"5-shot SVM model saved to {svm_model_path}")
    
    # Predict on TEST set only (no data leakage!)
    start = time.time()
    predict_label = clf.predict(test_data)
    
    # Compute metrics
    matrix = metrics.confusion_matrix(test_label, predict_label)
    print(f"\nConfusion Matrix:")
    print(matrix)
    
    OA = np.trace(matrix) / float(num_test) * 100
    print(f"\nOA = {OA:.2f}%")
    
    kappa_val, AA, CA = kappa(matrix, NUM_CLASSES)
    kappa_val = kappa_val * 100
    print(f"AA = {AA:.2f}%")
    print(f"Kappa = {kappa_val:.2f}%")
    print(f"Prediction time: {time.time() - start:.2f}s")
    
    # Store results
    all_results[num_labeled] = {
        'OA': OA,
        'AA': AA,
        'Kappa': kappa_val,
        'CA': CA
    }

# Print summary
print(f"\n{'='*60}")
print("=== FINAL RESULTS SUMMARY ===")
print(f"{'='*60}")
print(f"\n{'Shots':<10} {'OA (%)':<15} {'AA (%)':<15} {'Kappa (%)':<15}")
print("-" * 55)
for shots in FEW_SHOT_SAMPLES:
    r = all_results[shots]
    print(f"{shots:<10} {r['OA']:<15.2f} {r['AA']:<15.2f} {r['Kappa']:<15.2f}")

# Save results to file
with open('Houston/Houston_results.txt', 'w') as f:
    f.write("DMVL Houston Results\n")
    f.write(f"Seed: {SEED}\n")
    f.write(f"Number of classes: {NUM_CLASSES}\n")
    f.write(f"Test samples: {num_test}\n\n")
    
    for shots in FEW_SHOT_SAMPLES:
        r = all_results[shots]
        f.write(f"\n=== {shots}-shot ===")
        f.write(f"\nOA: {r['OA']:.2f}%")
        f.write(f"\nAA: {r['AA']:.2f}%")
        f.write(f"\nKappa: {r['Kappa']:.2f}%")
        f.write(f"\nClass-wise Accuracy: {r['CA']}\n")

print(f"\nResults saved to Houston/Houston_results.txt")
print("=== Evaluation Complete ===")