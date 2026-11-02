"""
DMVL Testing/Evaluation for Cross-Scene Dataset

Experiment 1 (Zero-shot): Few-shots from Houston-13, test on Houston-18 Test
Experiment 2 (Few-shot): Few-shots from Houston-18 Train, test on Houston-18 Test
"""

import numpy as np
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from kappa import kappa
from sklearn import metrics
import time
import h5py
import os

# Set seed for reproducibility
SEED = 42
np.random.seed(SEED)

# Cross-scene dataset parameters
NUM_CLASSES = 7  # Classes 1-7 (0 is background, excluded)
FEW_SHOT_SAMPLES = [1, 5, 10]

# Create output directory
if not os.path.exists('CrossScene'):
    os.makedirs('CrossScene')

# ===========================================
# Load Features
# ===========================================
print("=== Loading Features ===")

# Houston-13 features (for Exp1)
f = h5py.File('data/Houston13_feature.h5', 'r')
h13_data = f['data'][:]
h13_label = f['label'][:]
f.close()
print(f"Houston-13 features: {h13_data.shape}")
print(f"Houston-13 labels: {h13_label.shape}, range: {h13_label.min()}-{h13_label.max()}")

# Houston-18 Train features (for Exp2)
f = h5py.File('data/Houston18_feature_train.h5', 'r')
h18_train_data = f['data'][:]
h18_train_label = f['label'][:]
f.close()
print(f"Houston-18 Train features: {h18_train_data.shape}")
print(f"Houston-18 Train labels: {h18_train_label.shape}, range: {h18_train_label.min()}-{h18_train_label.max()}")

# Houston-18 Test features (for evaluation)
f = h5py.File('data/Houston18_feature_test.h5', 'r')
h18_test_data = f['data'][:]
h18_test_label = f['label'][:]
f.close()
print(f"Houston-18 Test features: {h18_test_data.shape}")
print(f"Houston-18 Test labels: {h18_test_label.shape}, range: {h18_test_label.min()}-{h18_test_label.max()}")

num_test = len(h18_test_label)

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
            print(f"Warning: Class {c} has only {len(class_indices)} samples, using all")
            selected = class_indices
        selected_indices.extend(selected)
    
    return data[selected_indices], labels[selected_indices]

def run_experiment(exp_name, train_data, train_label, test_data, test_label, few_shot_samples):
    """Run few-shot experiment"""
    print(f"\n{'='*70}")
    print(f"  {exp_name}")
    print(f"{'='*70}")
    
    results = {}
    
    for num_labeled in few_shot_samples:
        print(f"\n{'='*60}")
        print(f"=== {num_labeled}-shot ===")
        print(f"{'='*60}")
        
        # Reset seed for reproducibility
        np.random.seed(SEED)
        
        # Sample few-shot training data
        l_images, l_labels = sample_per_class(train_data, train_label, num_labeled)
        print(f"Few-shot samples: {len(l_labels)}")
        print(f"Classes sampled: {len(np.unique(l_labels))}")
        
        # Train SVM
        start = time.time()
        
        # Handle different shot settings
        if num_labeled == 1:
            # 1-shot: Use lower C for more regularization (prevent overfitting to 1 sample)
            cv_folds = None
            print("1-shot: fixed SVM params (C=0.1, gamma='scale')")
            clf = svm.SVC(kernel='rbf', C=0.1, gamma='scale')
        elif num_labeled == 10:
            # 10-shot: Use same range as 5-shot for consistency
            cv_folds = 3
            C = np.logspace(-2, 8, 11, base=2)  # Same as 5-shot
            gamma = np.logspace(-2, 8, 11, base=2)  # Same as 5-shot
            parameters = {'C': C, 'gamma': gamma}
            print(f"Using {cv_folds}-fold CV (10-shot)")
            clf = GridSearchCV(svm.SVC(kernel='rbf'), parameters, cv=cv_folds, refit=True)
        else:
            # 5-shot and others: Original params
            cv_folds = min(3, num_labeled)
            C = np.logspace(-2, 8, 11, base=2)
            gamma = np.logspace(-2, 8, 11, base=2)
            parameters = {'C': C, 'gamma': gamma}
            print(f"Using {cv_folds}-fold CV")
            clf = GridSearchCV(svm.SVC(kernel='rbf'), parameters, cv=cv_folds, refit=True)
        
        clf.fit(l_images, l_labels)
        
        print(f"Training time: {time.time() - start:.2f}s")
        if cv_folds:
            print(f"Best params: {clf.best_params_}")
            print(f"Best CV score: {clf.best_score_:.4f}")
        
        # Predict on test set
        start = time.time()
        predict_label = clf.predict(test_data)
        
        # Compute metrics
        matrix = metrics.confusion_matrix(test_label, predict_label)
        print(f"\nConfusion Matrix:")
        print(matrix)
        
        OA = np.trace(matrix) / float(len(test_label)) * 100
        print(f"\nOA = {OA:.2f}%")
        
        kappa_val, AA, CA = kappa(matrix, NUM_CLASSES)
        kappa_val = kappa_val * 100
        print(f"AA = {AA:.2f}%")
        print(f"Kappa = {kappa_val:.2f}%")
        print(f"Prediction time: {time.time() - start:.2f}s")
        
        results[num_labeled] = {
            'OA': OA,
            'AA': AA,
            'Kappa': kappa_val,
            'CA': CA
        }
    
    return results


# ===========================================
# EXPERIMENT 1: Zero-shot
# Few-shots from Houston-13, Test on Houston-18 Test
# ===========================================
exp1_results = run_experiment(
    "EXPERIMENT 1: Zero-shot (Few-shots from Houston-13)",
    h13_data, h13_label,
    h18_test_data, h18_test_label,
    FEW_SHOT_SAMPLES
)

# ===========================================
# EXPERIMENT 2: Few-shot
# Few-shots from Houston-18 Train, Test on Houston-18 Test
# ===========================================
exp2_results = run_experiment(
    "EXPERIMENT 2: Few-shot (Few-shots from Houston-18 Train)",
    h18_train_data, h18_train_label,
    h18_test_data, h18_test_label,
    FEW_SHOT_SAMPLES
)

# ===========================================
# Print Summary
# ===========================================
print(f"\n{'='*70}")
print("  FINAL RESULTS SUMMARY")
print(f"{'='*70}")

print(f"\n=== EXPERIMENT 1: Zero-shot (Few-shots from Houston-13) ===")
print(f"\n{'Shots':<10} {'OA (%)':<15} {'AA (%)':<15} {'Kappa (%)':<15}")
print("-" * 55)
for shots in FEW_SHOT_SAMPLES:
    r = exp1_results[shots]
    print(f"{shots:<10} {r['OA']:<15.2f} {r['AA']:<15.2f} {r['Kappa']:<15.2f}")

print(f"\n=== EXPERIMENT 2: Few-shot (Few-shots from Houston-18 Train) ===")
print(f"\n{'Shots':<10} {'OA (%)':<15} {'AA (%)':<15} {'Kappa (%)':<15}")
print("-" * 55)
for shots in FEW_SHOT_SAMPLES:
    r = exp2_results[shots]
    print(f"{shots:<10} {r['OA']:<15.2f} {r['AA']:<15.2f} {r['Kappa']:<15.2f}")

# ===========================================
# Save Results
# ===========================================
with open('CrossScene/CrossScene_results.txt', 'w') as f:
    f.write("DMVL Cross-Scene Results\n")
    f.write(f"Seed: {SEED}\n")
    f.write(f"Number of classes: {NUM_CLASSES}\n")
    f.write(f"Test samples: {num_test}\n\n")
    
    f.write("=== EXPERIMENT 1: Zero-shot (Few-shots from Houston-13) ===\n")
    for shots in FEW_SHOT_SAMPLES:
        r = exp1_results[shots]
        f.write(f"\n{shots}-shot:\n")
        f.write(f"  OA: {r['OA']:.2f}%\n")
        f.write(f"  AA: {r['AA']:.2f}%\n")
        f.write(f"  Kappa: {r['Kappa']:.2f}%\n")
    
    f.write("\n=== EXPERIMENT 2: Few-shot (Few-shots from Houston-18 Train) ===\n")
    for shots in FEW_SHOT_SAMPLES:
        r = exp2_results[shots]
        f.write(f"\n{shots}-shot:\n")
        f.write(f"  OA: {r['OA']:.2f}%\n")
        f.write(f"  AA: {r['AA']:.2f}%\n")
        f.write(f"  Kappa: {r['Kappa']:.2f}%\n")

print(f"\nResults saved to CrossScene/CrossScene_results.txt")
print("=== Evaluation Complete ===")