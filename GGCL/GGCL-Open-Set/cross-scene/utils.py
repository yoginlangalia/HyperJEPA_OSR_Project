"""
Utility functions for GGCL HSI Classification
Adapted from: https://github.com/eecn/Hyperspectral-Classification
"""

import random
import numpy as np
from sklearn.metrics import confusion_matrix, cohen_kappa_score
import sklearn.model_selection
import os
import torch
from scipy import io
import logging


# ============================================
# Device and Reproducibility Utils
# ============================================

def get_device(ordinal=-1):
    """Get computation device"""
    if ordinal < 0:
        print("Computation on CPU")
        device = torch.device('cpu')
    elif torch.cuda.is_available():
        print(f"Computation on CUDA GPU device {ordinal}")
        device = torch.device(f'cuda:{ordinal}')
    else:
        print("/!\\ CUDA was requested but is not available! Using CPU.")
        device = torch.device('cpu')
    return device


def set_seed(seed=42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ============================================
# File I/O Utils
# ============================================

def open_file(dataset):
    """Load dataset file based on extension"""
    _, ext = os.path.splitext(dataset)
    ext = ext.lower()
    if ext == '.mat':
        return io.loadmat(dataset)
    elif ext == '.npy':
        return np.load(dataset)
    else:
        raise ValueError(f"Unknown file format: {ext}")


# ============================================
# Data Processing Utils
# ============================================

def sliding_window(image, step=10, window_size=(20, 20), with_data=True):
    """
    Sliding window generator over an input image.
    
    Args:
        image: 2D+ image to slide the window on (H x W x C)
        step: stride of the sliding window
        window_size: (width, height) of the window
        with_data: bool, return data along with coordinates
        
    Yields:
        ([data], x, y, w, h) - patch data and coordinates
    """
    w, h = window_size
    W, H = image.shape[:2]
    offset_w = (W - w) % step
    offset_h = (H - h) % step
    
    for x in range(0, W - w + offset_w, step):
        if x + w > W:
            x = W - w
        for y in range(0, H - h + offset_h, step):
            if y + h > H:
                y = H - h
            if with_data:
                yield image[x:x + w, y:y + h], x, y, w, h
            else:
                yield x, y, w, h


def count_sliding_window(top, step=10, window_size=(20, 20)):
    """Count the number of windows in an image."""
    sw = sliding_window(top, step, window_size, with_data=False)
    return sum(1 for _ in sw)


def sample_gt(gt, train_size, mode='random'):
    """
    Extract a fixed percentage of samples from an array of labels.
    
    Args:
        gt: 2D array of int labels
        train_size: float (percentage) or int (fixed number per class)
        mode: 'random' or 'fixed'
        
    Returns:
        train_gt, test_gt: 2D arrays of int labels
    """
    indices = np.nonzero(gt)
    X = list(zip(*indices))  # x,y features
    y = gt[indices].ravel()  # classes
    train_gt = np.zeros_like(gt)
    test_gt = np.zeros_like(gt)
    
    if train_size > 1:
        train_size = int(train_size)
    
    if mode == 'random':
        train_indices, test_indices = sklearn.model_selection.train_test_split(
            X, train_size=train_size, stratify=y, random_state=42
        )
        train_indices = [list(t) for t in zip(*train_indices)]
        test_indices = [list(t) for t in zip(*test_indices)]
        train_gt[tuple(train_indices)] = gt[tuple(train_indices)]
        test_gt[tuple(test_indices)] = gt[tuple(test_indices)]
        
    elif mode == 'fixed':
        print(f"Sampling {mode} with train size = {train_size}")
        train_indices, test_indices = [], []
        for c in np.unique(gt):
            if c == 0:
                continue
            indices_c = np.nonzero(gt == c)
            X_c = list(zip(*indices_c))
            train, test = sklearn.model_selection.train_test_split(
                X_c, train_size=train_size, random_state=42
            )
            train_indices += train
            test_indices += test
        train_indices = [list(t) for t in zip(*train_indices)]
        test_indices = [list(t) for t in zip(*test_indices)]
        train_gt[tuple(train_indices)] = gt[tuple(train_indices)]
        test_gt[tuple(test_indices)] = gt[tuple(test_indices)]
    else:
        raise ValueError(f"{mode} sampling is not implemented.")
    
    return train_gt, test_gt


def build_dataset(mat, gt, ignored_labels=None):
    """
    Create a list of training samples based on an image and a mask.
    
    Args:
        mat: 3D hyperspectral matrix (H x W x C)
        gt: 2D ground truth labels
        ignored_labels: list of classes to ignore
        
    Returns:
        samples: array of spectra
        labels: array of labels
    """
    samples = []
    labels = []
    
    if ignored_labels is None:
        ignored_labels = []
    
    assert mat.shape[:2] == gt.shape[:2]
    
    for label in np.unique(gt):
        if label in ignored_labels:
            continue
        indices = np.nonzero(gt == label)
        samples += list(mat[indices])
        labels += len(indices[0]) * [label]
    
    return np.asarray(samples), np.asarray(labels)


# ============================================
# Metrics and Evaluation Utils
# ============================================

def metrics(prediction, target, ignored_labels=[], n_classes=None):
    """
    Compute classification metrics.
    
    Args:
        prediction: array of predicted labels
        target: array of ground truth labels
        ignored_labels: labels to ignore
        n_classes: number of classes
        
    Returns:
        dict with Accuracy, AA, Kappa, F1 scores, confusion matrix
    """
    ignored_mask = np.zeros(target.shape, dtype=bool)
    for l in ignored_labels:
        ignored_mask[target == l] = True
    ignored_mask = ~ignored_mask
    
    target = target[ignored_mask]
    prediction = prediction[ignored_mask]
    
    results = {}
    
    n_classes = np.max(target) + 1 if n_classes is None else n_classes
    
    cm = confusion_matrix(target, prediction, labels=range(n_classes))
    results["Confusion matrix"] = cm
    
    # Overall Accuracy (OA)
    total = np.sum(cm)
    accuracy = sum([cm[x][x] for x in range(len(cm))])
    accuracy *= 100 / float(total)
    results["OA"] = accuracy
    
    # Per-class accuracy and Average Accuracy (AA)
    per_class_acc = []
    for i in range(len(cm)):
        if np.sum(cm[i, :]) > 0:
            per_class_acc.append(cm[i, i] / np.sum(cm[i, :]) * 100)
        else:
            per_class_acc.append(0)
    results["Per-class accuracy"] = per_class_acc
    # FIX: Include ALL classes in AA calculation, not just non-zero
    # This prevents misleading 100% AA when most classes have 0% accuracy
    results["AA"] = np.mean(per_class_acc)
    
    # F1 scores (per-class)
    F1scores = np.zeros(len(cm))
    for i in range(len(cm)):
        try:
            F1 = 2. * cm[i, i] / (np.sum(cm[i, :]) + np.sum(cm[:, i]))
        except ZeroDivisionError:
            F1 = 0.
        F1scores[i] = F1
    results["F1 scores"] = F1scores
    
    # F1 macro (average of per-class F1)
    results["F1"] = np.mean(F1scores) * 100
    
    # Kappa coefficient
    pa = np.trace(cm) / float(total)
    pe = np.sum(np.sum(cm, axis=0) * np.sum(cm, axis=1)) / float(total * total)
    kappa = (pa - pe) / (1 - pe) if (1 - pe) != 0 else 0
    results["Kappa"] = kappa * 100  # Convert to percentage
    
    return results


def show_results(results, label_values=None):
    """Print classification results"""
    text = "\n" + "=" * 50 + "\n"
    text += "CLASSIFICATION RESULTS\n"
    text += "=" * 50 + "\n"
    
    text += f"Overall Accuracy (OA): {results['OA']:.2f}%\n"
    text += f"Average Accuracy (AA): {results['AA']:.2f}%\n"
    text += f"Kappa Coefficient: {results['Kappa']:.2f}%\n"
    text += "-" * 50 + "\n"
    
    if label_values is not None:
        text += "Per-class accuracy:\n"
        for i, (label, acc) in enumerate(zip(label_values, results['Per-class accuracy'])):
            text += f"  Class {i} ({label}): {acc:.2f}%\n"
    
    print(text)
    return text


# ============================================
# Training Utils
# ============================================

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Recorder:
    """Records training history"""
    def __init__(self, save_path=None):
        self.history = {
            'train_loss': [],
            'test_oa': [],
            'test_aa': [],
            'test_kappa': [],
            'best_oa': 0,
            'best_aa': 0,
            'best_kappa': 0,
            'best_epoch': 0
        }
        self.save_path = save_path
    
    def update(self, epoch, train_loss=None, test_results=None):
        if train_loss is not None:
            self.history['train_loss'].append(train_loss)
        
        if test_results is not None:
            oa = test_results.get('OA', 0)
            aa = test_results.get('AA', 0)
            kappa = test_results.get('Kappa', 0)
            
            self.history['test_oa'].append(oa)
            self.history['test_aa'].append(aa)
            self.history['test_kappa'].append(kappa)
            
            if oa > self.history['best_oa']:
                self.history['best_oa'] = oa
                self.history['best_aa'] = aa
                self.history['best_kappa'] = kappa
                self.history['best_epoch'] = epoch
                return True
        return False
    
    def get_best(self):
        return {
            'OA': self.history['best_oa'],
            'AA': self.history['best_aa'],
            'Kappa': self.history['best_kappa'],
            'epoch': self.history['best_epoch']
        }
    
    def save(self, filename=None):
        if filename is None and self.save_path is not None:
            filename = os.path.join(self.save_path, 'training_history.npy')
        if filename:
            np.save(filename, self.history)


def compute_imf_weights(ground_truth, n_classes=None, ignored_classes=[]):
    """
    Compute inverse median frequency weights for class balancing.
    """
    n_classes = np.max(ground_truth) if n_classes is None else n_classes
    weights = np.zeros(n_classes)
    frequencies = np.zeros(n_classes)
    
    for c in range(0, n_classes):
        if c in ignored_classes:
            continue
        frequencies[c] = np.count_nonzero(ground_truth == c)
    
    frequencies /= np.sum(frequencies)
    idx = np.nonzero(frequencies)
    median = np.median(frequencies[idx])
    weights[idx] = median / frequencies[idx]
    weights[frequencies == 0] = 0.
    
    return weights


# ============================================
# Logging Utils
# ============================================

def setup_logger(name, log_file=None, level=logging.INFO):
    """Setup logger with console and optional file handler"""
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    # File handler
    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    
    return logger
