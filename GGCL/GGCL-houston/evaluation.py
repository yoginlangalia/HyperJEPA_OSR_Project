"""
Evaluation Module for GGCL
Wraps metrics computation for HSI classification
"""

import numpy as np
import torch
from utils import metrics, show_results


class HSIEvaluation:
    """
    Evaluation class for Hyperspectral Image Classification
    
    Computes OA, AA, Kappa and per-class accuracies
    """
    def __init__(self, num_classes, label_values=None, ignored_labels=[0]):
        self.num_classes = num_classes
        self.label_values = label_values
        self.ignored_labels = ignored_labels
        self.reset()
    
    def reset(self):
        """Reset accumulated predictions"""
        self.predictions = []
        self.targets = []
    
    def update(self, predictions, targets):
        """
        Update with batch predictions and targets
        
        Args:
            predictions: predicted class labels (tensor or array)
            targets: ground truth labels (tensor or array)
        """
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.cpu().numpy()
        
        predictions = predictions.flatten()
        targets = targets.flatten()
        
        self.predictions.extend(predictions.tolist())
        self.targets.extend(targets.tolist())
    
    def compute(self):
        """
        Compute all metrics
        
        Returns:
            dict with OA, AA, Kappa, per-class accuracy, confusion matrix
        """
        predictions = np.array(self.predictions)
        targets = np.array(self.targets)
        
        results = metrics(
            predictions, targets,
            ignored_labels=self.ignored_labels,
            n_classes=self.num_classes
        )
        
        return results
    
    def get_oa(self):
        """Get Overall Accuracy"""
        results = self.compute()
        return results['OA']
    
    def get_aa(self):
        """Get Average Accuracy"""
        results = self.compute()
        return results['AA']
    
    def get_kappa(self):
        """Get Kappa coefficient"""
        results = self.compute()
        return results['Kappa']
    
    def print_results(self):
        """Print formatted results"""
        results = self.compute()
        show_results(results, self.label_values)
        return results


def evaluate_model(model, test_loader, device, num_classes=16):
    """
    Evaluate model on test set
    
    Args:
        model: trained model
        test_loader: test data loader
        device: computation device
        num_classes: number of classes
        
    Returns:
        dict with evaluation metrics
    """
    model.eval()
    evaluator = HSIEvaluation(num_classes=num_classes)
    
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            target = target.to(device)
            
            # Forward pass - only get logits
            outputs = model(data)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            
            predictions = torch.argmax(outputs, dim=1)
            evaluator.update(predictions, target)
    
    results = evaluator.compute()
    return results


def few_shot_evaluate(model, test_loader, device, num_classes=16, shots=[1, 5, 10]):
    """
    Few-shot evaluation (for comparison with other methods)
    
    Note: GGCL doesn't use few-shot in the same way as SVM-based methods,
    but this is provided for completeness
    """
    results = {}
    for shot in shots:
        result = evaluate_model(model, test_loader, device, num_classes)
        results[f'{shot}-shot'] = result
    return results
