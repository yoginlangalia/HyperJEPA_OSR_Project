"""
Weight Scheduling Functions for GGCL
Controls the balance between contrastive loss and cross-entropy loss during training
"""

import numpy as np


def get_fun(epoch, max_epoch=200, weight_type='linear'):
    """
    Get contrastive loss weight that decreases during training.
    Initially emphasizes contrastive learning, gradually shifts to supervised.
    
    Args:
        epoch: current epoch
        max_epoch: total number of epochs
        weight_type: 'linear', 'cosine', 'step', or 'exp'
        
    Returns:
        weight for contrastive loss (decreases from 1 to 0)
    """
    t = epoch / max_epoch
    
    if weight_type == 'linear':
        # Linear decrease from 1 to 0
        weight = 1.0 - t
        
    elif weight_type == 'cosine':
        # Cosine annealing from 1 to 0
        weight = 0.5 * (1 + np.cos(np.pi * t))
        
    elif weight_type == 'step':
        # Step decrease at certain milestones
        if t < 0.33:
            weight = 1.0
        elif t < 0.66:
            weight = 0.5
        else:
            weight = 0.1
            
    elif weight_type == 'exp':
        # Exponential decay
        weight = np.exp(-3 * t)
        
    elif weight_type == 'constant':
        # Fixed weight
        weight = 0.5
        
    else:
        weight = 1.0 - t
    
    return max(0.0, min(1.0, weight))


def get_fun_label(epoch, max_epoch=200, weight_type='linear'):
    """
    Get cross-entropy loss weight that increases during training.
    Complements get_fun - starts low, increases as training progresses.
    
    Args:
        epoch: current epoch
        max_epoch: total number of epochs
        weight_type: same types as get_fun
        
    Returns:
        weight for CE loss (increases from 0 to 1)
    """
    return 1.0 - get_fun(epoch, max_epoch, weight_type)


class WeightScheduler:
    """
    Dynamic weight scheduler for GGCL training
    Controls contrastive vs supervised loss balance
    """
    def __init__(self, max_epoch=200, weight_type='linear', 
                 initial_contra_weight=1.0, final_contra_weight=0.0):
        self.max_epoch = max_epoch
        self.weight_type = weight_type
        self.initial_weight = initial_contra_weight
        self.final_weight = final_contra_weight
    
    def get_weights(self, epoch):
        """
        Get both contrastive and CE weights for given epoch
        
        Returns:
            contra_weight: weight for contrastive loss
            ce_weight: weight for cross-entropy loss
        """
        base_weight = get_fun(epoch, self.max_epoch, self.weight_type)
        
        # Scale to the specified range
        contra_weight = self.final_weight + base_weight * (self.initial_weight - self.final_weight)
        ce_weight = 1.0 - contra_weight
        
        return contra_weight, ce_weight
    
    def __call__(self, epoch):
        return self.get_weights(epoch)


# Pre-defined weight functions for specific training regimes
def warm_up_then_decay(epoch, max_epoch=200, warmup_epochs=10):
    """
    Warm-up period then decay
    For first warmup_epochs: only contrastive
    Then: gradually shift to supervised
    """
    if epoch < warmup_epochs:
        return 1.0
    else:
        return get_fun(epoch - warmup_epochs, max_epoch - warmup_epochs, 'linear')


def guided_weight(epoch, max_epoch=200):
    """
    GGCL paper-style weight scheduling
    Higher contrastive weight early, smoother transition
    """
    t = epoch / max_epoch
    # Sigmoid-like transition
    weight = 1.0 / (1.0 + np.exp(5 * (t - 0.5)))
    return weight
