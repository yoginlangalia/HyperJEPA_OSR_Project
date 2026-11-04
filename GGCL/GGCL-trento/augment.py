"""
Data Augmentation for GGCL Contrastive Learning
Uses Odd-Even spectral masking to create two views for contrastive learning
(This is more appropriate for HSI data than generic image augmentation)
"""

import torch


def augment_odd_even(data):
    """
    Spectral augmentation by odd-even channel masking.
    Creates two complementary views by masking alternate spectral bands.
    
    This is more appropriate for HSI data as it creates spectrally-distinct views
    while preserving spatial information.
    
    Args:
        data: (batch, C, H, W) tensor where C is spectral channels
        
    Returns:
        odd_view: tensor with even channels zeroed (keeps odd: 1, 3, 5, ...)
        even_view: tensor with odd channels zeroed (keeps even: 0, 2, 4, ...)
    """
    B, C, H, W = data.shape
    
    # Odd channels: 1, 3, 5, ... (0-indexed)
    odd_indices = list(range(1, C, 2))
    # Even channels: 0, 2, 4, ...
    even_indices = list(range(0, C, 2))
    
    # Create views by zeroing out alternate channels
    odd_view = data.clone()
    even_view = data.clone()
    
    # Odd view: zero out even channels (keep odd)
    odd_view[:, even_indices, :, :] = 0
    # Even view: zero out odd channels (keep even)
    even_view[:, odd_indices, :, :] = 0
    
    return odd_view, even_view


def do_augment(augment_type, data):
    """
    Apply augmentation to create two different views for contrastive learning.
    
    Args:
        augment_type: 'odd_even' (recommended for HSI) or 'flip_noise'
        data: input tensor (batch, spectral, width, height)
        
    Returns:
        left_data, right_data: two augmented views
    """
    if augment_type is None:
        return data.clone(), data.clone()
    
    if augment_type == 'odd_even' or augment_type == ['odd_even']:
        return augment_odd_even(data)
    
    # Default to odd-even for HSI data
    return augment_odd_even(data)


class ContrastiveAugmentation:
    """
    Class-based augmentation for more control
    """
    def __init__(self, augmentation_type='odd_even'):
        self.augmentation_type = augmentation_type
    
    def __call__(self, data):
        return do_augment(self.augmentation_type, data)
