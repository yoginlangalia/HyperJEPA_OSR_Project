"""
PCA Module for GGCL Houston
Reduces spectral dimensionality from 144 bands to n_components
"""

import numpy as np
from sklearn.decomposition import PCA


def apply_pca(data, n_components=15):
    """
    Apply PCA to reduce spectral dimensions.
    
    Args:
        data: HSI cube (H, W, C) where C=144 spectral bands
        n_components: Number of PCA components to keep (default: 15)
        
    Returns:
        reduced_data: (H, W, n_components)
    """
    H, W, C = data.shape
    print(f"\nApplying PCA: {C} bands → {n_components} components")
    
    # Reshape to 2D: (H*W, C)
    data_reshaped = data.reshape(-1, C)
    
    # Fit PCA on all pixels (no train/test leakage concern for unsupervised PCA)
    pca = PCA(n_components=n_components, random_state=42)
    data_reduced = pca.fit_transform(data_reshaped)
    
    # Reshape back to 3D: (H, W, n_components)
    reduced_data = data_reduced.reshape(H, W, n_components)
    
    # Print explained variance
    explained_var = np.sum(pca.explained_variance_ratio_) * 100
    print(f"  Explained variance: {explained_var:.2f}%")
    print(f"  Output shape: {reduced_data.shape}")
    
    return reduced_data.astype(np.float32)
