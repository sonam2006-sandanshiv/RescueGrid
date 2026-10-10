import numpy as np
from scipy.ndimage import uniform_filter, variance

def lee_filter(img, size=3):
    """
    Fast implementation of a standard Lee filter for speckle reduction.
    img: 2D array (e.g., S1 intensity)
    size: window size
    """
    img_mean = uniform_filter(img, (size, size))
    img_sqr_mean = uniform_filter(img**2, (size, size))
    img_variance = img_sqr_mean - img_mean**2
    
    overall_variance = variance(img)
    
    # Estimate noise variance based on standard SAR assumptions
    # This is a simplification; in practice ENL (Equivalent Number of Looks) is used.
    noise_variance = img_mean**2 / (overall_variance + 1e-6) # arbitrary heuristic for noise variance
    
    # The weight
    img_variance = np.maximum(img_variance, 0.0)
    weight = img_variance / (img_variance + noise_variance + 1e-6)
    
    filtered_img = img_mean + weight * (img - img_mean)
    return filtered_img
