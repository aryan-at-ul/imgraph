"""
Functions for extracting color features from image regions.
"""

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

def mean_color_features(image, node_info, color_space='rgb'):
    """
    Extracts mean color features from image regions.
    
    Parameters
    ----------
    image : numpy.ndarray
        Input image with shape (H, W, C)
    node_info : dict
        Node information dictionary from node creation
    color_space : str, optional
        Color space to use for feature extraction, by default 'rgb'.
        Options: 'rgb', 'hsv', 'lab', 'ycrcb'
        
    Returns
    -------
    torch.Tensor
        Node features tensor with shape (N, C) where C is the number of color channels
    """
    # Convert color space if necessary
    try:
        import cv2
    except ImportError:
        raise ImportError("OpenCV (cv2) is required for color space conversion.")
    
    if color_space.lower() != 'rgb':
        if color_space.lower() == 'hsv':
            img_converted = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        elif color_space.lower() == 'lab':
            img_converted = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        elif color_space.lower() == 'ycrcb':
            img_converted = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        else:
            raise ValueError(f"Unsupported color space: {color_space}")
    else:
        img_converted = image.copy()
    
    # Extract features based on node type
    features = []
    
    if 'segments' in node_info:  # Superpixel nodes
        segments = node_info['segments']
        unique_segments = np.unique(segments)
        
        for i in unique_segments:
            mask = segments == i
            region = img_converted[mask]
            
            if len(region) > 0:
                mean_values = np.mean(region, axis=0)
            else:
                mean_values = np.zeros(img_converted.shape[-1])
                
            features.append(mean_values)
    
    elif 'patches' in node_info:  # Patch nodes
        patches = node_info['patches']
        
        for patch in patches:
            mean_values = np.mean(patch, axis=(0, 1))
            features.append(mean_values)
    
    elif 'values' in node_info:  # Pixel nodes
        # For pixel nodes, values are already stored
        features = node_info['values']
    
    # Convert to tensor
    return torch.tensor(np.array(features), dtype=torch.float)

def mean_std_color_features(image, node_info, color_space='rgb', normalize=True):
    """
    Extracts mean and standard deviation color features from image regions.
    
    Parameters
    ----------
    image : numpy.ndarray
        Input image with shape (H, W, C)
    node_info : dict
        Node information dictionary from node creation
    color_space : str, optional
        Color space to use for feature extraction, by default 'rgb'.
        Options: 'rgb', 'hsv', 'lab', 'ycrcb'
    normalize : bool, optional
        Whether to normalize features, by default True
        
    Returns
    -------
    torch.Tensor
        Node features tensor with shape (N, 2*C) where C is the number of color channels
    """
    # Convert color space if necessary
    try:
        import cv2
    except ImportError:
        raise ImportError("OpenCV (cv2) is required for color space conversion.")
    
    if color_space.lower() != 'rgb':
        if color_space.lower() == 'hsv':
            img_converted = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        elif color_space.lower() == 'lab':
            img_converted = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        elif color_space.lower() == 'ycrcb':
            img_converted = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        else:
            raise ValueError(f"Unsupported color space: {color_space}")
    else:
        img_converted = image.copy()
    
    # Extract features based on node type
    features = []
    
    if 'segments' in node_info:  # Superpixel nodes
        segments = node_info['segments']
        unique_segments = np.unique(segments)
        
        for i in unique_segments:
            mask = segments == i
            region = img_converted[mask]
            
            if len(region) > 0:
                mean_values = np.mean(region, axis=0)
                std_values = np.std(region, axis=0)
            else:
                mean_values = np.zeros(img_converted.shape[-1])
                std_values = np.zeros(img_converted.shape[-1])
                
            # Concatenate mean and std
            features.append(np.concatenate([mean_values, std_values]))
    
    elif 'patches' in node_info:  # Patch nodes
        patches = node_info['patches']
        
        for patch in patches:
            mean_values = np.mean(patch, axis=(0, 1))
            std_values = np.std(patch, axis=(0, 1))
            
            # Concatenate mean and std
            features.append(np.concatenate([mean_values, std_values]))
    
    elif 'values' in node_info:  # Pixel nodes
        # For pixel nodes, we can't compute std, so we just duplicate the values
        values = node_info['values']
        
        for value in values:
            features.append(np.concatenate([value, np.zeros_like(value)]))
    
    # Convert to numpy array
    features = np.array(features)
    
    # Normalize if requested
    if normalize:
        scaler = StandardScaler()
        features = scaler.fit_transform(features)
    
    # Convert to tensor
    return torch.tensor(features, dtype=torch.float)

def histogram_color_features(image, node_info, bins=8, color_space='rgb', normalize=True):
    """
    Extracts color histogram features from image regions.
    
    Parameters
    ----------
    image : numpy.ndarray
        Input image with shape (H, W, C)
    node_info : dict
        Node information dictionary from node creation
    bins : int, optional
        Number of bins per channel, by default 8
    color_space : str, optional
        Color space to use for feature extraction, by default 'rgb'.
        Options: 'rgb', 'hsv', 'lab', 'ycrcb'
    normalize : bool, optional
        Whether to normalize histograms, by default True
        
    Returns
    -------
    torch.Tensor
        Node features tensor with shape (N, bins*C) where C is the number of color channels
    """
    # Convert color space if necessary
    try:
        import cv2
    except ImportError:
        raise ImportError("OpenCV (cv2) is required for color space conversion.")
    
    if color_space.lower() != 'rgb':
        if color_space.lower() == 'hsv':
            img_converted = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        elif color_space.lower() == 'lab':
            img_converted = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        elif color_space.lower() == 'ycrcb':
            img_converted = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        else:
            raise ValueError(f"Unsupported color space: {color_space}")
    else:
        img_converted = image.copy()
    
    # Define histogram bins
    num_channels = img_converted.shape[-1]
    channels = list(range(num_channels))
    hist_bins = [bins] * num_channels
    hist_ranges = [(0, 256)] * num_channels if img_converted.dtype == np.uint8 else [(0, 1)] * num_channels
    
    # Extract features based on node type
    features = []
    
    if 'segments' in node_info:  # Superpixel nodes
        segments = node_info['segments']
        unique_segments = np.unique(segments)
        
        for i in unique_segments:
            mask = segments == i
            
            # Create masked image for histogram calculation
            masked_img = np.zeros_like(img_converted)
            masked_img[mask] = img_converted[mask]
            
            # Calculate histogram for each channel
            region_hist = []
            for c in range(num_channels):
                hist, _ = np.histogram(
                    img_converted[mask, c].flatten(), 
                    bins=bins, 
                    range=hist_ranges[c], 
                    density=normalize
                )
                region_hist.extend(hist)
            
            features.append(region_hist)
    
    elif 'patches' in node_info:  # Patch nodes
        patches = node_info['patches']
        
        for patch in patches:
            # Calculate histogram for each channel
            patch_hist = []
            for c in range(num_channels):
                hist, _ = np.histogram(
                    patch[:, :, c].flatten(), 
                    bins=bins, 
                    range=hist_ranges[c], 
                    density=normalize
                )
                patch_hist.extend(hist)
            
            features.append(patch_hist)
    
    elif 'values' in node_info:  # Pixel nodes
        # For pixel nodes, we can't compute histograms
        # Instead, we return one-hot encoded bins
        values = node_info['values']
        
        for value in values:
            pixel_hist = []
            for c in range(num_channels):
                bin_idx = min(int(value[c] * bins) if img_converted.dtype != np.uint8 else int(value[c] / 256 * bins), bins - 1)
                one_hot = np.zeros(bins)
                one_hot[bin_idx] = 1
                pixel_hist.extend(one_hot)
            
            features.append(pixel_hist)
    
    # Convert to tensor
    return torch.tensor(np.array(features), dtype=torch.float)
