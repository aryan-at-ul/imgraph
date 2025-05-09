"""
Functions for extracting texture features from image regions.
"""

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

def lbp_features(image, node_info, radius=1, n_points=8, normalize=True):
    """
    Extracts Local Binary Pattern (LBP) features from image regions.
    
    Parameters
    ----------
    image : numpy.ndarray
        Input image with shape (H, W, C)
    node_info : dict
        Node information dictionary from node creation
    radius : int, optional
        Radius of the LBP circle, by default 1
    n_points : int, optional
        Number of points around the circle, by default 8
    normalize : bool, optional
        Whether to normalize histograms, by default True
        
    Returns
    -------
    torch.Tensor
        Node features tensor with shape (N, 2^n_points)
    """
    try:
        from skimage.feature import local_binary_pattern
    except ImportError:
        raise ImportError("scikit-image is required for LBP feature extraction.")
    
    # Convert to grayscale if needed
    if len(image.shape) == 3 and image.shape[2] > 1:
        try:
            import cv2
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        except ImportError:
            # Simple grayscale conversion as fallback
            gray = np.mean(image, axis=2)
    else:
        gray = image
    
    # Ensure image is in float format
    if gray.dtype == np.uint8:
        gray = gray.astype(np.float32) / 255.0
    
    # Compute LBP texture
    lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
    
    # Number of bins - uniform LBP has n_points + 2 patterns
    n_bins = n_points + 2
    
    # Extract features based on node type
    features = []
    
    if 'segments' in node_info:  # Superpixel nodes
        segments = node_info['segments']
        unique_segments = np.unique(segments)
        
        for i in unique_segments:
            mask = segments == i
            region_lbp = lbp[mask]
            
            if len(region_lbp) > 0:
                hist, _ = np.histogram(region_lbp, bins=n_bins, range=(0, n_bins), density=normalize)
            else:
                hist = np.zeros(n_bins)
                
            features.append(hist)
    
    elif 'patches' in node_info:  # Patch nodes
        patches = node_info['patches']
        bboxes = node_info['bboxes']
        
        for bbox in bboxes:
            top, left, bottom, right = bbox
            patch_lbp = lbp[top:bottom, left:right]
            
            if patch_lbp.size > 0:
                hist, _ = np.histogram(patch_lbp, bins=n_bins, range=(0, n_bins), density=normalize)
            else:
                hist = np.zeros(n_bins)
                
            features.append(hist)
    
    elif 'values' in node_info:  # Pixel nodes
        # For pixel nodes, we can't compute texture features
        # Just return zeros
        positions = node_info['positions']
        features = np.zeros((len(positions), n_bins))
    
    # Convert to tensor
    features = np.array(features)
    
    # Normalize if requested
    if normalize:
        features = features / (np.sum(features, axis=1, keepdims=True) + 1e-10)
    
    return torch.tensor(features, dtype=torch.float)

def glcm_features(image, node_info, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], normalize=True):
    """
    Extracts Gray-Level Co-occurrence Matrix (GLCM) features from image regions.
    
    Parameters
    ----------
    image : numpy.ndarray
        Input image with shape (H, W, C)
    node_info : dict
        Node information dictionary from node creation
    distances : list, optional
        List of distances for GLCM, by default [1]
    angles : list, optional
        List of angles for GLCM, by default [0, np.pi/4, np.pi/2, 3*np.pi/4]
    normalize : bool, optional
        Whether to normalize features, by default True
        
    Returns
    -------
    torch.Tensor
        Node features tensor with shape (N, F) where F depends on the properties computed
    """
    try:
        from skimage.feature import graycomatrix, graycoprops
    except ImportError:
        raise ImportError("scikit-image is required for GLCM feature extraction.")
    
    # Convert to grayscale if needed
    if len(image.shape) == 3 and image.shape[2] > 1:
        try:
            import cv2
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        except ImportError:
            # Simple grayscale conversion as fallback
            gray = np.mean(image, axis=2)
    else:
        gray = image
    
    # Ensure image is in uint8 format for GLCM
    if gray.dtype != np.uint8:
        gray = (gray * 255).astype(np.uint8)
    
    # Properties to compute
    properties = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
    
    # Extract features based on node type
    features = []
    
    if 'segments' in node_info:  # Superpixel nodes
        segments = node_info['segments']
        unique_segments = np.unique(segments)
        
        for i in unique_segments:
            mask = segments == i
            
            # Get bounding box of segment for efficiency
            rows, cols = np.where(mask)
            if len(rows) == 0 or len(cols) == 0:
                # Empty segment, add zeros
                feature_vector = np.zeros(len(properties) * len(distances) * len(angles))
                features.append(feature_vector)
                continue
                
            rmin, rmax = np.min(rows), np.max(rows)
            cmin, cmax = np.min(cols), np.max(cols)
            
            # Create binary mask for the region
            region_mask = mask[rmin:rmax+1, cmin:cmax+1]
            
            # Extract the region
            region = gray[rmin:rmax+1, cmin:cmax+1].copy()
            
            # Set pixels outside the region to 0
            region[~region_mask] = 0
            
            # Skip tiny regions
            if region.shape[0] < 2 or region.shape[1] < 2:
                feature_vector = np.zeros(len(properties) * len(distances) * len(angles))
                features.append(feature_vector)
                continue
            
            # Compute GLCM
            try:
                glcm = graycomatrix(region, distances=distances, angles=angles, levels=256, symmetric=True, normed=True)
                
                # Compute properties
                feature_vector = []
                for prop in properties:
                    feature_vector.extend(graycoprops(glcm, prop).flatten())
                
                features.append(feature_vector)
            except:
                # Fallback if GLCM computation fails
                feature_vector = np.zeros(len(properties) * len(distances) * len(angles))
                features.append(feature_vector)
    
    elif 'patches' in node_info:  # Patch nodes
        bboxes = node_info['bboxes']
        
        for bbox in bboxes:
            top, left, bottom, right = bbox
            patch = gray[top:bottom, left:right]
            
            # Skip tiny patches
            if patch.shape[0] < 2 or patch.shape[1] < 2:
                feature_vector = np.zeros(len(properties) * len(distances) * len(angles))
                features.append(feature_vector)
                continue
            
            # Compute GLCM
            try:
                glcm = graycomatrix(patch, distances=distances, angles=angles, levels=256, symmetric=True, normed=True)
                
                # Compute properties
                feature_vector = []
                for prop in properties:
                    feature_vector.extend(graycoprops(glcm, prop).flatten())
                
                features.append(feature_vector)
            except:
                # Fallback if GLCM computation fails
                feature_vector = np.zeros(len(properties) * len(distances) * len(angles))
                features.append(feature_vector)
    
    elif 'values' in node_info:  # Pixel nodes
        # For pixel nodes, we can't compute texture features
        # Just return zeros
        positions = node_info['positions']
        features = np.zeros((len(positions), len(properties) * len(distances) * len(angles)))
    
    # Convert to numpy array
    features = np.array(features)
    
    # Normalize if requested
    if normalize:
        scaler = StandardScaler()
        features = scaler.fit_transform(features)
    
    # Convert to tensor
    return torch.tensor(features, dtype=torch.float)

def gabor_features(image, node_info, frequencies=[0.1, 0.25, 0.5], orientations=[0, np.pi/4, np.pi/2, 3*np.pi/4], normalize=True):
    """
    Extracts Gabor filter features from image regions.
    
    Parameters
    ----------
    image : numpy.ndarray
        Input image with shape (H, W, C)
    node_info : dict
        Node information dictionary from node creation
    frequencies : list, optional
        List of frequencies for Gabor filters, by default [0.1, 0.25, 0.5]
    orientations : list, optional
        List of orientations for Gabor filters, by default [0, np.pi/4, np.pi/2, 3*np.pi/4]
    normalize : bool, optional
        Whether to normalize features, by default True
        
    Returns
    -------
    torch.Tensor
        Node features tensor with shape (N, len(frequencies)*len(orientations)*2)
    """
    try:
        from skimage.filters import gabor_kernel
        from scipy import ndimage as ndi
    except ImportError:
        raise ImportError("scikit-image and scipy are required for Gabor feature extraction.")
    
    # Convert to grayscale if needed
    if len(image.shape) == 3 and image.shape[2] > 1:
        try:
            import cv2
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        except ImportError:
            # Simple grayscale conversion as fallback
            gray = np.mean(image, axis=2)
    else:
        gray = image
    
    # Ensure image is in float format
    if gray.dtype == np.uint8:
        gray = gray.astype(np.float32) / 255.0
    
    # Generate Gabor kernels
    kernels = []
    for frequency in frequencies:
        for orientation in orientations:
            kernel = gabor_kernel(frequency, theta=orientation)
            kernels.append(kernel)
    
    # Apply Gabor filters to full image
    gabor_responses = []
    for kernel in kernels:
        # Apply filter
        filtered_real = ndi.convolve(gray, np.real(kernel), mode='wrap')
        filtered_imag = ndi.convolve(gray, np.imag(kernel), mode='wrap')
        
        # Compute magnitude
        magnitude = np.sqrt(filtered_real**2 + filtered_imag**2)
        
        gabor_responses.append(magnitude)
    
    # Extract features based on node type
    features = []
    
    if 'segments' in node_info:  # Superpixel nodes
        segments = node_info['segments']
        unique_segments = np.unique(segments)
        
        for i in unique_segments:
            mask = segments == i
            
            # Compute mean and std of Gabor responses for each kernel
            feature_vector = []
            for response in gabor_responses:
                region_response = response[mask]
                
                if len(region_response) > 0:
                    mean = np.mean(region_response)
                    std = np.std(region_response)
                else:
                    mean = 0
                    std = 0
                
                feature_vector.extend([mean, std])
            
            features.append(feature_vector)
    
    elif 'patches' in node_info:  # Patch nodes
        bboxes = node_info['bboxes']
        
        for bbox in bboxes:
            top, left, bottom, right = bbox
            
            # Compute mean and std of Gabor responses for each kernel
            feature_vector = []
            for response in gabor_responses:
                patch_response = response[top:bottom, left:right]
                
                if patch_response.size > 0:
                    mean = np.mean(patch_response)
                    std = np.std(patch_response)
                else:
                    mean = 0
                    std = 0
                
                feature_vector.extend([mean, std])
            
            features.append(feature_vector)
    
    elif 'values' in node_info:  # Pixel nodes
        positions = node_info['positions']
        
        for pos in positions:
            i, j = pos
            
            # Sample Gabor responses at pixel position
            feature_vector = []
            for response in gabor_responses:
                value = response[i, j]
                # For pixel nodes, we don't have std
                feature_vector.extend([value, 0])
            
            features.append(feature_vector)
    
    # Convert to numpy array
    features = np.array(features)
    
    # Normalize if requested
    if normalize:
        scaler = StandardScaler()
        features = scaler.fit_transform(features)
    
    # Convert to tensor
    return torch.tensor(features, dtype=torch.float)
