"""
Functions for creating graph nodes from superpixel segmentation.
"""

import numpy as np
from skimage.segmentation import slic, felzenszwalb
from skimage.measure import regionprops
import torch

def slic_superpixel_nodes(image, n_segments=100, compactness=10, sigma=0):
    """
    Creates graph nodes using SLIC superpixel segmentation.
    
    Parameters
    ----------
    image : numpy.ndarray
        Input image with shape (H, W, C)
    n_segments : int, optional
        Number of segments to extract, by default 100
    compactness : float, optional
        Compactness parameter for SLIC, by default 10
    sigma : float, optional
        Width of Gaussian smoothing kernel for pre-processing, by default 0
        
    Returns
    -------
    dict
        Dictionary containing node information:
        - 'segments' : superpixel segmentation map
        - 'centroids' : node centroid coordinates (N, 2)
        - 'masks' : binary masks for each superpixel
        - 'bboxes' : bounding boxes for each superpixel (N, 4) as (min_row, min_col, max_row, max_col)
    """
    # Ensure image is in the right format
    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255.0
    
    # Apply SLIC segmentation
    segments = slic(image, n_segments=n_segments, compactness=compactness, 
                    sigma=sigma, start_label=0)
    
    # Extract region properties
    regions = regionprops(segments + 1)  # Add 1 to avoid background=0 issues
    
    # Extract centroid coordinates
    centroids = np.array([region.centroid for region in regions])
    
    # Create masks for each superpixel
    unique_segments = np.unique(segments)
    masks = []
    bboxes = []
    
    for i in unique_segments:
        mask = segments == i
        masks.append(mask)
        
        # Get bounding box
        props = regionprops(mask.astype(int))
        if props:
            minr, minc, maxr, maxc = props[0].bbox
            bboxes.append((minr, minc, maxr, maxc))
        else:
            # Fallback if regionprops fails
            rows, cols = np.where(mask)
            if len(rows) > 0 and len(cols) > 0:
                minr, maxr = np.min(rows), np.max(rows)
                minc, maxc = np.min(cols), np.max(cols)
                bboxes.append((minr, minc, maxr, maxc))
            else:
                bboxes.append((0, 0, 1, 1))  # Fallback bbox
    
    return {
        'segments': segments,
        'centroids': centroids,
        'masks': masks,
        'bboxes': bboxes
    }

def felzenszwalb_superpixel_nodes(image, scale=100, sigma=0.5, min_size=50):
    """
    Creates graph nodes using Felzenszwalb's superpixel segmentation.
    
    Parameters
    ----------
    image : numpy.ndarray
        Input image with shape (H, W, C)
    scale : float, optional
        Scale parameter for Felzenszwalb, by default 100
    sigma : float, optional
        Width of Gaussian smoothing kernel for pre-processing, by default 0.5
    min_size : int, optional
        Minimum component size, by default 50
        
    Returns
    -------
    dict
        Dictionary containing node information:
        - 'segments' : superpixel segmentation map
        - 'centroids' : node centroid coordinates (N, 2)
        - 'masks' : binary masks for each superpixel
        - 'bboxes' : bounding boxes for each superpixel (N, 4) as (min_row, min_col, max_row, max_col)
    """
    # Ensure image is in the right format
    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255.0
    
    # Apply Felzenszwalb segmentation
    segments = felzenszwalb(image, scale=scale, sigma=sigma, min_size=min_size)
    
    # Extract region properties
    regions = regionprops(segments + 1)  # Add 1 to avoid background=0 issues
    
    # Extract centroid coordinates
    centroids = np.array([region.centroid for region in regions])
    
    # Create masks for each superpixel
    unique_segments = np.unique(segments)
    masks = []
    bboxes = []
    
    for i in unique_segments:
        mask = segments == i
        masks.append(mask)
        
        # Get bounding box
        props = regionprops(mask.astype(int))
        if props:
            minr, minc, maxr, maxc = props[0].bbox
            bboxes.append((minr, minc, maxr, maxc))
        else:
            # Fallback if regionprops fails
            rows, cols = np.where(mask)
            if len(rows) > 0 and len(cols) > 0:
                minr, maxr = np.min(rows), np.max(rows)
                minc, maxc = np.min(cols), np.max(cols)
                bboxes.append((minr, minc, maxr, maxc))
            else:
                bboxes.append((0, 0, 1, 1))  # Fallback bbox
    
    return {
        'segments': segments,
        'centroids': centroids,
        'masks': masks,
        'bboxes': bboxes
    }