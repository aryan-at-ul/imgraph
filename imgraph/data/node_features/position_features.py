"""
Functions for extracting positional features from image regions.
"""

import numpy as np
import torch

def position_features(image, node_info):
    """
    Extracts positional features from image regions.
    
    Parameters
    ----------
    image : numpy.ndarray
        Input image with shape (H, W, C)
    node_info : dict
        Node information dictionary from node creation
        
    Returns
    -------
    torch.Tensor
        Node features tensor with shape (N, 2) containing (y, x) coordinates
    """
    # Extract centroids
    if 'centroids' in node_info:
        centroids = node_info['centroids']
    else:
        raise ValueError("Node info does not contain centroids")
    
    # Convert to tensor
    return torch.tensor(centroids, dtype=torch.float)

def normalized_position_features(image, node_info, include_distance=True):
    """
    Extracts normalized positional features from image regions.
    
    Parameters
    ----------
    image : numpy.ndarray
        Input image with shape (H, W, C)
    node_info : dict
        Node information dictionary from node creation
    include_distance : bool, optional
        Whether to include distance to center, by default True
        
    Returns
    -------
    torch.Tensor
        Node features tensor with shape (N, 2) or (N, 3) if include_distance is True
    """
    # Extract centroids
    if 'centroids' in node_info:
        centroids = node_info['centroids']
    else:
        raise ValueError("Node info does not contain centroids")
    
    # Get image dimensions
    height, width = image.shape[:2]
    
    # Normalize coordinates
    normalized = np.zeros_like(centroids, dtype=np.float32)
    normalized[:, 0] = centroids[:, 0] / height  # y-coordinate
    normalized[:, 1] = centroids[:, 1] / width   # x-coordinate
    
    # Include distance to center if requested
    if include_distance:
        # Calculate image center
        center_y, center_x = height / 2, width / 2
        
        # Calculate normalized distances
        distances = np.sqrt(
            ((centroids[:, 0] - center_y) / height) ** 2 + 
            ((centroids[:, 1] - center_x) / width) ** 2
        )
        
        # Normalize by max possible distance (corner to center)
        max_distance = np.sqrt(0.5)
        distances = distances / max_distance
        
        # Add distances as a third feature
        normalized = np.column_stack((normalized, distances))
    
    # Convert to tensor
    return torch.tensor(normalized, dtype=torch.float)

def polar_position_features(image, node_info):
    """
    Extracts polar position features from image regions.
    
    Parameters
    ----------
    image : numpy.ndarray
        Input image with shape (H, W, C)
    node_info : dict
        Node information dictionary from node creation
        
    Returns
    -------
    torch.Tensor
        Node features tensor with shape (N, 2) containing (r, theta) coordinates
    """
    # Extract centroids
    if 'centroids' in node_info:
        centroids = node_info['centroids']
    else:
        raise ValueError("Node info does not contain centroids")
    
    # Get image dimensions
    height, width = image.shape[:2]
    
    # Calculate image center
    center_y, center_x = height / 2, width / 2
    
    # Calculate polar coordinates
    polar = np.zeros((centroids.shape[0], 2), dtype=np.float32)
    
    # Calculate normalized distances (r)
    polar[:, 0] = np.sqrt(
        ((centroids[:, 0] - center_y) / height) ** 2 + 
        ((centroids[:, 1] - center_x) / width) ** 2
    )
    
    # Normalize by max possible distance (corner to center)
    max_distance = np.sqrt(0.5)
    polar[:, 0] = polar[:, 0] / max_distance
    
    # Calculate angles (theta)
    polar[:, 1] = np.arctan2(
        centroids[:, 0] - center_y,
        centroids[:, 1] - center_x
    )
    
    # Normalize angles to [0, 1]
    polar[:, 1] = (polar[:, 1] + np.pi) / (2 * np.pi)
    
    # Convert to tensor
    return torch.tensor(polar, dtype=torch.float)

def grid_position_features(image, node_info, grid_size=(4, 4)):
    """
    Extracts grid-based position features using one-hot encoding.
    
    Parameters
    ----------
    image : numpy.ndarray
        Input image with shape (H, W, C)
    node_info : dict
        Node information dictionary from node creation
    grid_size : tuple, optional
        Size of the grid (rows, cols), by default (4, 4)
        
    Returns
    -------
    torch.Tensor
        Node features tensor with shape (N, grid_size[0] * grid_size[1])
    """
    # Extract centroids
    if 'centroids' in node_info:
        centroids = node_info['centroids']
    else:
        raise ValueError("Node info does not contain centroids")
    
    # Get image dimensions
    height, width = image.shape[:2]
    
    # Calculate grid cell size
    cell_height = height / grid_size[0]
    cell_width = width / grid_size[1]
    
    # Initialize features
    features = np.zeros((centroids.shape[0], grid_size[0] * grid_size[1]), dtype=np.float32)
    
    # Assign nodes to grid cells
    for i, (y, x) in enumerate(centroids):
        # Calculate grid indices
        grid_y = min(int(y / cell_height), grid_size[0] - 1)
        grid_x = min(int(x / cell_width), grid_size[1] - 1)
        
        # Calculate linear index
        linear_idx = grid_y * grid_size[1] + grid_x
        
        # Set the corresponding feature to 1
        features[i, linear_idx] = 1.0
    
    # Convert to tensor
    return torch.tensor(features, dtype=torch.float)
