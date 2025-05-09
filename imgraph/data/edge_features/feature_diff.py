"""
Functions for creating edge features based on node feature differences.
"""

import numpy as np
import torch

def feature_difference(node_features, edge_index, normalize=True, mode='absolute'):
    """
    Computes edge features as differences between node features.
    
    Parameters
    ----------
    node_features : torch.Tensor
        Node feature tensor with shape (N, F)
    edge_index : torch.Tensor
        Edge index tensor with shape (2, E)
    normalize : bool, optional
        Whether to normalize features, by default True
    mode : str, optional
        Difference mode, by default 'absolute'
        Options: 'absolute', 'squared', 'relative', 'concat'
        
    Returns
    -------
    torch.Tensor
        Edge feature tensor with shape (E, F) or (E, 2*F) if mode='concat'
    """
    if edge_index.shape[1] == 0:
        if mode == 'concat':
            return torch.zeros((0, 2 * node_features.shape[1]), dtype=torch.float)
        else:
            return torch.zeros((0, node_features.shape[1]), dtype=torch.float)
    
    # Extract source and target node indices
    src, dst = edge_index
    
    # Extract source and target node features
    src_features = node_features[src]
    dst_features = node_features[dst]
    
    # Compute differences
    if mode == 'absolute':
        diff = torch.abs(src_features - dst_features)
    elif mode == 'squared':
        diff = (src_features - dst_features) ** 2
    elif mode == 'relative':
        # Avoid division by zero
        epsilon = 1e-10
        diff = torch.abs(src_features - dst_features) / (torch.abs(src_features) + torch.abs(dst_features) + epsilon)
    elif mode == 'concat':
        # Concatenate source and target features
        diff = torch.cat([src_features, dst_features], dim=1)
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    
    # Normalize if requested
    if normalize and mode != 'concat':
        # Compute max value for each feature dimension
        max_values, _ = torch.max(diff, dim=0, keepdim=True)
        max_values[max_values == 0] = 1.0  # Avoid division by zero
        
        # Normalize
        diff = diff / max_values
    
    return diff

def color_difference(image, node_info, edge_index, color_space='rgb', normalize=True):
    """
    Computes edge features as color differences between nodes.
    
    Parameters
    ----------
    image : numpy.ndarray
        Input image with shape (H, W, C)
    node_info : dict
        Node information dictionary from node creation
    edge_index : torch.Tensor
        Edge index tensor with shape (2, E)
    color_space : str, optional
        Color space to use, by default 'rgb'
        Options: 'rgb', 'hsv', 'lab'
    normalize : bool, optional
        Whether to normalize features, by default True
        
    Returns
    -------
    torch.Tensor
        Edge feature tensor with shape (E, C)
    """
    # Extract mean colors per node
    colors = []
    
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
    
    if 'segments' in node_info:  # Superpixel nodes
        segments = node_info['segments']
        unique_segments = np.unique(segments)
        
        for i in unique_segments:
            mask = segments == i
            if np.any(mask):
                region = img_converted[mask]
                mean_color = np.mean(region, axis=0)
                colors.append(mean_color)
            else:
                # Handle empty segments (should not happen)
                colors.append(np.zeros(img_converted.shape[-1]))
    
    elif 'patches' in node_info:  # Patch nodes
        patches = node_info['patches']
        
        for patch in patches:
            mean_color = np.mean(patch, axis=(0, 1))
            colors.append(mean_color)
    
    elif 'values' in node_info:  # Pixel nodes
        colors = node_info['values']
    
    else:
        raise ValueError("Unsupported node type")
    
    # Convert to tensor
    node_colors = torch.tensor(np.array(colors), dtype=torch.float)
    
    # Compute feature differences
    return feature_difference(node_colors, edge_index, normalize=normalize, mode='absolute')

def normalized_color_difference(image, node_info, edge_index, color_space='lab'):
    """
    Computes normalized color differences between nodes, optimized for perceptual similarity.
    
    Parameters
    ----------
    image : numpy.ndarray
        Input image with shape (H, W, C)
    node_info : dict
        Node information dictionary from node creation
    edge_index : torch.Tensor
        Edge index tensor with shape (2, E)
    color_space : str, optional
        Color space to use, by default 'lab'
        
    Returns
    -------
    torch.Tensor
        Edge feature tensor with shape (E, 1) containing perceptual color differences
    """
    # For perceptual color difference, LAB color space is best
    try:
        import cv2
    except ImportError:
        raise ImportError("OpenCV (cv2) is required for color space conversion.")
    
    # Convert to LAB color space
    if color_space.lower() == 'lab':
        img_converted = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    else:
        # Use requested color space
        if color_space.lower() == 'hsv':
            img_converted = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        elif color_space.lower() == 'ycrcb':
            img_converted = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        elif color_space.lower() == 'rgb':
            img_converted = image.copy()
        else:
            raise ValueError(f"Unsupported color space: {color_space}")
    
    # Extract mean colors per node
    colors = []
    
    if 'segments' in node_info:  # Superpixel nodes
        segments = node_info['segments']
        unique_segments = np.unique(segments)
        
        for i in unique_segments:
            mask = segments == i
            if np.any(mask):
                region = img_converted[mask]
                mean_color = np.mean(region, axis=0)
                colors.append(mean_color)
            else:
                colors.append(np.zeros(img_converted.shape[-1]))
    
    elif 'patches' in node_info:  # Patch nodes
        patches = node_info['patches']
        bboxes = node_info['bboxes']
        
        for i, bbox in enumerate(bboxes):
            top, left, bottom, right = bbox
            region = img_converted[top:bottom, left:right]
            
            if region.size > 0:
                mean_color = np.mean(region, axis=(0, 1))
                colors.append(mean_color)
            else:
                colors.append(np.zeros(img_converted.shape[-1]))
    
    elif 'values' in node_info:  # Pixel nodes
        colors = node_info['values']
    
    else:
        raise ValueError("Unsupported node type")
    
    # Convert to tensor
    node_colors = torch.tensor(np.array(colors), dtype=torch.float)
    
    # Extract source and target node indices
    if edge_index.shape[1] == 0:
        return torch.zeros((0, 1), dtype=torch.float)
    
    src, dst = edge_index
    
    # Extract source and target node colors
    src_colors = node_colors[src]
    dst_colors = node_colors[dst]
    
    # Compute color differences
    if color_space.lower() == 'lab':
        # Delta E in CIELAB space
        L1, a1, b1 = src_colors[:, 0], src_colors[:, 1], src_colors[:, 2]
        L2, a2, b2 = dst_colors[:, 0], dst_colors[:, 1], dst_colors[:, 2]
        
        # Delta E formula
        delta_E = torch.sqrt((L2 - L1) ** 2 + (a2 - a1) ** 2 + (b2 - b1) ** 2)
        
        # Normalize
        delta_E = delta_E / 100.0  # Typical range of Delta E is 0-100
        
        return delta_E.unsqueeze(1)
    else:
        # Use Euclidean distance for other color spaces
        diff = torch.sqrt(torch.sum((dst_colors - src_colors) ** 2, dim=1))
        
        # Normalize
        max_diff = torch.max(diff)
        if max_diff > 0:
            diff = diff / max_diff
        
        return diff.unsqueeze(1)
