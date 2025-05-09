"""
Functions for creating edge features based on region boundaries.
"""

import numpy as np
import torch
from skimage.segmentation import find_boundaries
from skimage.feature import canny

def boundary_strength(image, node_info, edge_index, normalize=True):
    """
    Computes edge features based on boundary strength between regions.
    
    Parameters
    ----------
    image : numpy.ndarray
        Input image with shape (H, W, C)
    node_info : dict
        Node information dictionary from node creation
    edge_index : torch.Tensor
        Edge index tensor with shape (2, E)
    normalize : bool, optional
        Whether to normalize features, by default True
        
    Returns
    -------
    torch.Tensor
        Edge feature tensor with shape (E, 1) containing boundary strengths
    """
    # Handle empty edge index
    if edge_index.shape[1] == 0:
        return torch.zeros((0, 1), dtype=torch.float)
    
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
    
    # Compute image edges using Canny edge detector
    edges = canny(gray, sigma=1.0)
    
    # Extract source and target node indices
    src, dst = edge_index
    
    # Compute boundary strengths based on node type
    boundary_strengths = []
    
    if 'segments' in node_info:  # Superpixel nodes
        segments = node_info['segments']
        
        # Compute segment boundaries
        segment_boundaries = find_boundaries(segments, mode='thick')
        
        # Combine with edge detection for stronger boundaries
        combined_boundaries = np.logical_or(segment_boundaries, edges)
        
        # For each edge, compute the boundary strength between the two segments
        for s, d in zip(src.tolist(), dst.tolist()):
            # Create masks for both segments
            mask_s = segments == s
            mask_d = segments == d
            
            # Dilate both masks
            from scipy.ndimage import binary_dilation
            struct = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
            
            dilated_s = binary_dilation(mask_s, structure=struct)
            dilated_d = binary_dilation(mask_d, structure=struct)
            
            # Find boundary pixels between s and d
            boundary_region = dilated_s & dilated_d
            
            # Count boundary pixels
            if np.any(boundary_region):
                # Measure boundary strength as the average edge response
                boundary_edge_response = combined_boundaries[boundary_region]
                strength = np.mean(boundary_edge_response) if np.any(boundary_edge_response) else 0
            else:
                strength = 0
            
            boundary_strengths.append(strength)
    
    elif 'patches' in node_info:  # Patch nodes
        positions = node_info['positions']
        bboxes = node_info['bboxes']
        
        # Create patch index mask
        patch_mask = np.zeros(gray.shape, dtype=int)
        
        for i, bbox in enumerate(bboxes):
            top, left, bottom, right = bbox
            patch_mask[top:bottom, left:right] = i
        
        # For each edge, estimate the boundary strength
        for s, d in zip(src.tolist(), dst.tolist()):
            # Get positions of the patches
            pos_s = positions[s]
            pos_d = positions[d]
            
            # Check if patches are adjacent
            if (abs(pos_s[0] - pos_d[0]) <= 1 and abs(pos_s[1] - pos_d[1]) <= 1):
                # Get bounding boxes
                bbox_s = bboxes[s]
                bbox_d = bboxes[d]
                
                # Determine boundary region
                top_s, left_s, bottom_s, right_s = bbox_s
                top_d, left_d, bottom_d, right_d = bbox_d
                
                # Find overlapping region
                top = max(top_s, top_d)
                bottom = min(bottom_s, bottom_d)
                left = max(left_s, left_d)
                right = min(right_s, right_d)
                
                # Check if there's a valid boundary
                if top < bottom and left < right:
                    # Extract boundary region
                    boundary_region = edges[top:bottom, left:right]
                    
                    # Measure boundary strength
                    strength = np.mean(boundary_region) if boundary_region.size > 0 else 0
                else:
                    # Patches are adjacent but don't share a boundary
                    strength = 0
            else:
                # Patches are not adjacent
                strength = 0
            
            boundary_strengths.append(strength)
    
    else:
        # For other node types, use a default value
        boundary_strengths = [0] * edge_index.shape[1]
    
    # Convert to tensor
    boundary_strengths = torch.tensor(boundary_strengths, dtype=torch.float).unsqueeze(1)
    
    # Normalize if requested
    if normalize:
        max_strength = torch.max(boundary_strengths)
        if max_strength > 0:
            boundary_strengths = boundary_strengths / max_strength
    
    return boundary_strengths

def boundary_orientation(image, node_info, edge_index, num_bins=8):
    """
    Computes edge features based on boundary orientation between regions.
    
    Parameters
    ----------
    image : numpy.ndarray
        Input image with shape (H, W, C)
    node_info : dict
        Node information dictionary from node creation
    edge_index : torch.Tensor
        Edge index tensor with shape (2, E)
    num_bins : int, optional
        Number of orientation bins, by default 8
        
    Returns
    -------
    torch.Tensor
        Edge feature tensor with shape (E, num_bins) containing orientation histograms
    """
    # Handle empty edge index
    if edge_index.shape[1] == 0:
        return torch.zeros((0, num_bins), dtype=torch.float)
    
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
    
    try:
        # Compute gradients
        from scipy import ndimage
        
        # Sobel gradients
        grad_y = ndimage.sobel(gray, axis=0)
        grad_x = ndimage.sobel(gray, axis=1)
        
        # Compute gradient magnitude and orientation
        grad_mag = np.sqrt(grad_y**2 + grad_x**2)
        grad_orientation = np.arctan2(grad_y, grad_x)
        
        # Convert orientation to degrees and shift to [0, 180)
        grad_orientation_deg = np.rad2deg(grad_orientation) % 180
    except ImportError:
        # Fallback method for gradient computation
        grad_mag = np.zeros_like(gray)
        grad_orientation_deg = np.zeros_like(gray)
    
    # Extract source and target node indices
    src, dst = edge_index
    
    # Compute orientation histograms based on node type
    orientation_histograms = []
    
    if 'segments' in node_info:  # Superpixel nodes
        segments = node_info['segments']
        
        # Compute segment boundaries
        segment_boundaries = find_boundaries(segments, mode='thin')
        
        # For each edge, compute the orientation histogram at the boundary
        for s, d in zip(src.tolist(), dst.tolist()):
            # Create masks for both segments
            mask_s = segments == s
            mask_d = segments == d
            
            # Dilate both masks
            from scipy.ndimage import binary_dilation
            struct = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
            
            dilated_s = binary_dilation(mask_s, structure=struct)
            dilated_d = binary_dilation(mask_d, structure=struct)
            
            # Find boundary pixels between s and d
            boundary_region = dilated_s & dilated_d & segment_boundaries
            
            # Compute orientation histogram
            if np.any(boundary_region):
                # Extract orientations and magnitudes at boundary
                boundary_orientations = grad_orientation_deg[boundary_region]
                boundary_magnitudes = grad_mag[boundary_region]
                
                # Compute weighted histogram
                hist, _ = np.histogram(
                    boundary_orientations,
                    bins=num_bins,
                    range=(0, 180),
                    weights=boundary_magnitudes
                )
                
                # Normalize histogram
                hist_sum = np.sum(hist)
                if hist_sum > 0:
                    hist = hist / hist_sum
            else:
                # No boundary found
                hist = np.zeros(num_bins)
            
            orientation_histograms.append(hist)
    
    elif 'patches' in node_info:  # Patch nodes
        positions = node_info['positions']
        bboxes = node_info['bboxes']
        
        # For each edge, estimate the boundary orientation
        for s, d in zip(src.tolist(), dst.tolist()):
            # Get positions of the patches
            pos_s = positions[s]
            pos_d = positions[d]
            
            # Check if patches are adjacent
            if (abs(pos_s[0] - pos_d[0]) <= 1 and abs(pos_s[1] - pos_d[1]) <= 1):
                # Get bounding boxes
                bbox_s = bboxes[s]
                bbox_d = bboxes[d]
                
                # Determine boundary region
                top_s, left_s, bottom_s, right_s = bbox_s
                top_d, left_d, bottom_d, right_d = bbox_d
                
                # Find overlapping region
                top = max(top_s, top_d)
                bottom = min(bottom_s, bottom_d)
                left = max(left_s, left_d)
                right = min(right_s, right_d)
                
                # Check if there's a valid boundary
                if top < bottom and left < right:
                    # Extract boundary orientations and magnitudes
                    boundary_orientations = grad_orientation_deg[top:bottom, left:right]
                    boundary_magnitudes = grad_mag[top:bottom, left:right]
                    
                    # Compute weighted histogram
                    hist, _ = np.histogram(
                        boundary_orientations.flatten(),
                        bins=num_bins,
                        range=(0, 180),
                        weights=boundary_magnitudes.flatten()
                    )
                    
                    # Normalize histogram
                    hist_sum = np.sum(hist)
                    if hist_sum > 0:
                        hist = hist / hist_sum
                else:
                    # Patches are adjacent but don't share a boundary
                    hist = np.zeros(num_bins)
            else:
                # Patches are not adjacent
                hist = np.zeros(num_bins)
            
            orientation_histograms.append(hist)
    
    else:
        # For other node types, use a uniform distribution
        uniform_hist = np.ones(num_bins) / num_bins
        orientation_histograms = [uniform_hist] * edge_index.shape[1]
    
    # Convert to tensor
    return torch.tensor(np.array(orientation_histograms), dtype=torch.float)

def boundary_contrast(image, node_info, edge_index, normalize=True):
    """
    Computes edge features based on contrast across region boundaries.
    
    Parameters
    ----------
    image : numpy.ndarray
        Input image with shape (H, W, C)
    node_info : dict
        Node information dictionary from node creation
    edge_index : torch.Tensor
        Edge index tensor with shape (2, E)
    normalize : bool, optional
        Whether to normalize features, by default True
        
    Returns
    -------
    torch.Tensor
        Edge feature tensor with shape (E, 1) containing boundary contrast
    """
    # Handle empty edge index
    if edge_index.shape[1] == 0:
        return torch.zeros((0, 1), dtype=torch.float)
    
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
    
    # Extract source and target node indices
    src, dst = edge_index
    
    # Compute region means based on node type
    region_means = []
    
    if 'segments' in node_info:  # Superpixel nodes
        segments = node_info['segments']
        unique_segments = np.unique(segments)
        
        # Calculate mean intensity for each segment
        for i in unique_segments:
            mask = segments == i
            region = gray[mask]
            
            if region.size > 0:
                mean = np.mean(region)
            else:
                mean = 0
            
            region_means.append(mean)
        
        # Compute contrast for each edge
        contrasts = []
        
        for s, d in zip(src.tolist(), dst.tolist()):
            # Get mean intensities
            mean_s = region_means[s]
            mean_d = region_means[d]
            
            # Compute absolute difference
            contrast = abs(mean_s - mean_d)
            contrasts.append(contrast)
    
    elif 'patches' in node_info:  # Patch nodes
        patches = node_info['patches']
        
        # Calculate mean intensity for each patch
        for patch in patches:
            if patch.ndim == 3:
                # Convert to grayscale if needed
                if patch.shape[2] > 1:
                    patch_gray = np.mean(patch, axis=2)
                else:
                    patch_gray = patch[:, :, 0]
            else:
                patch_gray = patch
                
            if patch_gray.size > 0:
                mean = np.mean(patch_gray)
            else:
                mean = 0
            
            region_means.append(mean)
        
        # Compute contrast for each edge
        contrasts = []
        
        for s, d in zip(src.tolist(), dst.tolist()):
            # Get mean intensities
            mean_s = region_means[s]
            mean_d = region_means[d]
            
            # Compute absolute difference
            contrast = abs(mean_s - mean_d)
            contrasts.append(contrast)
    
    else:
        # For other node types, use default values
        contrasts = [0] * edge_index.shape[1]
    
    # Convert to tensor
    contrasts = torch.tensor(contrasts, dtype=torch.float).unsqueeze(1)
    
    # Normalize if requested
    if normalize:
        max_contrast = torch.max(contrasts)
        if max_contrast > 0:
            contrasts = contrasts / max_contrast
    
    return contrasts
