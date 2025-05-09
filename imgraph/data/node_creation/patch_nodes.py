"""
Functions for creating graph nodes based on regular image patches.
"""

import numpy as np

def regular_patch_nodes(image, patch_size=16, stride=None):
    """
    Creates graph nodes by dividing the image into regular patches.
    
    Parameters
    ----------
    image : numpy.ndarray
        Input image with shape (H, W, C)
    patch_size : int or tuple, optional
        Size of the patches to extract, by default 16.
        If int, square patches of size (patch_size, patch_size) are extracted.
        If tuple, patches of size (patch_size[0], patch_size[1]) are extracted.
    stride : int or tuple, optional
        Stride between patches, by default None (equal to patch_size).
        If int, stride of (stride, stride) is used.
        If tuple, stride of (stride[0], stride[1]) is used.
        
    Returns
    -------
    dict
        Dictionary containing node information:
        - 'patches' : list of image patches
        - 'centroids' : node centroid coordinates (N, 2)
        - 'bboxes' : bounding boxes for each patch (N, 4) as (min_row, min_col, max_row, max_col)
        - 'positions' : grid positions of each patch (N, 2) as (row_idx, col_idx)
    """
    # Handle different patch_size and stride formats
    if isinstance(patch_size, int):
        patch_size = (patch_size, patch_size)
    
    if stride is None:
        stride = patch_size
    elif isinstance(stride, int):
        stride = (stride, stride)
    
    # Get image dimensions
    height, width = image.shape[:2]
    
    # Calculate number of patches in each dimension
    n_patches_h = 1 + (height - patch_size[0]) // stride[0]
    n_patches_w = 1 + (width - patch_size[1]) // stride[1]
    
    # Initialize variables
    patches = []
    centroids = []
    bboxes = []
    positions = []
    
    # Extract patches
    for i in range(n_patches_h):
        for j in range(n_patches_w):
            # Calculate patch coordinates
            top = i * stride[0]
            left = j * stride[1]
            bottom = min(top + patch_size[0], height)
            right = min(left + patch_size[1], width)
            
            # Extract patch
            patch = image[top:bottom, left:right]
            
            # Handle boundary patches (resize if needed)
            if patch.shape[0] != patch_size[0] or patch.shape[1] != patch_size[1]:
                # Skip incomplete patches option 1:
                # continue
                
                # Use partial patches option 2:
                pass
            
            # Calculate centroid
            centroid = (top + (bottom - top) / 2, left + (right - left) / 2)
            
            # Store patch information
            patches.append(patch)
            centroids.append(centroid)
            bboxes.append((top, left, bottom, right))
            positions.append((i, j))
    
    return {
        'patches': patches,
        'centroids': np.array(centroids),
        'bboxes': np.array(bboxes),
        'positions': np.array(positions)
    }