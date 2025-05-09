"""
Functions for creating region adjacency-based edges between nodes.
"""

import numpy as np
import torch
from scipy.ndimage import find_objects
from skimage.segmentation import find_boundaries

def region_adjacency_edges(node_info, image=None, connectivity=2):
    """
    Creates edges between adjacent regions (superpixels).
    
    Parameters
    ----------
    node_info : dict
        Node information dictionary from node creation
    image : numpy.ndarray, optional
        Input image, by default None
    connectivity : int, optional
        Connectivity for determining adjacency, by default 2 (8-connected)
        Set to 1 for 4-connected adjacency.
        
    Returns
    -------
    torch.Tensor
        Edge index tensor with shape (2, E)
    """
    if 'segments' not in node_info:
        raise ValueError("Node info does not contain segments")
    
    segments = node_info['segments']
    
    try:
        # Try using skimage's RAG functionality
        from skimage.future import graph
        rag = graph.RAG(segments, connectivity=connectivity)
        
        # Convert RAG to edge list
        edge_list = []
        for u, v in rag.edges():
            # Skip self-loops
            if u != v:
                edge_list.append((u, v))
                edge_list.append((v, u))  # Make bidirectional
    except ImportError:
        # Manual implementation as fallback
        # Find boundaries
        boundaries = find_boundaries(segments, connectivity=connectivity, mode='thick')
        
        # Create edge list
        edge_list = []
        unique_segments = np.unique(segments)
        
        # For each segment, find its neighbors
        for i in unique_segments:
            # Create mask for current segment
            segment_mask = segments == i
            
            # Dilate mask to find neighbors
            from scipy.ndimage import binary_dilation
            
            # Use 4-connected or 8-connected structure
            if connectivity == 1:
                struct = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
            else:
                struct = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
            
            dilated = binary_dilation(segment_mask, structure=struct)
            
            # Find neighbors
            neighbors = np.unique(segments[dilated & ~segment_mask])
            
            # Add edges
            for j in neighbors:
                if i != j:
                    edge_list.append((i, j))
    
    # Convert to tensor
    if edge_list:
        edge_index = torch.tensor(edge_list, dtype=torch.long).t()
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    
    return edge_index

def region_boundary_edges(node_info, image=None, min_boundary_size=10):
    """
    Creates edges between regions with shared boundaries.
    Also computes boundary length as an edge attribute.
    
    Parameters
    ----------
    node_info : dict
        Node information dictionary from node creation
    image : numpy.ndarray, optional
        Input image, by default None
    min_boundary_size : int, optional
        Minimum boundary size to create an edge, by default 10
        
    Returns
    -------
    tuple
        (edge_index, edge_attr)
        - edge_index: torch.Tensor with shape (2, E)
        - edge_attr: torch.Tensor with shape (E, 1) containing boundary sizes
    """
    if 'segments' not in node_info:
        raise ValueError("Node info does not contain segments")
    
    segments = node_info['segments']
    
    # Find boundaries
    boundaries = find_boundaries(segments, mode='thick')
    
    # Create edge list and boundary sizes
    edge_list = []
    boundary_sizes = []
    
    unique_segments = np.unique(segments)
    
    # For each pair of segments, find their boundary size
    for i in unique_segments:
        for j in unique_segments:
            if i >= j:  # Skip self and avoid duplicates
                continue
            
            # Create masks for both segments
            mask_i = segments == i
            mask_j = segments == j
            
            # Dilate both masks
            from scipy.ndimage import binary_dilation
            struct = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
            
            dilated_i = binary_dilation(mask_i, structure=struct)
            dilated_j = binary_dilation(mask_j, structure=struct)
            
            # Find shared boundary pixels
            boundary_ij = boundaries & dilated_i & dilated_j
            boundary_size = np.sum(boundary_ij)
            
            # Add edge if boundary is large enough
            if boundary_size >= min_boundary_size:
                edge_list.append((i, j))
                edge_list.append((j, i))  # Make bidirectional
                boundary_sizes.append(boundary_size)
                boundary_sizes.append(boundary_size)  # Same for bidirectional
    
    # Convert to tensors
    if edge_list:
        edge_index = torch.tensor(edge_list, dtype=torch.long).t()
        edge_attr = torch.tensor(boundary_sizes, dtype=torch.float).view(-1, 1)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 1), dtype=torch.float)
    
    return edge_index, edge_attr

def superpixel_containment_edges(node_info, image=None, levels=2):
    """
    Creates hierarchical edges between superpixels at different levels.
    Useful for creating hierarchical graph representations.
    
    Parameters
    ----------
    node_info : dict
        Node information dictionary from node creation
    image : numpy.ndarray, optional
        Input image, by default None
    levels : int, optional
        Number of hierarchical levels, by default 2
        
    Returns
    -------
    torch.Tensor
        Edge index tensor with shape (2, E)
    """
    if 'segments' not in node_info:
        raise ValueError("Node info does not contain segments")
    
    try:
        from skimage.segmentation import slic
    except ImportError:
        raise ImportError("scikit-image is required for hierarchical superpixel segmentation.")
    
    segments = node_info['segments']
    
    # Get image
    if image is None:
        raise ValueError("Image is required for hierarchical segmentation")
    
    # Number of segments at the finest level
    n_segments_base = len(np.unique(segments))
    
    # Create segments at coarser levels
    hierarchical_segments = [segments]  # Start with the finest level
    
    for level in range(1, levels):
        n_segments = max(n_segments_base // (2**level), 2)  # Reduce by factor of 2 per level
        
        # Create coarser segmentation
        coarse_segments = slic(image, n_segments=n_segments, compactness=10, sigma=0, start_label=0)
        hierarchical_segments.append(coarse_segments)
    
    # Create edge list for hierarchical connections
    edge_list = []
    
    # Connect fine to coarse segments
    for level in range(levels - 1):
        fine_segments = hierarchical_segments[level]
        coarse_segments = hierarchical_segments[level + 1]
        
        # Find containment relationships
        for i in np.unique(fine_segments):
            # Find which coarse segment contains this fine segment
            fine_mask = fine_segments == i
            coarse_labels = coarse_segments[fine_mask]
            
            # Most common coarse label
            from scipy.stats import mode
            if len(coarse_labels) > 0:
                most_common = mode(coarse_labels)[0][0]
                
                # Add containment edge
                edge_list.append((i, most_common + n_segments_base))  # Offset coarse indices
                edge_list.append((most_common + n_segments_base, i))  # Bidirectional
    
    # Add regular adjacency edges for each level
    for segments in hierarchical_segments:
        try:
            from skimage.future import graph
            rag = graph.RAG(segments)
            
            # Add edges
            for u, v in rag.edges():
                if u != v:
                    edge_list.append((u, v))
                    edge_list.append((v, u))
        except ImportError:
            # Simple adjacency fallback
            pass
    
    # Convert to tensor
    if edge_list:
        edge_index = torch.tensor(edge_list, dtype=torch.long).t()
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    
    return edge_index
