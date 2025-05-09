"""
Functions for creating similarity-based edges between nodes.
"""

import numpy as np
import torch
from scipy.spatial import distance_matrix
from sklearn.metrics.pairwise import cosine_similarity

def feature_similarity_edges(node_features, threshold=0.7, metric='cosine', k=None):
    """
    Creates edges between nodes based on feature similarity.
    
    Parameters
    ----------
    node_features : torch.Tensor
        Node feature tensor with shape (N, F)
    threshold : float, optional
        Similarity threshold, by default 0.7
    metric : str, optional
        Similarity metric, by default 'cosine'
        Options: 'cosine', 'euclidean', 'correlation'
    k : int, optional
        If provided, connect each node to its k most similar neighbors,
        ignoring the threshold, by default None
        
    Returns
    -------
    torch.Tensor
        Edge index tensor with shape (2, E)
    """
    # Convert tensor to numpy if needed
    if isinstance(node_features, torch.Tensor):
        node_features_np = node_features.detach().cpu().numpy()
    else:
        node_features_np = node_features
    
    # Calculate similarity matrix
    if metric == 'cosine':
        # Higher value = more similar
        sim_matrix = cosine_similarity(node_features_np)
    elif metric == 'euclidean':
        # Lower value = more similar, so invert
        dist_matrix = distance_matrix(node_features_np, node_features_np)
        max_dist = np.max(dist_matrix) if np.max(dist_matrix) > 0 else 1
        sim_matrix = 1 - (dist_matrix / max_dist)
    elif metric == 'correlation':
        # Use correlation coefficient
        # Normalize features
        normalized = node_features_np - np.mean(node_features_np, axis=1, keepdims=True)
        norms = np.linalg.norm(normalized, axis=1, keepdims=True)
        normalized = normalized / (norms + 1e-8)
        
        # Compute correlation
        sim_matrix = np.dot(normalized, normalized.T)
    else:
        raise ValueError(f"Unsupported metric: {metric}")
    
    # Create edge list
    edge_list = []
    
    if k is not None:
        # K most similar for each node
        k = min(k, len(node_features_np) - 1)
        
        for i in range(len(node_features_np)):
            # Get similarities to node i
            similarities = sim_matrix[i]
            
            # Set self-similarity to -inf to exclude self
            similarities[i] = -np.inf
            
            # Get indices of k most similar nodes
            most_similar = np.argpartition(similarities, -k)[-k:]
            
            # Add edges
            for j in most_similar:
                edge_list.append((i, j))
    else:
        # Threshold-based
        for i in range(len(node_features_np)):
            for j in range(len(node_features_np)):
                if i != j and sim_matrix[i, j] >= threshold:
                    edge_list.append((i, j))
    
    # Convert to tensor
    if edge_list:
        edge_index = torch.tensor(edge_list, dtype=torch.long).t()
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    
    return edge_index

def color_similarity_edges(image, node_info, threshold=0.8, k=None):
    """
    Creates edges between nodes based on color similarity.
    
    Parameters
    ----------
    image : numpy.ndarray
        Input image with shape (H, W, C)
    node_info : dict
        Node information dictionary from node creation
    threshold : float, optional
        Similarity threshold, by default 0.8
    k : int, optional
        If provided, connect each node to its k most similar neighbors,
        ignoring the threshold, by default None
        
    Returns
    -------
    torch.Tensor
        Edge index tensor with shape (2, E)
    """
    # Extract mean colors per node
    colors = []
    
    if 'segments' in node_info:  # Superpixel nodes
        segments = node_info['segments']
        unique_segments = np.unique(segments)
        
        for i in unique_segments:
            mask = segments == i
            if np.any(mask):
                region = image[mask]
                mean_color = np.mean(region, axis=0)
                colors.append(mean_color)
            else:
                # Handle empty segments (should not happen)
                colors.append(np.zeros(image.shape[-1]))
    
    elif 'patches' in node_info:  # Patch nodes
        patches = node_info['patches']
        
        for patch in patches:
            mean_color = np.mean(patch, axis=(0, 1))
            colors.append(mean_color)
    
    elif 'values' in node_info:  # Pixel nodes
        colors = node_info['values']
    
    else:
        raise ValueError("Unsupported node type")
    
    # Convert to numpy array
    colors = np.array(colors)
    
    # Compute feature similarity edges
    return feature_similarity_edges(colors, threshold=threshold, metric='cosine', k=k)

def texture_similarity_edges(image, node_info, threshold=0.7, k=None):
    """
    Creates edges between nodes based on texture similarity.
    
    Parameters
    ----------
    image : numpy.ndarray
        Input image with shape (H, W, C)
    node_info : dict
        Node information dictionary from node creation
    threshold : float, optional
        Similarity threshold, by default 0.7
    k : int, optional
        If provided, connect each node to its k most similar neighbors,
        ignoring the threshold, by default None
        
    Returns
    -------
    torch.Tensor
        Edge index tensor with shape (2, E)
    """
    try:
        from skimage.feature import local_binary_pattern
    except ImportError:
        raise ImportError("scikit-image is required for texture similarity.")
    
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
    
    # Compute LBP
    lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')
    
    # Extract texture histograms
    histograms = []
    
    if 'segments' in node_info:  # Superpixel nodes
        segments = node_info['segments']
        unique_segments = np.unique(segments)
        
        for i in unique_segments:
            mask = segments == i
            if np.any(mask):
                region_lbp = lbp[mask]
                hist, _ = np.histogram(region_lbp, bins=10, range=(0, 10), density=True)
                histograms.append(hist)
            else:
                # Handle empty segments (should not happen)
                histograms.append(np.zeros(10))
    
    elif 'patches' in node_info:  # Patch nodes
        bboxes = node_info['bboxes']
        
        for bbox in bboxes:
            top, left, bottom, right = bbox
            patch_lbp = lbp[top:bottom, left:right]
            
            if patch_lbp.size > 0:
                hist, _ = np.histogram(patch_lbp, bins=10, range=(0, 10), density=True)
                histograms.append(hist)
            else:
                # Handle empty patches (should not happen)
                histograms.append(np.zeros(10))
    
    elif 'values' in node_info:  # Pixel nodes
        # For pixel nodes, we can't compute texture features
        # Just use dummy features
        histograms = np.random.rand(len(node_info['positions']), 10)
    
    else:
        raise ValueError("Unsupported node type")
    
    # Convert to numpy array
    histograms = np.array(histograms)
    
    # Compute feature similarity edges
    return feature_similarity_edges(histograms, threshold=threshold, metric='cosine', k=k)

def spatial_similarity_edges(node_info, threshold=0.2, k=None):
    """
    Creates edges between nodes based on spatial proximity.
    
    Parameters
    ----------
    node_info : dict
        Node information dictionary from node creation
    threshold : float, optional
        Similarity threshold as a fraction of the image size, by default 0.2
    k : int, optional
        If provided, connect each node to its k closest neighbors,
        ignoring the threshold, by default None
        
    Returns
    -------
    torch.Tensor
        Edge index tensor with shape (2, E)
    """
    if 'centroids' not in node_info:
        raise ValueError("Node info does not contain centroids")
    
    centroids = node_info['centroids']
    
    # Calculate pairwise distances
    dist_matrix = distance_matrix(centroids, centroids)
    
    # Calculate max possible distance for normalization
    max_dist = np.max(dist_matrix) if np.max(dist_matrix) > 0 else 1
    
    # Convert to similarity (higher = more similar)
    sim_matrix = 1 - (dist_matrix / max_dist)
    
    # Create edge list
    edge_list = []
    
    if k is not None:
        # K closest neighbors for each node
        k = min(k, len(centroids) - 1)
        
        for i in range(len(centroids)):
            # Get similarities to node i
            similarities = sim_matrix[i]
            
            # Set self-similarity to -inf to exclude self
            similarities[i] = -np.inf
            
            # Get indices of k most similar nodes
            most_similar = np.argpartition(similarities, -k)[-k:]
            
            # Add edges
            for j in most_similar:
                edge_list.append((i, j))
    else:
        # Threshold-based
        for i in range(len(centroids)):
            for j in range(len(centroids)):
                if i != j and sim_matrix[i, j] >= threshold:
                    edge_list.append((i, j))
    
    # Convert to tensor
    if edge_list:
        edge_index = torch.tensor(edge_list, dtype=torch.long).t()
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    
    return edge_index
