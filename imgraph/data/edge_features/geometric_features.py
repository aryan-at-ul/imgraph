"""
Functions for creating geometric edge features.
"""

import numpy as np
import torch

def distance_features(node_info, edge_index, normalize=True):
    """
    Computes edge features based on the distance between node centroids.
    
    Parameters
    ----------
    node_info : dict
        Node information dictionary from node creation
    edge_index : torch.Tensor
        Edge index tensor with shape (2, E)
    normalize : bool, optional
        Whether to normalize distances, by default True
        
    Returns
    -------
    torch.Tensor
        Edge feature tensor with shape (E, 1) containing distances
    """
    if 'centroids' not in node_info:
        raise ValueError("Node info does not contain centroids")
    
    centroids = node_info['centroids']
    
    # Convert to tensor
    centroids_tensor = torch.tensor(centroids, dtype=torch.float)
    
    # Handle empty edge index
    if edge_index.shape[1] == 0:
        return torch.zeros((0, 1), dtype=torch.float)
    
    # Extract source and target node indices
    src, dst = edge_index
    
    # Extract source and target node centroids
    src_centroids = centroids_tensor[src]
    dst_centroids = centroids_tensor[dst]
    
    # Compute Euclidean distances
    distances = torch.sqrt(torch.sum((dst_centroids - src_centroids) ** 2, dim=1))
    
    # Normalize if requested
    if normalize:
        max_dist = torch.max(distances)
        if max_dist > 0:
            distances = distances / max_dist
    
    # Reshape to (E, 1)
    return distances.unsqueeze(1)

def angle_features(node_info, edge_index, normalize=True):
    """
    Computes edge features based on the angle between nodes.
    
    Parameters
    ----------
    node_info : dict
        Node information dictionary from node creation
    edge_index : torch.Tensor
        Edge index tensor with shape (2, E)
    normalize : bool, optional
        Whether to normalize angles, by default True
        
    Returns
    -------
    torch.Tensor
        Edge feature tensor with shape (E, 2) containing [sin(θ), cos(θ)]
    """
    if 'centroids' not in node_info:
        raise ValueError("Node info does not contain centroids")
    
    centroids = node_info['centroids']
    
    # Convert to tensor
    centroids_tensor = torch.tensor(centroids, dtype=torch.float)
    
    # Handle empty edge index
    if edge_index.shape[1] == 0:
        return torch.zeros((0, 2), dtype=torch.float)
    
    # Extract source and target node indices
    src, dst = edge_index
    
    # Extract source and target node centroids
    src_centroids = centroids_tensor[src]
    dst_centroids = centroids_tensor[dst]
    
    # Compute displacement vectors
    delta = dst_centroids - src_centroids
    
    # Compute angles
    angles = torch.atan2(delta[:, 0], delta[:, 1])  # atan2(y, x)
    
    # Convert to sin and cos for periodicity-invariant representation
    sin_angles = torch.sin(angles)
    cos_angles = torch.cos(angles)
    
    # Combine sine and cosine
    angle_features = torch.stack([sin_angles, cos_angles], dim=1)
    
    return angle_features

def relative_position_features(node_info, edge_index, image=None):
    """
    Computes edge features based on the relative position between nodes.
    
    Parameters
    ----------
    node_info : dict
        Node information dictionary from node creation
    edge_index : torch.Tensor
        Edge index tensor with shape (2, E)
    image : numpy.ndarray, optional
        Input image, by default None
        
    Returns
    -------
    torch.Tensor
        Edge feature tensor with shape (E, 2) containing normalized relative positions
    """
    if 'centroids' not in node_info:
        raise ValueError("Node info does not contain centroids")
    
    centroids = node_info['centroids']
    
    # Convert to tensor
    centroids_tensor = torch.tensor(centroids, dtype=torch.float)
    
    # Handle empty edge index
    if edge_index.shape[1] == 0:
        return torch.zeros((0, 2), dtype=torch.float)
    
    # Extract source and target node indices
    src, dst = edge_index
    
    # Extract source and target node centroids
    src_centroids = centroids_tensor[src]
    dst_centroids = centroids_tensor[dst]
    
    # Compute relative positions
    rel_pos = dst_centroids - src_centroids
    
    # Normalize by image size if available
    if image is not None:
        height, width = image.shape[:2]
        rel_pos[:, 0] = rel_pos[:, 0] / height  # y-coordinate
        rel_pos[:, 1] = rel_pos[:, 1] / width   # x-coordinate
    else:
        # Normalize by maximum absolute value
        max_abs = torch.max(torch.abs(rel_pos))
        if max_abs > 0:
            rel_pos = rel_pos / max_abs
    
    return rel_pos

def spatial_relationship_features(node_info, edge_index, image=None):
    """
    Computes comprehensive edge features based on spatial relationships.
    
    Parameters
    ----------
    node_info : dict
        Node information dictionary from node creation
    edge_index : torch.Tensor
        Edge index tensor with shape (2, E)
    image : numpy.ndarray, optional
        Input image, by default None
        
    Returns
    -------
    torch.Tensor
        Edge feature tensor with shape (E, 5) containing:
        [distance, sin(θ), cos(θ), Δy, Δx]
    """
    if 'centroids' not in node_info:
        raise ValueError("Node info does not contain centroids")
    
    centroids = node_info['centroids']
    
    # Convert to tensor
    centroids_tensor = torch.tensor(centroids, dtype=torch.float)
    
    # Handle empty edge index
    if edge_index.shape[1] == 0:
        return torch.zeros((0, 5), dtype=torch.float)
    
    # Extract source and target node indices
    src, dst = edge_index
    
    # Extract source and target node centroids
    src_centroids = centroids_tensor[src]
    dst_centroids = centroids_tensor[dst]
    
    # Compute displacement vectors
    delta = dst_centroids - src_centroids
    
    # Compute distances
    distances = torch.sqrt(torch.sum(delta ** 2, dim=1))
    
    # Normalize distances
    max_dist = torch.max(distances)
    if max_dist > 0:
        norm_distances = distances / max_dist
    else:
        norm_distances = distances
    
    # Compute angles
    angles = torch.atan2(delta[:, 0], delta[:, 1])  # atan2(y, x)
    
    # Convert to sin and cos
    sin_angles = torch.sin(angles)
    cos_angles = torch.cos(angles)
    
    # Normalize relative positions
    if image is not None:
        height, width = image.shape[:2]
        norm_delta_y = delta[:, 0] / height
        norm_delta_x = delta[:, 1] / width
    else:
        # Normalize by maximum absolute value
        max_abs_y = torch.max(torch.abs(delta[:, 0]))
        max_abs_x = torch.max(torch.abs(delta[:, 1]))
        
        if max_abs_y > 0:
            norm_delta_y = delta[:, 0] / max_abs_y
        else:
            norm_delta_y = delta[:, 0]
            
        if max_abs_x > 0:
            norm_delta_x = delta[:, 1] / max_abs_x
        else:
            norm_delta_x = delta[:, 1]
    
    # Combine all features
    features = torch.stack([
        norm_distances,
        sin_angles,
        cos_angles,
        norm_delta_y,
        norm_delta_x
    ], dim=1)
    
    return features

def direction_encoding_features(node_info, edge_index, num_directions=8):
    """
    Encodes edge directions using a one-hot encoding.
    
    Parameters
    ----------
    node_info : dict
        Node information dictionary from node creation
    edge_index : torch.Tensor
        Edge index tensor with shape (2, E)
    num_directions : int, optional
        Number of direction bins, by default 8
        
    Returns
    -------
    torch.Tensor
        Edge feature tensor with shape (E, num_directions)
    """
    if 'centroids' not in node_info:
        raise ValueError("Node info does not contain centroids")
    
    centroids = node_info['centroids']
    
    # Convert to tensor
    centroids_tensor = torch.tensor(centroids, dtype=torch.float)
    
    # Handle empty edge index
    if edge_index.shape[1] == 0:
        return torch.zeros((0, num_directions), dtype=torch.float)
    
    # Extract source and target node indices
    src, dst = edge_index
    
    # Extract source and target node centroids
    src_centroids = centroids_tensor[src]
    dst_centroids = centroids_tensor[dst]
    
    # Compute displacement vectors
    delta = dst_centroids - src_centroids
    
    # Compute angles in radians (range: [-π, π])
    angles = torch.atan2(delta[:, 0], delta[:, 1])
    
    # Convert to the range [0, 2π]
    angles = (angles + 2*np.pi) % (2*np.pi)
    
    # Determine direction bin for each edge
    bin_width = 2*np.pi / num_directions
    direction_bins = (angles / bin_width).long() % num_directions
    
    # Create one-hot encoding
    one_hot = torch.zeros((len(direction_bins), num_directions), dtype=torch.float)
    one_hot.scatter_(1, direction_bins.unsqueeze(1), 1)
    
    return one_hot
