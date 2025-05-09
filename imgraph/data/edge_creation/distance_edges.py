"""
Functions for creating distance-based edges between nodes.
"""

import numpy as np
import torch
from scipy.spatial import distance_matrix

def distance_threshold_edges(node_info, threshold=50, image=None):
    """
    Creates edges between nodes within a distance threshold.
    
    Parameters
    ----------
    node_info : dict
        Node information dictionary from node creation
    threshold : float, optional
        Distance threshold, by default 50
    image : numpy.ndarray, optional
        Input image, by default None
        
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
    
    # Create edges for node pairs within threshold
    edge_list = []
    
    for i in range(len(centroids)):
        for j in range(len(centroids)):
            if i != j and dist_matrix[i, j] <= threshold:
                edge_list.append((i, j))
    
    # Convert to tensor
    if edge_list:
        edge_index = torch.tensor(edge_list, dtype=torch.long).t()
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    
    return edge_index

def k_nearest_edges(node_info, k=6, bidirectional=True, image=None):
    """
    Creates edges between each node and its k nearest neighbors.
    
    Parameters
    ----------
    node_info : dict
        Node information dictionary from node creation
    k : int, optional
        Number of nearest neighbors, by default 6
    bidirectional : bool, optional
        Whether to make edges bidirectional, by default True
    image : numpy.ndarray, optional
        Input image, by default None
        
    Returns
    -------
    torch.Tensor
        Edge index tensor with shape (2, E)
    """
    if 'centroids' not in node_info:
        raise ValueError("Node info does not contain centroids")
    
    centroids = node_info['centroids']
    
    # Handle edge case with too few nodes
    if len(centroids) <= 1:
        return torch.zeros((2, 0), dtype=torch.long)
    
    # Adjust k if needed
    k = min(k, len(centroids) - 1)
    
    # Calculate pairwise distances
    dist_matrix = distance_matrix(centroids, centroids)
    
    # For each node, find k nearest neighbors
    edge_list = []
    
    for i in range(len(centroids)):
        # Get distances from node i to all other nodes
        distances = dist_matrix[i]
        
        # Set self-distance to inf to exclude self
        distances[i] = float('inf')
        
        # Get indices of k nearest neighbors
        nn_indices = np.argpartition(distances, k)[:k]
        
        # Add edges
        for j in nn_indices:
            edge_list.append((i, j))
            
            # Add reverse edge if bidirectional
            if bidirectional:
                edge_list.append((j, i))
    
    # Remove duplicates if bidirectional
    if bidirectional:
        edge_list = list(set(edge_list))
    
    # Convert to tensor
    if edge_list:
        edge_index = torch.tensor(edge_list, dtype=torch.long).t()
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    
    return edge_index

def delaunay_edges(node_info, image=None):
    """
    Creates edges based on Delaunay triangulation of node centroids.
    
    Parameters
    ----------
    node_info : dict
        Node information dictionary from node creation
    image : numpy.ndarray, optional
        Input image, by default None
        
    Returns
    -------
    torch.Tensor
        Edge index tensor with shape (2, E)
    """
    try:
        from scipy.spatial import Delaunay
    except ImportError:
        raise ImportError("scipy is required for Delaunay triangulation.")
    
    if 'centroids' not in node_info:
        raise ValueError("Node info does not contain centroids")
    
    centroids = node_info['centroids']
    
    # Handle edge cases
    if len(centroids) < 3:
        # Fall back to fully connected graph for < 3 nodes
        edge_list = []
        for i in range(len(centroids)):
            for j in range(len(centroids)):
                if i != j:
                    edge_list.append((i, j))
        
        if edge_list:
            edge_index = torch.tensor(edge_list, dtype=torch.long).t()
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
        
        return edge_index
    
    # Compute Delaunay triangulation
    tri = Delaunay(centroids)
    
    # Create edges from triangulation
    edge_list = []
    
    for simplex in tri.simplices:
        # Create edges for each side of the triangle
        edge_list.append((simplex[0], simplex[1]))
        edge_list.append((simplex[1], simplex[0]))
        edge_list.append((simplex[0], simplex[2]))
        edge_list.append((simplex[2], simplex[0]))
        edge_list.append((simplex[1], simplex[2]))
        edge_list.append((simplex[2], simplex[1]))
    
    # Remove duplicates
    edge_list = list(set(edge_list))
    
    # Convert to tensor
    if edge_list:
        edge_index = torch.tensor(edge_list, dtype=torch.long).t()
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    
    return edge_index
