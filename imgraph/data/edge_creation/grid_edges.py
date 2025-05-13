"""
Functions for creating grid-based edges between nodes.
"""

import numpy as np
import torch

def grid_4_edges(node_info, image=None):
    """
    Creates 4-connected grid edges (up, down, left, right) between nodes.
    
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
    if 'positions' not in node_info:
        raise ValueError("Node info does not contain positions")
    
    positions = node_info['positions']
    
    # Find grid dimensions
    if positions.size == 0:
        return torch.zeros((2, 0), dtype=torch.long)
    
    max_row = int(np.max(positions[:, 0]))
    max_col = int(np.max(positions[:, 1]))
    
    # Create position to node index mapping
    pos_to_idx = {(int(pos[0]), int(pos[1])): i for i, pos in enumerate(positions)}
    
    # Initialize edge lists
    edge_list = []
    
    # Create 4-connected edges
    for pos, idx in pos_to_idx.items():
        row, col = pos
        
        # Up
        if (row - 1, col) in pos_to_idx:
            edge_list.append((idx, pos_to_idx[(row - 1, col)]))
        
        # Down
        if (row + 1, col) in pos_to_idx:
            edge_list.append((idx, pos_to_idx[(row + 1, col)]))
        
        # Left
        if (row, col - 1) in pos_to_idx:
            edge_list.append((idx, pos_to_idx[(row, col - 1)]))
        
        # Right
        if (row, col + 1) in pos_to_idx:
            edge_list.append((idx, pos_to_idx[(row, col + 1)]))
    
    # Convert to tensor
    if edge_list:
        edge_index = torch.tensor(edge_list, dtype=torch.long).t()
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    
    return edge_index

def grid_8_edges(node_info, image=None):
    """
    Creates 8-connected grid edges (including diagonals) between nodes.
    
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
    if 'positions' not in node_info:
        raise ValueError("Node info does not contain positions")
    
    positions = node_info['positions']
    
    # Find grid dimensions
    if positions.size == 0:
        return torch.zeros((2, 0), dtype=torch.long)
    
    max_row = int(np.max(positions[:, 0]))
    max_col = int(np.max(positions[:, 1]))
    
    # Create position to node index mapping
    pos_to_idx = {(int(pos[0]), int(pos[1])): i for i, pos in enumerate(positions)}
    
    # Initialize edge lists
    edge_list = []
    
    # Create 8-connected edges
    for pos, idx in pos_to_idx.items():
        row, col = pos
        
        # Check all 8 neighbors
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue  # Skip self
                
                neighbor_pos = (row + dr, col + dc)
                if neighbor_pos in pos_to_idx:
                    edge_list.append((idx, pos_to_idx[neighbor_pos]))
    
    # Convert to tensor
    if edge_list:
        edge_index = torch.tensor(edge_list, dtype=torch.long).t()
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    
    return edge_index

def grid_radius_edges(node_info, radius=2, image=None):
    """
    Creates grid edges within a given radius.
    
    Parameters
    ----------
    node_info : dict
        Node information dictionary from node creation
    radius : int, optional
        Radius of connectivity, by default 2
    image : numpy.ndarray, optional
        Input image, by default None
        
    Returns
    -------
    torch.Tensor
        Edge index tensor with shape (2, E)
    """
    if 'positions' not in node_info:
        raise ValueError("Node info does not contain positions")
    
    positions = node_info['positions']
    
    # Create position to node index mapping
    pos_to_idx = {(int(pos[0]), int(pos[1])): i for i, pos in enumerate(positions)}
    
    # Initialize edge lists
    edge_list = []
    
    # Create radius-connected edges
    for pos, idx in pos_to_idx.items():
        row, col = pos
        
        # Check all neighbors within radius
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                if dr == 0 and dc == 0:
                    continue  # Skip self
                
                # Check if within radius
                if dr**2 + dc**2 > radius**2:
                    continue
                
                neighbor_pos = (row + dr, col + dc)
                if neighbor_pos in pos_to_idx:
                    edge_list.append((idx, pos_to_idx[neighbor_pos]))
    
    # Convert to tensor
    if edge_list:
        edge_index = torch.tensor(edge_list, dtype=torch.long).t()
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    
    return edge_index

def dense_grid_edges(node_info, image=None):
    """
    Creates fully-connected edges between all nodes in a grid.
    
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
    # Get number of nodes
    if 'positions' in node_info:
        n_nodes = len(node_info['positions'])
    elif 'centroids' in node_info:
        n_nodes = len(node_info['centroids'])
    else:
        raise ValueError("Node info does not contain positions or centroids")
    
    # Create edge lists
    edge_list = []
    
    # Create fully-connected edges
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j:
                edge_list.append((i, j))
    
    # Convert to tensor
    if edge_list:
        edge_index = torch.tensor(edge_list, dtype=torch.long).t()
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    
    return edge_index
