"""
Backward compatibility module for imgraph.

This module provides the old function names that were used in previous versions.
"""

import warnings
import numpy as np
import torch
from skimage.segmentation import slic
from torch_geometric.data import Data

from imgraph.data.make_graph import GraphBuilder
from imgraph.data.node_creation import slic_superpixel_nodes
from imgraph.data.node_features import mean_color_features
from imgraph.data.edge_creation import region_adjacency_edges

def image_transform_slic(image, n_segments=100, compactness=10):
    """
    Legacy function for SLIC superpixel segmentation.
    
    Parameters
    ----------
    image : numpy.ndarray
        Input image
    n_segments : int, optional
        Number of segments, by default 100
    compactness : int, optional
        Compactness parameter, by default 10
        
    Returns
    -------
    dict
        Node information dictionary
    """
    warnings.warn(
        "image_transform_slic is deprecated. Use slic_superpixel_nodes instead.",
        DeprecationWarning, stacklevel=2
    )
    return slic_superpixel_nodes(image, n_segments=n_segments, compactness=compactness)

def make_edges(node_info, image=None):
    """
    Legacy function for creating edges between nodes.
    
    Parameters
    ----------
    node_info : dict
        Node information dictionary
    image : numpy.ndarray, optional
        Input image, by default None
        
    Returns
    -------
    torch.Tensor
        Edge index tensor
    """
    warnings.warn(
        "make_edges is deprecated. Use region_adjacency_edges instead.",
        DeprecationWarning, stacklevel=2
    )
    return region_adjacency_edges(node_info, image)

def graph_generator(image, n_segments=100, compactness=10):
    """
    Legacy function for generating a graph from an image.
    
    Parameters
    ----------
    image : numpy.ndarray
        Input image
    n_segments : int, optional
        Number of segments, by default 100
    compactness : int, optional
        Compactness parameter, by default 10
        
    Returns
    -------
    torch_geometric.data.Data
        Graph data object
    """
    warnings.warn(
        "graph_generator is deprecated. Use GraphBuilder or GraphPresets instead.",
        DeprecationWarning, stacklevel=2
    )
    
    # Create nodes
    node_info = slic_superpixel_nodes(image, n_segments=n_segments, compactness=compactness)
    
    # Extract node features
    node_features = mean_color_features(image, node_info)
    
    # Create edges
    edge_index = region_adjacency_edges(node_info)
    
    # Create graph data object
    graph_data = Data(
        x=node_features,
        edge_index=edge_index,
        num_nodes=len(node_features),
        image_size=torch.tensor(image.shape[:2])
    )
    
    # Store node info for visualization
    graph_data.node_info = node_info
    
    return graph_data