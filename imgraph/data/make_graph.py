"""
Main module for constructing a graph from an image.
"""

import numpy as np
import torch
from torch_geometric.data import Data

class GraphBuilder:
    """
    A class for constructing a graph from an image.
    
    Parameters
    ----------
    node_creation_method : callable
        Method for creating nodes from an image
    node_feature_method : callable
        Method for extracting node features
    edge_creation_method : callable
        Method for creating edges between nodes
    edge_feature_method : callable, optional
        Method for extracting edge features, by default None
    """
    
    def __init__(self, node_creation_method, node_feature_method, edge_creation_method, edge_feature_method=None):
        """Initialize the GraphBuilder."""
        self.node_creation_method = node_creation_method
        self.node_feature_method = node_feature_method
        self.edge_creation_method = edge_creation_method
        self.edge_feature_method = edge_feature_method
    
    def build_graph(self, image):
        """
        Builds a graph from an image.
        
        Parameters
        ----------
        image : numpy.ndarray
            Input image with shape (H, W, C)
            
        Returns
        -------
        torch_geometric.data.Data
            Graph representation of the image
        """
        # Create nodes
        node_info = self.node_creation_method(image)
        
        # Extract node features
        node_features = self.node_feature_method(image, node_info)
        
        # Create edges
        edge_index = self.edge_creation_method(node_info, image)
        
        # Extract edge features (if method provided)
        if self.edge_feature_method is not None:
            edge_attr = self.edge_feature_method(image, node_info, edge_index)
        else:
            edge_attr = None
        
        # Create graph data object
        graph_data = Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            # Add additional metadata
            num_nodes=len(node_features),
            image_size=torch.tensor(image.shape[:2])
        )
        
        # Store node info for visualization or further processing
        graph_data.node_info = node_info
        
        return graph_data
    
    def __call__(self, image):
        """
        Alias for build_graph method.
        
        Parameters
        ----------
        image : numpy.ndarray
            Input image with shape (H, W, C)
            
        Returns
        -------
        torch_geometric.data.Data
            Graph representation of the image
        """
        return self.build_graph(image)
    
    def visualize_graph(self, image, graph=None):
        """
        Visualizes the graph on top of the image.
        
        Parameters
        ----------
        image : numpy.ndarray
            Input image with shape (H, W, C)
        graph : torch_geometric.data.Data, optional
            Graph to visualize, by default None (build a new graph)
            
        Returns
        -------
        matplotlib.figure.Figure
            Figure with the visualization
        """
        import matplotlib.pyplot as plt
        
        # Build graph if not provided
        if graph is None:
            graph = self.build_graph(image)
        
        # Get node positions
        if hasattr(graph, 'node_info') and 'centroids' in graph.node_info:
            node_positions = graph.node_info['centroids']
        else:
            raise ValueError("Graph does not contain node positions")
        
        # Get edge indices
        edge_index = graph.edge_index.cpu().numpy()
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Display image
        ax.imshow(image)
        
        # Plot nodes
        ax.scatter(node_positions[:, 1], node_positions[:, 0], c='red', s=10, alpha=0.7)
        
        # Plot edges
        for i in range(edge_index.shape[1]):
            src_idx = edge_index[0, i]
            dst_idx = edge_index[1, i]
            
            src_pos = node_positions[src_idx]
            dst_pos = node_positions[dst_idx]
            
            ax.plot([src_pos[1], dst_pos[1]], [src_pos[0], dst_pos[0]], 'b-', alpha=0.3, linewidth=0.5)
        
        # Set title
        ax.set_title(f"Graph Visualization: {len(node_positions)} nodes, {edge_index.shape[1]} edges")
        
        # Turn off axis
        ax.axis('off')
        
        return fig

class MultiGraphBuilder:
    """
    A class for constructing multiple graphs from an image using different methods.
    
    Parameters
    ----------
    builders : list of GraphBuilder
        List of graph builders to use
    """
    
    def __init__(self, builders):
        """Initialize the MultiGraphBuilder."""
        self.builders = builders
    
    def build_graphs(self, image):
        """
        Builds multiple graphs from an image.
        
        Parameters
        ----------
        image : numpy.ndarray
            Input image with shape (H, W, C)
            
        Returns
        -------
        list of torch_geometric.data.Data
            List of graph representations of the image
        """
        return [builder.build_graph(image) for builder in self.builders]
    
    def __call__(self, image):
        """
        Alias for build_graphs method.
        
        Parameters
        ----------
        image : numpy.ndarray
            Input image with shape (H, W, C)
            
        Returns
        -------
        list of torch_geometric.data.Data
            List of graph representations of the image
        """
        return self.build_graphs(image)

def combine_features(feature_methods, image, node_info):
    """
    Combines multiple node feature extraction methods.
    
    Parameters
    ----------
    feature_methods : list of callable
        List of feature extraction methods
    image : numpy.ndarray
        Input image with shape (H, W, C)
    node_info : dict
        Node information dictionary from node creation
        
    Returns
    -------
    torch.Tensor
        Combined node features
    """
    features = [method(image, node_info) for method in feature_methods]
    return torch.cat(features, dim=1)

def combine_edge_features(feature_methods, image, node_info, edge_index):
    """
    Combines multiple edge feature extraction methods.
    
    Parameters
    ----------
    feature_methods : list of callable
        List of feature extraction methods
    image : numpy.ndarray
        Input image with shape (H, W, C)
    node_info : dict
        Node information dictionary from node creation
    edge_index : torch.Tensor
        Edge index tensor with shape (2, E)
        
    Returns
    -------
    torch.Tensor
        Combined edge features
    """
    features = [method(image, node_info, edge_index) for method in feature_methods]
    return torch.cat(features, dim=1)