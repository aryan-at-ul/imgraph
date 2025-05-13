"""
Plot functions for visualizing image graphs.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import SpectralClustering

def visualize_graph(image, graph, ax=None, node_size=20, edge_width=0.5, node_color='red', edge_color='blue', alpha=0.6):
    """
    Visualizes a graph overlaid on an image.
    
    Parameters
    ----------
    image : numpy.ndarray
        Input image
    graph : torch_geometric.data.Data
        Graph data
    ax : matplotlib.axes.Axes, optional
        Axes to plot on, by default None
    node_size : int, optional
        Size of nodes, by default 20
    edge_width : float, optional
        Width of edges, by default 0.5
    node_color : str, optional
        Color of nodes, by default 'red'
    edge_color : str, optional
        Color of edges, by default 'blue'
    alpha : float, optional
        Transparency, by default 0.6
        
    Returns
    -------
    matplotlib.figure.Figure
        Figure with the visualization
    """
    # Create figure and axes if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    else:
        fig = ax.figure
    
    # Display image
    ax.imshow(image)
    
    # Get node positions
    if hasattr(graph, 'node_info') and 'centroids' in graph.node_info:
        node_positions = graph.node_info['centroids']
    else:
        raise ValueError("Graph does not contain node positions")
    
    # Get edge indices
    edge_index = graph.edge_index.cpu().numpy()
    
    # Plot nodes
    ax.scatter(node_positions[:, 1], node_positions[:, 0], c=node_color, s=node_size, alpha=alpha)
    
    # Plot edges
    for i in range(edge_index.shape[1]):
        src_idx = edge_index[0, i]
        dst_idx = edge_index[1, i]
        
        src_pos = node_positions[src_idx]
        dst_pos = node_positions[dst_idx]
        
        ax.plot([src_pos[1], dst_pos[1]], [src_pos[0], dst_pos[0]], color=edge_color, alpha=alpha*0.5, linewidth=edge_width)
    
    # Set title and turn off axis
    ax.set_title(f"Graph: {graph.num_nodes} nodes, {edge_index.shape[1]} edges")
    ax.axis('off')
    
    return fig

def visualize_graph_with_features(image, graph, node_feature_idx=None, edge_feature_idx=None, cmap='viridis', ax=None):
    """
    Visualizes a graph with node or edge features as colors.
    
    Parameters
    ----------
    image : numpy.ndarray
        Input image
    graph : torch_geometric.data.Data
        Graph data
    node_feature_idx : int, optional
        Index of node feature to visualize, by default None
    edge_feature_idx : int, optional
        Index of edge feature to visualize, by default None
    cmap : str, optional
        Colormap, by default 'viridis'
    ax : matplotlib.axes.Axes, optional
        Axes to plot on, by default None
        
    Returns
    -------
    matplotlib.figure.Figure
        Figure with the visualization
    """
    # Create figure and axes if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 10))
    else:
        fig = ax.figure
    
    # Display image
    ax.imshow(image)
    
    # Get node positions
    if hasattr(graph, 'node_info') and 'centroids' in graph.node_info:
        node_positions = graph.node_info['centroids']
    else:
        raise ValueError("Graph does not contain node positions")
    
    # Get edge indices
    edge_index = graph.edge_index.cpu().numpy()
    
    # Visualize node features
    if node_feature_idx is not None and graph.x is not None:
        # Extract feature values
        node_features = graph.x[:, node_feature_idx].cpu().numpy()
        
        # Normalize feature values
        norm = Normalize(vmin=node_features.min(), vmax=node_features.max())
        
        # Create a colormap
        cmap_obj = cm.get_cmap(cmap)
        
        # Plot nodes with feature colors
        sc = ax.scatter(node_positions[:, 1], node_positions[:, 0], 
                         c=node_features, cmap=cmap_obj, norm=norm, s=30)
        
        # Add colorbar
        cbar = plt.colorbar(sc, ax=ax, shrink=0.8)
        cbar.set_label(f'Node Feature {node_feature_idx}')
        
        # Plot edges
        for i in range(edge_index.shape[1]):
            src_idx = edge_index[0, i]
            dst_idx = edge_index[1, i]
            
            src_pos = node_positions[src_idx]
            dst_pos = node_positions[dst_idx]
            
            ax.plot([src_pos[1], dst_pos[1]], [src_pos[0], dst_pos[0]], 'b-', alpha=0.2, linewidth=0.5)
    
    # Visualize edge features
    elif edge_feature_idx is not None and hasattr(graph, 'edge_attr') and graph.edge_attr is not None:
        # Extract feature values
        edge_features = graph.edge_attr[:, edge_feature_idx].cpu().numpy()
        
        # Normalize feature values
        norm = Normalize(vmin=edge_features.min(), vmax=edge_features.max())
        
        # Create a colormap
        cmap_obj = cm.get_cmap(cmap)
        
        # Plot nodes
        ax.scatter(node_positions[:, 1], node_positions[:, 0], c='gray', s=20, alpha=0.5)
        
        # Plot edges with feature colors
        lines = []
        colors = []
        
        for i in range(edge_index.shape[1]):
            src_idx = edge_index[0, i]
            dst_idx = edge_index[1, i]
            
            src_pos = node_positions[src_idx]
            dst_pos = node_positions[dst_idx]
            
            # Plot edge with color
            line = ax.plot([src_pos[1], dst_pos[1]], [src_pos[0], dst_pos[0]], 
                            color=cmap_obj(norm(edge_features[i])), linewidth=2, alpha=0.7)[0]
            
            lines.append(line)
            colors.append(edge_features[i])
        
        # Add colorbar
        sm = cm.ScalarMappable(cmap=cmap_obj, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.8)
        cbar.set_label(f'Edge Feature {edge_feature_idx}')
    
    else:
        # Default visualization
        ax.scatter(node_positions[:, 1], node_positions[:, 0], c='red', s=20, alpha=0.6)
        
        # Plot edges
        for i in range(edge_index.shape[1]):
            src_idx = edge_index[0, i]
            dst_idx = edge_index[1, i]
            
            src_pos = node_positions[src_idx]
            dst_pos = node_positions[dst_idx]
            
            ax.plot([src_pos[1], dst_pos[1]], [src_pos[0], dst_pos[0]], 'b-', alpha=0.3, linewidth=0.5)
    
    # Set title and turn off axis
    if node_feature_idx is not None:
        ax.set_title(f"Graph with Node Feature {node_feature_idx}")
    elif edge_feature_idx is not None:
        ax.set_title(f"Graph with Edge Feature {edge_feature_idx}")
    else:
        ax.set_title(f"Graph: {graph.num_nodes} nodes, {edge_index.shape[1]} edges")
    
    ax.axis('off')
    
    return fig

def plot_node_feature_distribution(graph, feature_indices=None, num_features=5, figsize=(12, 8)):
    """
    Plots the distribution of node features.
    
    Parameters
    ----------
    graph : torch_geometric.data.Data
        Graph data
    feature_indices : list, optional
        Indices of features to plot, by default None (auto-select)
    num_features : int, optional
        Number of features to plot if feature_indices is None, by default 5
    figsize : tuple, optional
        Figure size, by default (12, 8)
        
    Returns
    -------
    matplotlib.figure.Figure
        Figure with the visualization
    """
    if graph.x is None or graph.x.shape[1] == 0:
        raise ValueError("Graph does not contain node features")
    
    # Select features to plot
    if feature_indices is None:
        feature_dim = graph.x.shape[1]
        feature_indices = list(range(min(num_features, feature_dim)))
    
    # Create figure
    fig, axs = plt.subplots(len(feature_indices), 1, figsize=figsize, sharex=True)
    
    # Handle single feature case
    if len(feature_indices) == 1:
        axs = [axs]
    
    # Plot each feature distribution
    for i, feat_idx in enumerate(feature_indices):
        feature_values = graph.x[:, feat_idx].cpu().numpy()
        axs[i].hist(feature_values, bins=30, alpha=0.7)
        axs[i].set_title(f"Node Feature {feat_idx} Distribution")
        axs[i].set_ylabel("Count")
    
    axs[-1].set_xlabel("Feature Value")
    plt.tight_layout()
    
    return fig

def plot_edge_feature_distribution(graph, feature_indices=None, num_features=3, figsize=(12, 8)):
    """
    Plots the distribution of edge features.
    
    Parameters
    ----------
    graph : torch_geometric.data.Data
        Graph data
    feature_indices : list, optional
        Indices of features to plot, by default None (auto-select)
    num_features : int, optional
        Number of features to plot if feature_indices is None, by default 3
    figsize : tuple, optional
        Figure size, by default (12, 8)
        
    Returns
    -------
    matplotlib.figure.Figure
        Figure with the visualization
    """
    if not hasattr(graph, 'edge_attr') or graph.edge_attr is None or graph.edge_attr.shape[1] == 0:
        raise ValueError("Graph does not contain edge features")
    
    # Select features to plot
    if feature_indices is None:
        feature_dim = graph.edge_attr.shape[1]
        feature_indices = list(range(min(num_features, feature_dim)))
    
    # Create figure
    fig, axs = plt.subplots(len(feature_indices), 1, figsize=figsize, sharex=True)
    
    # Handle single feature case
    if len(feature_indices) == 1:
        axs = [axs]
    
    # Plot each feature distribution
    for i, feat_idx in enumerate(feature_indices):
        feature_values = graph.edge_attr[:, feat_idx].cpu().numpy()
        axs[i].hist(feature_values, bins=30, alpha=0.7)
        axs[i].set_title(f"Edge Feature {feat_idx} Distribution")
        axs[i].set_ylabel("Count")
    
    axs[-1].set_xlabel("Feature Value")
    plt.tight_layout()
    
    return fig

def visualize_adjacency_matrix(graph, cmap='Blues', figsize=(10, 8)):
    """
    Visualizes the adjacency matrix of the graph.
    
    Parameters
    ----------
    graph : torch_geometric.data.Data
        Graph data
    cmap : str, optional
        Colormap, by default 'Blues'
    figsize : tuple, optional
        Figure size, by default (10, 8)
        
    Returns
    -------
    matplotlib.figure.Figure
        Figure with the visualization
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get edge indices
    edge_index = graph.edge_index.cpu().numpy()
    
    # Create sparse adjacency matrix
    num_nodes = graph.num_nodes
    adj_matrix = np.zeros((num_nodes, num_nodes))
    
    for i in range(edge_index.shape[1]):
        src, dst = edge_index[0, i], edge_index[1, i]
        adj_matrix[src, dst] = 1
    
    # Visualize adjacency matrix
    im = ax.imshow(adj_matrix, cmap=cmap)
    
    # Add colorbar
    plt.colorbar(im, ax=ax)
    
    # Set title and labels
    ax.set_title("Adjacency Matrix")
    ax.set_xlabel("Node Index")
    ax.set_ylabel("Node Index")
    
    return fig

def visualize_node_importance(image, graph, importance, cmap='plasma', ax=None, alpha=0.7):
    """
    Visualizes node importance on the graph.
    
    Parameters
    ----------
    image : numpy.ndarray
        Input image
    graph : torch_geometric.data.Data
        Graph data
    importance : torch.Tensor or numpy.ndarray
        Node importance scores
    cmap : str, optional
        Colormap, by default 'plasma'
    ax : matplotlib.axes.Axes, optional
        Axes to plot on, by default None
    alpha : float, optional
        Transparency, by default 0.7
        
    Returns
    -------
    matplotlib.figure.Figure
        Figure with the visualization
    """
    # Create figure and axes if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 10))
    else:
        fig = ax.figure
    
    # Display image
    ax.imshow(image)
    
    # Get node positions
    if hasattr(graph, 'node_info') and 'centroids' in graph.node_info:
        node_positions = graph.node_info['centroids']
    else:
        raise ValueError("Graph does not contain node positions")
    
    # Convert importance to numpy if it's a tensor
    if isinstance(importance, torch.Tensor):
        importance = importance.cpu().numpy()
    
    # Normalize importance scores
    norm = Normalize(vmin=importance.min(), vmax=importance.max())
    
    # Create a colormap
    cmap_obj = cm.get_cmap(cmap)
    
    # Plot nodes with importance scores as colors and sizes
    sizes = 20 + 100 * norm(importance)  # Scale sizes
    sc = ax.scatter(node_positions[:, 1], node_positions[:, 0], 
                     c=importance, cmap=cmap_obj, norm=norm, s=sizes, alpha=alpha)
    
    # Add colorbar
    cbar = plt.colorbar(sc, ax=ax, shrink=0.8)
    cbar.set_label('Node Importance')
    
    # Get edge indices
    edge_index = graph.edge_index.cpu().numpy()
    
    # Plot edges
    for i in range(edge_index.shape[1]):
        src_idx = edge_index[0, i]
        dst_idx = edge_index[1, i]
        
        src_pos = node_positions[src_idx]
        dst_pos = node_positions[dst_idx]
        
        ax.plot([src_pos[1], dst_pos[1]], [src_pos[0], dst_pos[0]], 'gray', alpha=0.2, linewidth=0.5)
    
    # Set title and turn off axis
    ax.set_title("Node Importance Visualization")
    ax.axis('off')
    
    return fig

def plot_graph_spectral_clustering(image, graph, n_clusters=5, cmap='tab10', ax=None):
    """
    Visualizes spectral clustering of the graph.
    
    Parameters
    ----------
    image : numpy.ndarray
        Input image
    graph : torch_geometric.data.Data
        Graph data
    n_clusters : int, optional
        Number of clusters, by default 5
    cmap : str, optional
        Colormap, by default 'tab10'
    ax : matplotlib.axes.Axes, optional
        Axes to plot on, by default None
        
    Returns
    -------
    matplotlib.figure.Figure
        Figure with the visualization
    """
    # Create figure and axes if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 10))
    else:
        fig = ax.figure
    
    # Display image
    ax.imshow(image)
    
    # Get node positions
    if hasattr(graph, 'node_info') and 'centroids' in graph.node_info:
        node_positions = graph.node_info['centroids']
    else:
        raise ValueError("Graph does not contain node positions")
    
    # Get edge indices and convert to adjacency matrix
    edge_index = graph.edge_index.cpu().numpy()
    num_nodes = graph.num_nodes
    
    # Create sparse adjacency matrix
    rows = edge_index[0]
    cols = edge_index[1]
    data = np.ones(len(rows))
    adj_matrix = csr_matrix((data, (rows, cols)), shape=(num_nodes, num_nodes))
    
    # Spectral clustering
    sc = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', 
                            assign_labels='discretize', random_state=42)
    cluster_labels = sc.fit_predict(adj_matrix)
    
    # Plot nodes with cluster colors
    cmap_obj = cm.get_cmap(cmap, n_clusters)
    sc = ax.scatter(node_positions[:, 1], node_positions[:, 0], 
                     c=cluster_labels, cmap=cmap_obj, s=30, alpha=0.8)
    
    # Add colorbar
    cbar = plt.colorbar(sc, ax=ax, ticks=range(n_clusters), shrink=0.8)
    cbar.set_label('Cluster')
    
    # Plot edges
    for i in range(edge_index.shape[1]):
        src_idx = edge_index[0, i]
        dst_idx = edge_index[1, i]
        
        src_pos = node_positions[src_idx]
        dst_pos = node_positions[dst_idx]
        
        # Only draw edges between nodes in the same cluster
        if cluster_labels[src_idx] == cluster_labels[dst_idx]:
            ax.plot([src_pos[1], dst_pos[1]], [src_pos[0], dst_pos[0]], 
                     color=cmap_obj(cluster_labels[src_idx]), alpha=0.3, linewidth=0.7)
        else:
            ax.plot([src_pos[1], dst_pos[1]], [src_pos[0], dst_pos[0]], 
                     'gray', alpha=0.1, linewidth=0.3)
    
    # Set title and turn off axis
    ax.set_title(f"Spectral Clustering (k={n_clusters})")
    ax.axis('off')
    
    return fig

def visualize_node_embeddings(graph, image=None, method='tsne', figsize=(12, 10)):
    """
    Visualizes node embeddings in 2D space.
    
    Parameters
    ----------
    graph : torch_geometric.data.Data
        Graph data
    image : numpy.ndarray, optional
        Input image for reference, by default None
    method : str, optional
        Dimensionality reduction method, by default 'tsne'
        Options: 'tsne', 'pca'
    figsize : tuple, optional
        Figure size, by default (12, 10)
        
    Returns
    -------
    matplotlib.figure.Figure
        Figure with the visualization
    """
    if graph.x is None or graph.x.shape[1] == 0:
        raise ValueError("Graph does not contain node features")
    
    # Create figure
    fig = plt.figure(figsize=figsize)
    
    # If image is provided, create a side-by-side layout
    if image is not None:
        gs = fig.add_gridspec(1, 2, width_ratios=[1, 1])
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])
        
        # Display image with graph
        if hasattr(graph, 'node_info') and 'centroids' in graph.node_info:
            visualize_graph(image, graph, ax=ax1)
        else:
            ax1.imshow(image)
            ax1.set_title("Original Image")
            ax1.axis('off')
    else:
        ax2 = fig.add_subplot(111)
    
    # Get node features
    node_features = graph.x.cpu().numpy()
    
    # Reduce dimensionality to 2D
    if method.lower() == 'tsne':
        embeddings = TSNE(n_components=2, random_state=42).fit_transform(node_features)
    elif method.lower() == 'pca':
        embeddings = PCA(n_components=2, random_state=42).fit_transform(node_features)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Get edge indices
    edge_index = graph.edge_index.cpu().numpy()
    
    # Plot embeddings
    ax2.scatter(embeddings[:, 0], embeddings[:, 1], c='blue', s=30, alpha=0.7)
    
    # Plot edges in embedding space
    for i in range(edge_index.shape[1]):
        src_idx = edge_index[0, i]
        dst_idx = edge_index[1, i]
        
        src_pos = embeddings[src_idx]
        dst_pos = embeddings[dst_idx]
        
        ax2.plot([src_pos[0], dst_pos[0]], [src_pos[1], dst_pos[1]], 'gray', alpha=0.2, linewidth=0.5)
    
    # Set title and labels
    ax2.set_title(f"Node Embeddings ({method.upper()})")
    ax2.set_xlabel("Dimension 1")
    ax2.set_ylabel("Dimension 2")
    
    plt.tight_layout()
    
    return fig

def compare_graphs(image, graphs, names, figsize=(15, 10)):
    """
    Compares multiple graphs side by side.
    
    Parameters
    ----------
    image : numpy.ndarray
        Input image
    graphs : list
        List of graph data objects
    names : list
        List of graph names
    figsize : tuple, optional
        Figure size, by default (15, 10)
        
    Returns
    -------
    matplotlib.figure.Figure
        Figure with the visualization
    """
    n_graphs = len(graphs)
    
    # Create figure with subplots
    fig, axs = plt.subplots(1, n_graphs + 1, figsize=figsize)
    
    # Display original image
    axs[0].imshow(image)
    axs[0].set_title("Original Image")
    axs[0].axis('off')
    
    # Display each graph
    for i, (graph, name) in enumerate(zip(graphs, names)):
        visualize_graph(image, graph, ax=axs[i+1])
        axs[i+1].set_title(f"{name}\n{graph.num_nodes} nodes, {graph.edge_index.shape[1]} edges")
    
    plt.tight_layout()
    
    return fig
