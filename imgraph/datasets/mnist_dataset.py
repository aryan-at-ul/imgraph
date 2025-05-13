"""
MNIST graph dataset.
"""

import os
import torch
import numpy as np
from torch_geometric.data import Dataset, Data
from torchvision.datasets import MNIST
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt

from imgraph import GraphPresets

class MNISTGraphDataset(Dataset):
    """
    Graph dataset for MNIST.
    
    Parameters
    ----------
    root : str
        Root directory for dataset storage
    preset : str or callable, optional
        Graph preset or custom graph builder function, by default 'slic_mean_color'
    transform : callable, optional
        Transform function applied to graphs, by default None
    pre_transform : callable, optional
        Pre-transform function applied to images before creating graphs, by default None
    force_reload : bool, optional
        Whether to force reload the dataset, by default False
    """
    
    def __init__(self, root, preset='slic_mean_color', transform=None, pre_transform=None, force_reload=False):
        """Initialize the MNIST graph dataset."""
        self.preset = preset
        self.force_reload = force_reload
        
        # Create graph builder
        if isinstance(preset, str):
            if preset == 'slic_mean_color':
                self.graph_builder = GraphPresets.slic_mean_color()
            elif preset == 'slic_color_position':
                self.graph_builder = GraphPresets.slic_color_position()
            elif preset == 'patches_color':
                self.graph_builder = GraphPresets.patches_color()
            elif preset == 'tiny_graph':
                self.graph_builder = GraphPresets.tiny_graph()
            elif preset == 'superpixel_comprehensive':
                self.graph_builder = GraphPresets.superpixel_comprehensive()
            else:
                raise ValueError(f"Unknown preset: {preset}")
        else:
            # Custom graph builder function
            self.graph_builder = preset
        
        # Initialize dataset
        super(MNISTGraphDataset, self).__init__(root, transform, pre_transform)
        
        # Load processed data
        self.data, self.slices = torch.load(self.processed_paths[0])
        
        # Set class names
        self.classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    
    @property
    def raw_file_names(self):
        """List of raw file names."""
        return ['training.pt', 'test.pt']
    
    @property
    def processed_file_names(self):
        """List of processed file names."""
        return [f'mnist_graphs_{self.preset}.pt']
    
    def download(self):
        """Download the MNIST dataset."""
        # Download MNIST
        MNIST(self.raw_dir, train=True, download=True)
        MNIST(self.raw_dir, train=False, download=True)
    
    def process(self):
        """Process the MNIST dataset into graphs."""
        # Check if processed file exists and force_reload is False
        if os.path.exists(self.processed_paths[0]) and not self.force_reload:
            return
        
        # Load MNIST
        train_dataset = MNIST(self.raw_dir, train=True, download=True)
        test_dataset = MNIST(self.raw_dir, train=False, download=True)
        
        # Combine train and test datasets
        data_list = []
        
        # Process training data
        print("Processing MNIST training data...")
        for img, label in tqdm(train_dataset):
            # Convert to numpy array
            img_np = np.array(img)
            
            # Add channel dimension for grayscale
            img_np = np.stack([img_np, img_np, img_np], axis=2)
            
            # Pre-transform if available
            if self.pre_transform is not None:
                img_np = self.pre_transform(img_np)
            
            # Convert to graph
            try:
                graph = self.graph_builder(img_np)
                
                # Add label
                graph.y = torch.tensor(label, dtype=torch.long)
                
                # Add to data list
                data_list.append(graph)
            except Exception as e:
                print(f"Error processing image: {e}")
        
        # Process test data
        print("Processing MNIST test data...")
        for img, label in tqdm(test_dataset):
            # Convert to numpy array
            img_np = np.array(img)
            
            # Add channel dimension for grayscale
            img_np = np.stack([img_np, img_np, img_np], axis=2)
            
            # Pre-transform if available
            if self.pre_transform is not None:
                img_np = self.pre_transform(img_np)
            
            # Convert to graph
            try:
                graph = self.graph_builder(img_np)
                
                # Add label
                graph.y = torch.tensor(label, dtype=torch.long)
                
                # Add to data list
                data_list.append(graph)
            except Exception as e:
                print(f"Error processing image: {e}")
        
        # Save processed data
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
    
    def len(self):
        """
        Get the number of graphs in the dataset.
        
        Returns
        -------
        int
            Number of graphs
        """
        return self.data.y.size(0)
    
    def get(self, idx):
        """
        Get a graph by index.
        
        Parameters
        ----------
        idx : int
            Index of the graph
            
        Returns
        -------
        torch_geometric.data.Data
            Graph data object
        """
        data = Data()
        for key in self.data.keys:
            item, slices = self.data[key], self.slices[key]
            start, end = slices[idx].item(), slices[idx + 1].item()
            if torch.is_tensor(item):
                s = list(item.size())
                if len(s) > 0:
                    s[0] = end - start
                    data[key] = item[start:end].view(s)
                else:
                    data[key] = item[start:end]
            elif start < end:
                data[key] = item[start:end]
        return data
    
    def visualize(self, idx, show_graph=True):
        """
        Visualize a graph.
        
        Parameters
        ----------
        idx : int
            Index of the graph to visualize
        show_graph : bool, optional
            Whether to show the graph representation, by default True
            
        Returns
        -------
        matplotlib.figure.Figure
            Figure with the visualization
        """
        # Get graph
        graph = self.get(idx)
        
        # Get original image from MNIST
        mnist = MNIST(self.raw_dir, train=idx < 60000, download=False)
        img, label = mnist[idx % 60000]
        
        # Convert to numpy array
        img_np = np.array(img)
        
        # Add channel dimension for grayscale
        img_np = np.stack([img_np, img_np, img_np], axis=2)
        
        # Create figure
        if show_graph:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
            
            # Display original image
            ax1.imshow(img_np)
            ax1.set_title(f"Original Image (Class: {label})")
            ax1.axis('off')
            
            # Display graph
            if hasattr(graph, 'node_info') and 'centroids' in graph.node_info:
                node_positions = graph.node_info['centroids']
                
                # Display image in background
                ax2.imshow(img_np)
                
                # Plot nodes
                ax2.scatter(node_positions[:, 1], node_positions[:, 0], c='red', s=10, alpha=0.7)
                
                # Plot edges
                edge_index = graph.edge_index.cpu().numpy()
                for i in range(edge_index.shape[1]):
                    src_idx = edge_index[0, i]
                    dst_idx = edge_index[1, i]
                    
                    src_pos = node_positions[src_idx]
                    dst_pos = node_positions[dst_idx]
                    
                    ax2.plot([src_pos[1], dst_pos[1]], [src_pos[0], dst_pos[0]], 'b-', alpha=0.3, linewidth=0.5)
                
                ax2.set_title(f"Graph: {graph.num_nodes} nodes, {graph.edge_index.shape[1]} edges")
                ax2.axis('off')
            else:
                ax2.text(0.5, 0.5, "Graph does not contain node positions", ha='center', va='center')
                ax2.axis('off')
        else:
            fig, ax = plt.subplots(figsize=(5, 5))
            
            # Display original image
            ax.imshow(img_np)
            ax.set_title(f"Original Image (Class: {label})")
            ax.axis('off')
        
        return fig

    def get_train_test_split(self, train_ratio=0.8, random_seed=42):
        """
        Split the dataset into train and test sets.
        
        Parameters
        ----------
        train_ratio : float, optional
            Ratio of training samples, by default 0.8
        random_seed : int, optional
            Random seed, by default 42
            
        Returns
        -------
        tuple
            (train_dataset, test_dataset)
        """
        from torch_geometric.data import Dataset
        
        # Set random seed
        np.random.seed(random_seed)
        
        # Get indices
        train_indices = np.arange(60000)
        test_indices = np.arange(60000, len(self))
        
        # Create datasets
        train_dataset = torch.utils.data.Subset(self, train_indices)
        test_dataset = torch.utils.data.Subset(self, test_indices)
        
        return train_dataset, test_dataset


def get_mnist_dataset(root='./data', train=True, transform=None):
    """
    Get the MNIST dataset.
    
    Args:
        root (str): Root directory where the dataset should be stored
        train (bool): If True, creates dataset from training set, otherwise creates from test set
        transform (callable, optional): A function/transform that takes in an PIL image and returns a transformed version
        
    Returns:
        MNISTGraphDataset: The MNIST dataset
    """
    return MNISTGraphDataset(root=root, train=train, transform=transform)