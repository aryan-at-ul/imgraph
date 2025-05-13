"""
Standard dataset wrappers for common image datasets like CIFAR.
"""

import os
import torch
import numpy as np
from torch_geometric.data import Dataset, Data
from torchvision.datasets import CIFAR10, CIFAR100
from tqdm import tqdm
import matplotlib.pyplot as plt

from imgraph import GraphPresets

class StandardGraphDataset(Dataset):
    """
    Base class for standard graph datasets.
    
    Parameters
    ----------
    root : str
        Root directory for dataset storage
    dataset_cls : class
        Dataset class from torchvision
    preset : str or callable, optional
        Graph preset or custom graph builder function, by default 'slic_mean_color'
    transform : callable, optional
        Transform function applied to graphs, by default None
    pre_transform : callable, optional
        Pre-transform function applied to images before creating graphs, by default None
    force_reload : bool, optional
        Whether to force reload the dataset, by default False
    """
    
    def __init__(self, root, dataset_cls, preset='slic_mean_color', transform=None, pre_transform=None, force_reload=False):
        """Initialize the standard graph dataset."""
        self.dataset_cls = dataset_cls
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
        super(StandardGraphDataset, self).__init__(root, transform, pre_transform)
        
        # Load processed data
        self.data, self.slices = torch.load(self.processed_paths[0])
    
    @property
    def raw_file_names(self):
        """List of raw file names."""
        return []
    
    @property
    def processed_file_names(self):
        """List of processed file names."""
        dataset_name = self.dataset_cls.__name__.lower()
        return [f'{dataset_name}_graphs_{self.preset}.pt']
    
    def download(self):
        """Download the dataset."""
        # Download using torchvision
        self.dataset_cls(self.raw_dir, train=True, download=True)
        self.dataset_cls(self.raw_dir, train=False, download=True)
    
    def process(self):
        """Process the dataset into graphs."""
        # Check if processed file exists and force_reload is False
        if os.path.exists(self.processed_paths[0]) and not self.force_reload:
            return
        
        # Load dataset
        train_dataset = self.dataset_cls(self.raw_dir, train=True, download=True)
        test_dataset = self.dataset_cls(self.raw_dir, train=False, download=True)
        
        # Store class names
        if hasattr(train_dataset, 'classes'):
            self.classes = train_dataset.classes
        
        # Combine train and test datasets
        data_list = []
        
        # Process training data
        print(f"Processing {self.dataset_cls.__name__} training data...")
        for img, label in tqdm(train_dataset):
            # Convert to numpy array
            img_np = np.array(img)
            
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
        print(f"Processing {self.dataset_cls.__name__} test data...")
        for img, label in tqdm(test_dataset):
            # Convert to numpy array
            img_np = np.array(img)
            
            # Pre-transform if available
            if self.pre_transform is not None:
                img_np = self.pre_transform(img_np)
            
            # Convert to graph
            try:
                graph = self.graph_builder(img_np)
                
                # Add label
                graph.y = torch.tensor(label, dtype=torch.long)
                
                # Add metadata for train/test split
                graph.is_test = torch.tensor([True], dtype=torch.bool)
                
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
    
    def visualize(self, idx):
        """
        Visualize a graph.
        
        Parameters
        ----------
        idx : int
            Index of the graph to visualize
            
        Returns
        -------
        matplotlib.figure.Figure
            Figure with the visualization
        """
        # Get graph
        graph = self.get(idx)
        
        # Get image from dataset
        is_train = not hasattr(graph, 'is_test') or not graph.is_test.item()
        dataset = self.dataset_cls(self.raw_dir, train=is_train, download=False)
        
        # Calculate index in the original dataset
        if is_train:
            orig_idx = idx
        else:
            # Count training samples
            train_dataset = self.dataset_cls(self.raw_dir, train=True, download=False)
            orig_idx = idx - len(train_dataset)
        
        # Get image
        img, label = dataset[orig_idx]
        img_np = np.array(img)
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        # Display original image
        ax1.imshow(img_np)
        class_name = self.classes[label] if hasattr(self, 'classes') else str(label)
        ax1.set_title(f"Original Image (Class: {class_name})")
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
        
        return fig
    
    def get_train_test_split(self):
        """
        Split the dataset into train and test sets.
        
        Returns
        -------
        tuple
            (train_dataset, test_dataset)
        """
        from torch.utils.data import Subset
        
        # Get indices
        train_indices = []
        test_indices = []
        
        for i in range(len(self)):
            graph = self.get(i)
            if hasattr(graph, 'is_test') and graph.is_test.item():
                test_indices.append(i)
            else:
                train_indices.append(i)
        
        # Create datasets
        train_dataset = Subset(self, train_indices)
        test_dataset = Subset(self, test_indices)
        
        return train_dataset, test_dataset


class CIFAR10GraphDataset(StandardGraphDataset):
    """
    Graph dataset for CIFAR-10.
    
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
        """Initialize the CIFAR-10 graph dataset."""
        super(CIFAR10GraphDataset, self).__init__(
            root, CIFAR10, preset, transform, pre_transform, force_reload
        )
        
        # Set class names
        self.classes = [
            'airplane', 'automobile', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck'
        ]


class CIFAR100GraphDataset(StandardGraphDataset):
    """
    Graph dataset for CIFAR-100.
    
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
        """Initialize the CIFAR-100 graph dataset."""
        super(CIFAR100GraphDataset, self).__init__(
            root, CIFAR100, preset, transform, pre_transform, force_reload
        )
        
        # Set class names from CIFAR-100
        cifar = CIFAR100(root, download=False)
        self.classes = cifar.classes
