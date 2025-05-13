"""
MedMNIST graph dataset.
"""

import os
import torch
import numpy as np
from torch_geometric.data import Dataset, Data
from tqdm import tqdm
import matplotlib.pyplot as plt

from imgraph import GraphPresets

class MedMNISTGraphDataset(Dataset):
    """
    Graph dataset for MedMNIST.
    
    Parameters
    ----------
    root : str
        Root directory for dataset storage
    name : str
        Name of the MedMNIST dataset (e.g., 'pathmnist', 'dermamnist', etc.)
    preset : str or callable, optional
        Graph preset or custom graph builder function, by default 'slic_color_position'
    transform : callable, optional
        Transform function applied to graphs, by default None
    pre_transform : callable, optional
        Pre-transform function applied to images before creating graphs, by default None
    force_reload : bool, optional
        Whether to force reload the dataset, by default False
    """
    
    def __init__(self, root, name, preset='slic_color_position', transform=None, pre_transform=None, force_reload=False):
        """Initialize the MedMNIST graph dataset."""
        self.name = name.lower()
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
        super(MedMNISTGraphDataset, self).__init__(root, transform, pre_transform)
        
        # Load processed data
        self.data, self.slices = torch.load(self.processed_paths[0])
    
    @property
    def raw_dir(self):
        """
        Get the raw directory.
        
        Returns
        -------
        str
            Raw directory
        """
        return os.path.join(self.root, 'raw')
    
    @property
    def processed_dir(self):
        """
        Get the processed directory.
        
        Returns
        -------
        str
            Processed directory
        """
        return os.path.join(self.root, 'processed')
    
    @property
    def raw_file_names(self):
        """List of raw file names."""
        return [
            f'{self.name}_train.npz',
            f'{self.name}_val.npz',
            f'{self.name}_test.npz'
        ]
    
    @property
    def processed_file_names(self):
        """List of processed file names."""
        return [f'{self.name}_graphs_{self.preset}.pt']
    
    def download(self):
        """Download the MedMNIST dataset."""
        try:
            import medmnist
            from medmnist import INFO
        except ImportError:
            raise ImportError("MedMNIST is required. Install it with pip install medmnist")
        
        # Check if dataset exists
        for file_name in self.raw_file_names:
            file_path = os.path.join(self.raw_dir, file_name)
            if not os.path.exists(file_path):
                # Create directory if it doesn't exist
                if not os.path.exists(self.raw_dir):
                    os.makedirs(self.raw_dir)
                
                # Download dataset
                print(f"Downloading {file_name}...")
                
                # Get dataset info
                dataset_info = INFO[self.name]
                
                # Get dataset class
                DataClass = getattr(medmnist, dataset_info['python_class'])
                
                # Download train set
                if 'train' in file_name:
                    DataClass(split='train', download=True, root=self.raw_dir)
                
                # Download validation set
                elif 'val' in file_name:
                    DataClass(split='val', download=True, root=self.raw_dir)
                
                # Download test set
                elif 'test' in file_name:
                    DataClass(split='test', download=True, root=self.raw_dir)
    
    def process(self):
        """Process the MedMNIST dataset into graphs."""
        # Check if processed file exists and force_reload is False
        if os.path.exists(self.processed_paths[0]) and not self.force_reload:
            return
        
        try:
            import medmnist
            from medmnist import INFO
        except ImportError:
            raise ImportError("MedMNIST is required. Install it with pip install medmnist")
        
        # Get dataset info
        dataset_info = INFO[self.name]
        
        # Get class names
        if 'task' in dataset_info and dataset_info['task'] == 'multi-label, binary-class':
            self.task = 'multi-label'
            self.classes = dataset_info.get('label', {})
        else:
            self.task = 'multi-class'
            self.classes = dataset_info.get('label', {})
        
        # Process data
        data_list = []
        
        # Process each split
        for split in ['train', 'val', 'test']:
            # Load data
            data_path = os.path.join(self.raw_dir, f'{self.name}_{split}.npz')
            data = np.load(data_path)
            
            images = data['images']
            labels = data['labels']
            
            print(f"Processing {split} set ({len(images)} images)...")
            
            for i, (image, label) in enumerate(tqdm(zip(images, labels), total=len(images))):
                # Convert to RGB if needed
                if image.shape[-1] != 3:
                    image = np.stack([image.squeeze()] * 3, axis=-1)
                
                # Pre-transform if available
                if self.pre_transform is not None:
                    image = self.pre_transform(image)
                
                # Convert to graph
                try:
                    graph = self.graph_builder(image)
                    
                    # Add label
                    if self.task == 'multi-label':
                        graph.y = torch.tensor(label, dtype=torch.float)
                    else:
                        graph.y = torch.tensor(label.squeeze(), dtype=torch.long)
                    
                    # Add split info
                    graph.split = split
                    
                    # Add to data list
                    data_list.append(graph)
                except Exception as e:
                    print(f"Error processing image {split}/{i}: {e}")
        
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
        
        # Get image
        try:
            import medmnist
            from medmnist import INFO
            
            # Get dataset info
            dataset_info = INFO[self.name]
            
            # Get dataset class
            DataClass = getattr(medmnist, dataset_info['python_class'])
            
            # Get split
            split = graph.split if hasattr(graph, 'split') else 'train'
            
            # Load dataset
            medmnist_dataset = DataClass(split=split, root=self.raw_dir)
            
            # Find index in the original dataset
            # This is an approximation as we don't store the original index
            label = graph.y.cpu().numpy()
            
            # Load all images and labels from the dataset
            images = medmnist_dataset.imgs
            labels = medmnist_dataset.labels
            
            # Find the first image with the same label
            # This is an approximation
            img_idx = None
            for i, l in enumerate(labels):
                if np.array_equal(l, label):
                    img_idx = i
                    break
            
            if img_idx is not None:
                # Get image
                img = images[img_idx]
                
                # Convert to RGB if needed
                if img.shape[-1] != 3:
                    img = np.stack([img.squeeze()] * 3, axis=-1)
            else:
                # Use a synthetic image
                img = np.ones((28, 28, 3)) * 200
        except:
            # Use a synthetic image if MedMNIST is not available
            img = np.ones((28, 28, 3)) * 200
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        # Display original image
        ax1.imshow(img)
        
        # Get label text
        if self.task == 'multi-label':
            label_text = ', '.join([self.classes.get(str(i), str(i)) for i, l in enumerate(graph.y) if l > 0.5])
        else:
            label_idx = graph.y.item()
            label_text = self.classes.get(str(label_idx), str(label_idx))
        
        ax1.set_title(f"Original Image (Class: {label_text})")
        ax1.axis('off')
        
        # Display graph
        if hasattr(graph, 'node_info') and 'centroids' in graph.node_info:
            node_positions = graph.node_info['centroids']
            
            # Display image in background
            ax2.imshow(img)
            
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
    
    def get_splits(self):
        """
        Get train, validation, and test datasets.
        
        Returns
        -------
        tuple
            (train_dataset, val_dataset, test_dataset)
        """
        from torch.utils.data import Subset
        
        # Get indices for each split
        train_indices = []
        val_indices = []
        test_indices = []
        
        for i in range(len(self)):
            graph = self.get(i)
            if hasattr(graph, 'split'):
                if graph.split == 'train':
                    train_indices.append(i)
                elif graph.split == 'val':
                    val_indices.append(i)
                elif graph.split == 'test':
                    test_indices.append(i)
            else:
                # If no split info, assume train
                train_indices.append(i)
        
        # Create datasets
        train_dataset = Subset(self, train_indices)
        val_dataset = Subset(self, val_indices)
        test_dataset = Subset(self, test_indices)
        
        return train_dataset, val_dataset, test_dataset