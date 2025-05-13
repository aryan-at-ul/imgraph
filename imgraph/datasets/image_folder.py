"""
Image folder dataset for graph neural networks.
"""

import os
import torch
import numpy as np
from torch_geometric.data import Dataset, Data
from torchvision.datasets import ImageFolder
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt

from imgraph import GraphPresets

class ImageFolderGraphDataset(Dataset):
    """
    Graph dataset for image folders.
    
    Parameters
    ----------
    root : str
        Root directory containing the image folder
    preset : str or callable, optional
        Graph preset or custom graph builder function, by default 'slic_mean_color'
    transform : callable, optional
        Transform function applied to graphs, by default None
    pre_transform : callable, optional
        Pre-transform function applied to images before creating graphs, by default None
    image_transform : callable, optional
        Transform function applied to images, by default None
    force_reload : bool, optional
        Whether to force reload the dataset, by default False
    """
    
    def __init__(self, root, preset='slic_mean_color', transform=None, pre_transform=None, 
                 image_transform=None, force_reload=False):
        """Initialize the ImageFolder graph dataset."""
        self.preset = preset
        self.force_reload = force_reload
        self.image_transform = image_transform
        
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
        super(ImageFolderGraphDataset, self).__init__(root, transform, pre_transform)
        
        # Load processed data
        self.data, self.slices = torch.load(self.processed_paths[0])
        
        # Create ImageFolder dataset for reference
        image_transform = image_transform or transforms.ToTensor()
        self.image_dataset = ImageFolder(self.raw_dir, transform=image_transform)
        
        # Set class names
        self.classes = self.image_dataset.classes
        self.class_to_idx = self.image_dataset.class_to_idx
    
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
        # ImageFolder has no specific raw files
        return []
    
    @property
    def processed_file_names(self):
        """List of processed file names."""
        return [f'image_folder_graphs_{self.preset}.pt']
    
    def download(self):
        """Download the dataset (not required for ImageFolder)."""
        # No download needed for ImageFolder
        pass
    
    def process(self):
        """Process the dataset into graphs."""
        # Check if processed file exists and force_reload is False
        if os.path.exists(self.processed_paths[0]) and not self.force_reload:
            return
        
        # Create image dataset
        image_transform = self.image_transform or transforms.ToTensor()
        dataset = ImageFolder(self.raw_dir, transform=image_transform)
        
        # Process images
        data_list = []
        print("Processing images...")
        for idx in tqdm(range(len(dataset))):
            # Get image and label
            img, label = dataset[idx]
            
            # Convert to numpy array if needed
            if isinstance(img, torch.Tensor):
                img = img.permute(1, 2, 0).numpy()
            
            # Pre-transform if available
            if self.pre_transform is not None:
                img = self.pre_transform(img)
            
            # Convert to graph
            try:
                graph = self.graph_builder(img)
                
                # Add label
                graph.y = torch.tensor(label, dtype=torch.long)
                
                # Add file path for reference
                path, _ = dataset.samples[idx]
                graph.path = path
                
                # Add to data list
                data_list.append(graph)
            except Exception as e:
                print(f"Error processing image {idx}: {e}")
        
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
            if key == 'path':
                data[key] = item[slices[idx]:slices[idx + 1]]
                continue
            
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
        
        # Get image path
        if hasattr(graph, 'path'):
            img_path = graph.path[0]
            
            # Load image
            try:
                from PIL import Image
                img = Image.open(img_path).convert('RGB')
                img_np = np.array(img)
            except:
                # Fallback if image can't be loaded
                img, _ = self.image_dataset[idx]
                if isinstance(img, torch.Tensor):
                    img_np = img.permute(1, 2, 0).numpy()
                else:
                    img_np = np.array(img)
        else:
            # Get image directly from dataset
            img, _ = self.image_dataset[idx]
            if isinstance(img, torch.Tensor):
                img_np = img.permute(1, 2, 0).numpy()
            else:
                img_np = np.array(img)
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        # Display original image
        ax1.imshow(img_np)
        class_name = self.classes[graph.y.item()] if hasattr(self, 'classes') else str(graph.y.item())
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
    
    def get_train_test_split(self, train_ratio=0.8, stratify=True, random_seed=42):
        """
        Split the dataset into train and test sets.
        
        Parameters
        ----------
        train_ratio : float, optional
            Ratio of training samples, by default 0.8
        stratify : bool, optional
            Whether to use stratified sampling based on class labels, by default True
        random_seed : int, optional
            Random seed, by default 42
            
        Returns
        -------
        tuple
            (train_dataset, test_dataset)
        """
        from torch.utils.data import Subset
        from sklearn.model_selection import train_test_split
        
        # Get indices and labels
        indices = list(range(len(self)))
        labels = [self.get(i).y.item() for i in indices]
        
        # Split indices
        if stratify:
            train_indices, test_indices = train_test_split(
                indices, train_size=train_ratio, stratify=labels, random_state=random_seed
            )
        else:
            train_indices, test_indices = train_test_split(
                indices, train_size=train_ratio, random_state=random_seed
            )
        
        # Create datasets
        train_dataset = Subset(self, train_indices)
        test_dataset = Subset(self, test_indices)
        
        return train_dataset, test_dataset
