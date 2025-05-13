"""
Image folder dataset for graph neural networks.
"""

import os
import torch
import numpy as np
from torch_geometric.data import Dataset
from torchvision.datasets import ImageFolder
from torchvision import transforms
from tqdm import tqdm

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
        self.processed_data = []
        
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
        
        # Create ImageFolder dataset for reference
        self.raw_dirpath = os.path.join(root, 'raw')
        os.makedirs(self.raw_dirpath, exist_ok=True)
        
        image_transform = image_transform or transforms.ToTensor()
        try:
            self.image_dataset = ImageFolder(self.raw_dirpath, transform=image_transform)
            # Set class names
            self.classes = self.image_dataset.classes
            self.class_to_idx = self.image_dataset.class_to_idx
        except (FileNotFoundError, RuntimeError) as e:
            print(f"Warning: Could not create ImageFolder dataset: {e}")
            self.image_dataset = None
            self.classes = []
            self.class_to_idx = {}
        
        # Initialize dataset
        super(ImageFolderGraphDataset, self).__init__(root, transform, pre_transform)
        
        # Load data
        self._load_or_process_data()
    
    def _load_or_process_data(self):
        """Load or process data depending on what's available."""
        processed_path = self.processed_paths[0]
        if os.path.exists(processed_path) and not self.force_reload:
            try:
                # Load individual graph files
                self._load_processed_graphs()
            except Exception as e:
                print(f"Error loading processed graphs: {e}")
                print("Processing data from scratch...")
                self.process()
        else:
            print("Processing data from scratch...")
            self.process()
    
    def _load_processed_graphs(self):
        """Load processed graphs."""
        # Find all processed graph files
        processed_dir = self.processed_dir
        graph_files = [f for f in os.listdir(processed_dir) if f.endswith('.pt') and f.startswith('graph_')]
        
        # Load graphs
        for file in graph_files:
            self.processed_data.append(torch.load(os.path.join(processed_dir, file)))
        
        print(f"Loaded {len(self.processed_data)} processed graphs")
    
    @property
    def raw_dir(self):
        """
        Get the raw directory.
        
        Returns
        -------
        str
            Raw directory
        """
        return self.raw_dirpath
    
    @property
    def processed_dir(self):
        """
        Get the processed directory.
        
        Returns
        -------
        str
            Processed directory
        """
        processed_dirpath = os.path.join(self.root, 'processed')
        os.makedirs(processed_dirpath, exist_ok=True)
        return processed_dirpath
    
    @property
    def raw_file_names(self):
        """List of raw file names."""
        # ImageFolder has no specific raw files
        return []
    
    @property
    def processed_file_names(self):
        """List of processed file names."""
        return ['done.txt']  # Marker file to indicate processing is done
    
    def download(self):
        """Download the dataset (not required for ImageFolder)."""
        # No download needed for ImageFolder
        pass
    
    def process(self):
        """Process the dataset into graphs."""
        # Check if processed file exists and force_reload is False
        processed_marker = self.processed_paths[0]
        if os.path.exists(processed_marker) and not self.force_reload:
            return
        
        # Create image dataset if it doesn't exist
        if self.image_dataset is None:
            image_transform = self.image_transform or transforms.ToTensor()
            try:
                self.image_dataset = ImageFolder(self.raw_dir, transform=image_transform)
                # Set class names
                self.classes = self.image_dataset.classes
                self.class_to_idx = self.image_dataset.class_to_idx
            except (FileNotFoundError, RuntimeError) as e:
                raise RuntimeError(f"Failed to create ImageFolder dataset: {e}")
        
        # Process images to individual graph files
        print("Processing images...")
        self.processed_data = []
        
        try:
            for idx in tqdm(range(len(self.image_dataset))):
                # Get image and label
                img, label = self.image_dataset[idx]
                
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
                    path, _ = self.image_dataset.samples[idx]
                    graph.path = path
                    
                    # Save the individual graph
                    output_path = os.path.join(self.processed_dir, f'graph_{idx}.pt')
                    torch.save(graph, output_path)
                    
                    # Add to processed data list
                    self.processed_data.append(graph)
                except Exception as e:
                    print(f"Error processing image {idx}: {e}")
            
            # Save a marker file to indicate processing is complete
            if len(self.processed_data) > 0:
                with open(processed_marker, 'w') as f:
                    f.write(f"Processed {len(self.processed_data)} graphs")
                
                print(f"Saved {len(self.processed_data)} processed graphs")
            else:
                raise ValueError("No graphs were created from the images")
                
        except Exception as e:
            # If anything goes wrong during processing, remove any potentially partial processed files
            for f in os.listdir(self.processed_dir):
                if f.startswith('graph_'):
                    os.remove(os.path.join(self.processed_dir, f))
            
            if os.path.exists(processed_marker):
                os.remove(processed_marker)
                
            raise RuntimeError(f"Error during processing: {e}")
    
    def len(self):
        """
        Get the number of graphs in the dataset.
        
        Returns
        -------
        int
            Number of graphs
        """
        return len(self.processed_data)
    
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
        data = self.processed_data[idx]
        
        if self.transform is not None:
            data = self.transform(data)
            
        return data
    
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
        labels = [self[i].y.item() for i in indices]
        
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