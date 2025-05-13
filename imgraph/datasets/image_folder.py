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
import gc  # Garbage collection

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
    max_in_memory : int, optional
        Maximum number of graphs to keep in memory at once, by default 1000
    """
    
    def __init__(self, root, preset='slic_mean_color', transform=None, pre_transform=None, 
                 image_transform=None, force_reload=False, max_in_memory=1000):
        """Initialize the ImageFolder graph dataset."""
        self.preset = preset
        self.force_reload = force_reload
        self.image_transform = image_transform or transforms.ToTensor()
        self.max_in_memory = max_in_memory
        
        # Only store image indices and labels, not the actual graphs
        self.all_indices = []
        self.all_labels = []
        self.processed_path = None
        
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
        
        try:
            self.image_dataset = ImageFolder(self.raw_dirpath, transform=self.image_transform)
            # Set class names
            self.classes = self.image_dataset.classes
            self.class_to_idx = self.image_dataset.class_to_idx
            print(f"Found ImageFolder dataset with {len(self.image_dataset)} images and {len(self.classes)} classes")
        except (FileNotFoundError, RuntimeError) as e:
            print(f"Warning: Could not create ImageFolder dataset: {e}")
            self.image_dataset = None
            self.classes = []
            self.class_to_idx = {}
            print("No image dataset created!")
        
        # Initialize dataset
        super(ImageFolderGraphDataset, self).__init__(root, transform, pre_transform)
        
        # Process images
        self.process()
        
        print(f"Dataset initialized with {len(self.all_indices)} images")
    
    @property
    def raw_dir(self):
        """Get the raw directory."""
        return self.raw_dirpath
    
    @property
    def processed_dir(self):
        """Get the processed directory."""
        processed_dirpath = os.path.join(self.root, 'processed')
        os.makedirs(processed_dirpath, exist_ok=True)
        return processed_dirpath
    
    @property
    def raw_file_names(self):
        """List of raw file names."""
        return []
    
    @property
    def processed_file_names(self):
        """List of processed file names."""
        return ['dataset_info.pt']
    
    def download(self):
        """Download the dataset (not required for ImageFolder)."""
        pass
    
    def process(self):
        """Process the dataset into graphs."""
        processed_path = self.processed_paths[0]
        self.processed_path = processed_path
        
        # Check if the dataset has already been processed
        if os.path.exists(processed_path) and not self.force_reload:
            try:
                # Load dataset info
                dataset_info = torch.load(processed_path)
                self.all_indices = dataset_info['indices']
                self.all_labels = dataset_info['labels']
                print(f"Loaded dataset info with {len(self.all_indices)} samples")
                return
            except Exception as e:
                print(f"Failed to load dataset info: {e}")
                print("Will process dataset from scratch")
        
        # Check if we have an image dataset
        if self.image_dataset is None or len(self.image_dataset) == 0:
            print("No images found in the dataset. Cannot process.")
            return
        
        # Process images
        print(f"Processing {len(self.image_dataset)} images into graphs...")
        indices = []
        labels = []
        
        for idx in tqdm(range(len(self.image_dataset))):
            try:
                # Get image and label
                _, label = self.image_dataset[idx]
                
                # Store index and label
                indices.append(idx)
                labels.append(label)
                
                # Process in small batches to save memory
                if len(indices) % 100 == 0:
                    # Save progress periodically
                    dataset_info = {
                        'indices': indices,
                        'labels': labels
                    }
                    torch.save(dataset_info, processed_path)
            
            except Exception as e:
                print(f"Error processing image {idx}: {e}")
        
        # Save final dataset info
        dataset_info = {
            'indices': indices,
            'labels': labels
        }
        torch.save(dataset_info, processed_path)
        
        # Set instance variables
        self.all_indices = indices
        self.all_labels = labels
        
        print(f"Processed {len(indices)} images successfully")
        print("Done!")
    
    def _process_image(self, idx):
        """Process an image into a graph."""
        if self.image_dataset is None:
            raise RuntimeError("No image dataset available")
            
        # Get image and label
        img, label = self.image_dataset[idx]
        
        # Convert to numpy array if needed
        if isinstance(img, torch.Tensor):
            img = img.permute(1, 2, 0).numpy()
        
        # Apply pre-transform if provided
        if self.pre_transform is not None:
            img = self.pre_transform(img)
        
        # Convert to graph
        graph = self.graph_builder(img)
        
        # Add label
        graph.y = torch.tensor(label, dtype=torch.long)
        
        return graph
    
    def len(self):
        """Get the number of graphs in the dataset."""
        return len(self.all_indices)
    
    def get(self, idx):
        """Get a graph by index."""
        if idx >= len(self.all_indices):
            raise IndexError(f"Index {idx} out of range for dataset with {len(self.all_indices)} samples")
            
        # Process the image on-the-fly to save memory
        image_idx = self.all_indices[idx]
        
        # Process the image
        data = self._process_image(image_idx)
        
        # Apply transform if needed
        if self.transform is not None:
            data = self.transform(data)
            
        return data
    
    def get_train_test_split(self, train_ratio=0.8, stratify=True, random_seed=42):
        """Split the dataset into train and test sets."""
        # Check if we have enough data
        if len(self) < 2:
            raise ValueError(f"Dataset has only {len(self)} samples, need at least 2 for splitting")
            
        from torch.utils.data import Subset
        from sklearn.model_selection import train_test_split
        
        # Get indices and labels
        indices = list(range(len(self)))
        labels = self.all_labels
        
        # Split indices
        if stratify and len(set(labels)) > 1:
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