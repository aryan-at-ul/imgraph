"""
Pipeline for processing folders of images into graphs.
"""

import os
import glob
import json
import torch
import numpy as np
from tqdm import tqdm
from torch_geometric.data import Dataset, Data
from imgraph.pipeline.config_pipeline import GraphPipeline

class FolderGraphPipeline:
    """
    A pipeline for processing folders of images into graphs.
    
    Parameters
    ----------
    input_dir : str
        Path to input directory containing images
    output_dir : str
        Path to output directory for saving graphs
    config : dict or str, optional
        Configuration dictionary or path to configuration file, by default None (uses default config)
    image_extensions : list, optional
        List of image file extensions to process, by default ['.jpg', '.jpeg', '.png']
    load_fn : callable, optional
        Function to load images, by default None (uses default loader)
    """
    
    def __init__(self, input_dir, output_dir, config=None, image_extensions=None, load_fn=None):
        """Initialize the FolderGraphPipeline."""
        self.input_dir = input_dir
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Set image extensions
        self.image_extensions = image_extensions or ['.jpg', '.jpeg', '.png']
        
        # Set image loader
        self.load_fn = load_fn or self._default_loader
        
        # Create graph pipeline
        if config is None:
            self.graph_pipeline = GraphPipeline(GraphPipeline.create_default_config())
        else:
            self.graph_pipeline = GraphPipeline(config)
        
        # Save config to output directory
        self.graph_pipeline.save_config(os.path.join(output_dir, 'graph_config.json'))
    
    def _default_loader(self, path):
        """
        Default image loader.
        
        Parameters
        ----------
        path : str
            Path to image file
            
        Returns
        -------
        numpy.ndarray
            Loaded image with shape (H, W, C)
        """
        try:
            import cv2
            # Load image in BGR format
            img = cv2.imread(path)
            # Convert to RGB
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img
        except ImportError:
            try:
                from PIL import Image
                img = Image.open(path).convert('RGB')
                return np.array(img)
            except ImportError:
                raise ImportError("Either OpenCV or PIL is required for loading images")
    
    def get_image_files(self):
        """
        Get list of image files in the input directory.
        
        Returns
        -------
        list
            List of image file paths
        """
        image_files = []
        
        for ext in self.image_extensions:
            image_files.extend(glob.glob(os.path.join(self.input_dir, f'**/*{ext}'), recursive=True))
        
        return sorted(image_files)
    
    def process(self, verbose=True):
        """
        Process all images in the input directory.
        
        Parameters
        ----------
        verbose : bool, optional
            Whether to display progress bar, by default True
            
        Returns
        -------
        list
            List of output file paths
        """
        # Get image files
        image_files = self.get_image_files()
        
        # Check if there are any image files
        if not image_files:
            print(f"No images found in {self.input_dir} with extensions {self.image_extensions}")
            return []
        
        # Process images
        output_files = []
        
        # Create progress bar if verbose
        if verbose:
            pbar = tqdm(total=len(image_files), desc="Processing images")
        
        for image_path in image_files:
            # Get relative path
            rel_path = os.path.relpath(image_path, self.input_dir)
            
            # Create output path
            output_path = os.path.join(self.output_dir, os.path.splitext(rel_path)[0] + '.pt')
            
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Load image
            try:
                image = self.load_fn(image_path)
                
                # Skip if image loading failed
                if image is None:
                    print(f"Failed to load image: {image_path}")
                    if verbose:
                        pbar.update(1)
                    continue
                
                # Process image
                graph = self.graph_pipeline.process(image)
                
                # Save graph
                torch.save(graph, output_path)
                
                # Add to output files
                output_files.append(output_path)
                
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
            
            # Update progress bar
            if verbose:
                pbar.update(1)
        
        # Close progress bar
        if verbose:
            pbar.close()
        
        return output_files
    
    def create_dataset(self, with_labels=False, label_fn=None):
        """
        Create a PyTorch Geometric dataset from the processed graphs.
        
        Parameters
        ----------
        with_labels : bool, optional
            Whether to include labels, by default False
        label_fn : callable, optional
            Function to extract labels from file paths, by default None
            
        Returns
        -------
        torch_geometric.data.Dataset
            Dataset of graphs
        """
        return FolderGraphDataset(
            self.output_dir,
            with_labels=with_labels,
            label_fn=label_fn,
            transform=None,
            pre_transform=None
        )

class FolderGraphDataset(Dataset):
    """
    PyTorch Geometric dataset for graph data stored in a folder.
    
    Parameters
    ----------
    root : str
        Root directory where the processed graphs are stored
    with_labels : bool, optional
        Whether to include labels, by default False
    label_fn : callable, optional
        Function to extract labels from file paths, by default None
    transform : callable, optional
        Transform function applied to each data object, by default None
    pre_transform : callable, optional
        Pre-transform function applied to each data object, by default None
    """
    
    def __init__(self, root, with_labels=False, label_fn=None, transform=None, pre_transform=None):
        """Initialize the FolderGraphDataset."""
        self.root = root
        self.with_labels = with_labels
        self.label_fn = label_fn
        self.graph_files = glob.glob(os.path.join(root, '**/*.pt'), recursive=True)
        
        # Ensure deterministic order
        self.graph_files = sorted(self.graph_files)
        
        super(FolderGraphDataset, self).__init__(root, transform, pre_transform)
    
    @property
    def raw_file_names(self):
        """List of raw file names."""
        return []
    
    @property
    def processed_file_names(self):
        """List of processed file names."""
        return [os.path.basename(f) for f in self.graph_files]
    
    def download(self):
        """Download method (not used)."""
        pass
    
    def process(self):
        """Process method (not used)."""
        pass
    
    def len(self):
        """
        Get the number of graphs in the dataset.
        
        Returns
        -------
        int
            Number of graphs
        """
        return len(self.graph_files)
    
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
        # Load graph
        graph = torch.load(self.graph_files[idx])
        
        # Add label if required
        if self.with_labels and not hasattr(graph, 'y'):
            if self.label_fn is not None:
                label = self.label_fn(self.graph_files[idx])
                graph.y = torch.tensor(label, dtype=torch.long)
            else:
                # Try to extract label from directory name
                parent_dir = os.path.basename(os.path.dirname(self.graph_files[idx]))
                try:
                    label = int(parent_dir)
                    graph.y = torch.tensor(label, dtype=torch.long)
                except ValueError:
                    # If parent directory name is not an integer, use it as a string label
                    graph.y = parent_dir
        
        return graph