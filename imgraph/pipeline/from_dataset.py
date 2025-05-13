import os 
import os.path as osp  
from typing import Optional
# from imgraph.datasets import get_minst_dataset, get_pneumonia_dataset
from imgraph.datasets import get_mnist_dataset#, get_pneumonia_dataset





def load_saved_datasets(dataset_name: str, super_pixels : Optional[int] = 10, feature_extractor : Optional[str] = 'resnet18' ,root: Optional[str] = None) -> None:
    r"""Loads the dataset from the local cache.

    Args:
        dataset_name (str): The name of the dataset.
        root (str, optional): The root directory where the dataset should be saved.
            (default: :obj:`None`)
    Returns:
        The dataset object.
    """

    if dataset_name.lower() == 'mnist':
        train_dataset, test_dataset = get_mnist_dataset()
        return train_dataset, test_dataset
    
    # if dataset_name.lower() == 'pneumonia':
    #     train_dataset, test_dataset = get_pneumonia_dataset(super_pixels,feature_extractor)
    #     return train_dataset, test_dataset
    


    """
Functions for loading and processing datasets.
"""

import os
import torch
from tqdm import tqdm
from imgraph.pipeline.config_pipeline import GraphPipeline

def load_saved_datasets(dataset_dir):
    """
    Load saved graph datasets.
    
    Parameters
    ----------
    dataset_dir : str
        Directory containing saved graph datasets
        
    Returns
    -------
    dict
        Dictionary of datasets
    """
    datasets = {}
    
    # List all files in the directory
    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            if file.endswith('.pt'):
                # Load dataset
                filepath = os.path.join(root, file)
                dataset_name = os.path.splitext(file)[0]
                
                try:
                    dataset = torch.load(filepath)
                    datasets[dataset_name] = dataset
                    print(f"Loaded dataset: {dataset_name}")
                except Exception as e:
                    print(f"Error loading dataset {dataset_name}: {e}")
    
    return datasets

def process_dataset(dataset, graph_pipeline, verbose=True):
    """
    Process a dataset into graphs.
    
    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        Dataset to process
    graph_pipeline : GraphPipeline
        Graph pipeline to use for processing
    verbose : bool, optional
        Whether to display progress bar, by default True
        
    Returns
    -------
    list
        List of graph data objects
    """
    graphs = []
    
    # Create progress bar if verbose
    if verbose:
        pbar = tqdm(total=len(dataset), desc="Processing dataset")
    
    for i in range(len(dataset)):
        # Get data
        data = dataset[i]
        
        # Extract image and label
        if isinstance(data, tuple):
            image, label = data
        else:
            image = data
            label = None
        
        # Convert to numpy array if needed
        if isinstance(image, torch.Tensor):
            image = image.permute(1, 2, 0).numpy()
        
        # Process image
        try:
            graph = graph_pipeline.process(image)
            
            # Add label if available
            if label is not None:
                if isinstance(label, torch.Tensor):
                    graph.y = label
                else:
                    graph.y = torch.tensor(label, dtype=torch.long)
            
            # Add to list
            graphs.append(graph)
            
        except Exception as e:
            print(f"Error processing image {i}: {e}")
        
        # Update progress bar
        if verbose:
            pbar.update(1)
    
    # Close progress bar
    if verbose:
        pbar.close()
    
    return graphs

def create_graph_pipeline(config=None):
    """
    Create a graph pipeline.
    
    Parameters
    ----------
    config : dict or str, optional
        Configuration dictionary or path to configuration file, by default None
        
    Returns
    -------
    GraphPipeline
        Graph pipeline
    """
    if config is None:
        return GraphPipeline(GraphPipeline.create_default_config())
    else:
        return GraphPipeline(config)
    

