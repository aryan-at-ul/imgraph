"""
Pipeline module for creating and processing graphs.
"""

from imgraph.pipeline.config_pipeline import GraphPipeline
from imgraph.pipeline.folder_pipeline import FolderGraphPipeline
from imgraph.pipeline.from_dataset import load_saved_datasets, process_dataset, create_graph_pipeline

# Define the main imports for the module
__all__ = [
    'GraphPipeline',
    'FolderGraphPipeline',
    'load_saved_datasets',
    'process_dataset',
    'create_graph_pipeline'
]