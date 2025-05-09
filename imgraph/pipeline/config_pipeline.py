"""
Pipeline for creating graphs from images using a configuration.
"""

import json
import os
from imgraph.data.presets import GraphPresets

class GraphPipeline:
    """
    A pipeline for processing images into graphs using a configuration.
    
    Parameters
    ----------
    config : dict or str
        Configuration dictionary or path to configuration file
    """
    
    def __init__(self, config):
        """Initialize the GraphPipeline."""
        if isinstance(config, str):
            # Load config from file
            with open(config, 'r') as f:
                self.config = json.load(f)
        else:
            self.config = config
        
        # Create graph builder from config
        self.graph_builder = GraphPresets.custom_preset(self.config)
    
    def process(self, image):
        """
        Process an image into a graph.
        
        Parameters
        ----------
        image : numpy.ndarray
            Input image with shape (H, W, C)
            
        Returns
        -------
        torch_geometric.data.Data
            Graph representation of the image
        """
        return self.graph_builder(image)
    
    def batch_process(self, images):
        """
        Process multiple images into graphs.
        
        Parameters
        ----------
        images : list of numpy.ndarray
            List of input images
            
        Returns
        -------
        list of torch_geometric.data.Data
            List of graph representations
        """
        return [self.process(image) for image in images]
    
    def save_config(self, path):
        """
        Save the configuration to a file.
        
        Parameters
        ----------
        path : str
            Path to save the configuration
        """
        with open(path, 'w') as f:
            json.dump(self.config, f, indent=4)
    
    @classmethod
    def from_preset(cls, preset_name, **kwargs):
        """
        Create a pipeline from a preset.
        
        Parameters
        ----------
        preset_name : str
            Name of the preset to use
        **kwargs
            Additional arguments to override preset parameters
            
        Returns
        -------
        GraphPipeline
            Pipeline with the specified preset
        """
        # Create base config based on preset
        if preset_name == 'slic_mean_color':
            config = {
                'node_creation': {'method': 'slic_superpixel_nodes', 'params': {'n_segments': 100, 'compactness': 10}},
                'node_features': {'method': 'mean_color_features', 'params': {}},
                'edge_creation': {'method': 'region_adjacency_edges', 'params': {}},
                'edge_features': {'method': None, 'params': {}}
            }
        elif preset_name == 'slic_color_position':
            config = {
                'node_creation': {'method': 'slic_superpixel_nodes', 'params': {'n_segments': 100, 'compactness': 10}},
                'node_features': {'method': 'custom', 'params': {
                    'features': [
                        {'method': 'mean_std_color_features', 'params': {}},
                        {'method': 'normalized_position_features', 'params': {}}
                    ]
                }},
                'edge_creation': {'method': 'region_adjacency_edges', 'params': {}},
                'edge_features': {'method': 'color_difference', 'params': {}}
            }
        elif preset_name == 'patches_color':
            config = {
                'node_creation': {'method': 'regular_patch_nodes', 'params': {'patch_size': 16}},
                'node_features': {'method': 'mean_std_color_features', 'params': {}},
                'edge_creation': {'method': 'grid_4_edges', 'params': {}},
                'edge_features': {'method': None, 'params': {}}
            }
        elif preset_name == 'patches_cnn':
            config = {
                'node_creation': {'method': 'regular_patch_nodes', 'params': {'patch_size': 32}},
                'node_features': {'method': 'pretrained_cnn_features', 'params': {'model_name': 'resnet18'}},
                'edge_creation': {'method': 'grid_4_edges', 'params': {}},
                'edge_features': {'method': None, 'params': {}}
            }
        elif preset_name == 'tiny_graph':
            config = {
                'node_creation': {'method': 'slic_superpixel_nodes', 'params': {'n_segments': 20, 'compactness': 10}},
                'node_features': {'method': 'mean_std_color_features', 'params': {}},
                'edge_creation': {'method': 'region_adjacency_edges', 'params': {}},
                'edge_features': {'method': None, 'params': {}}
            }
        elif preset_name == 'superpixel_comprehensive':
            config = {
                'node_creation': {'method': 'slic_superpixel_nodes', 'params': {'n_segments': 150, 'compactness': 15}},
                'node_features': {'method': 'custom', 'params': {
                    'features': [
                        {'method': 'mean_std_color_features', 'params': {}},
                        {'method': 'lbp_features', 'params': {}},
                        {'method': 'normalized_position_features', 'params': {}}
                    ]
                }},
                'edge_creation': {'method': 'region_adjacency_edges', 'params': {}},
                'edge_features': {'method': 'custom', 'params': {
                    'features': [
                        {'method': 'color_difference', 'params': {}},
                        {'method': 'distance_features', 'params': {}},
                        {'method': 'boundary_strength', 'params': {}}
                    ]
                }}
            }
        else:
            raise ValueError(f"Unknown preset: {preset_name}")
        
        # Override with kwargs
        for section in kwargs:
            if section in config:
                for param, value in kwargs[section].items():
                    config[section]['params'][param] = value
        
        return cls(config)

    @staticmethod
    def list_presets():
        """
        List available presets.
        
        Returns
        -------
        list
            List of preset names
        """
        return [
            'slic_mean_color',
            'slic_color_position',
            'patches_color',
            'patches_cnn',
            'tiny_graph',
            'superpixel_comprehensive'
        ]
    
    @staticmethod
    def create_default_config():
        """
        Create a default configuration.
        
        Returns
        -------
        dict
            Default configuration
        """
        return {
            'node_creation': {
                'method': 'slic_superpixel_nodes',
                'params': {
                    'n_segments': 100,
                    'compactness': 10,
                    'sigma': 0
                }
            },
            'node_features': {
                'method': 'mean_std_color_features',
                'params': {
                    'color_space': 'rgb',
                    'normalize': True
                }
            },
            'edge_creation': {
                'method': 'region_adjacency_edges',
                'params': {
                    'connectivity': 2
                }
            },
            'edge_features': {
                'method': 'color_difference',
                'params': {
                    'color_space': 'rgb',
                    'normalize': True
                }
            }
        }
    
    def optimize_config(self, metric_fn, param_grid, images, labels=None):
        """
        Optimize configuration parameters using grid search.
        
        Parameters
        ----------
        metric_fn : callable
            Function to evaluate graph quality
        param_grid : dict
            Dictionary of parameters to search
        images : list
            List of images to use for optimization
        labels : list, optional
            List of labels if needed for evaluation, by default None
            
        Returns
        -------
        dict
            Optimized configuration
        """
        best_score = float('-inf')
        best_config = self.config.copy()
        
        # Implement a simple grid search
        import itertools
        
        # Generate parameter combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        param_combinations = list(itertools.product(*param_values))
        
        for combination in param_combinations:
            # Update configuration
            config = self.config.copy()
            
            for i, param_name in enumerate(param_names):
                # Parse parameter path (e.g., "node_creation.params.n_segments")
                parts = param_name.split('.')
                target = config
                
                for part in parts[:-1]:
                    if part not in target:
                        target[part] = {}
                    target = target[part]
                
                # Set parameter value
                target[parts[-1]] = combination[i]
            
            # Create graph builder with updated config
            pipeline = GraphPipeline(config)
            
            # Process images
            graphs = pipeline.batch_process(images)
            
            # Evaluate
            score = metric_fn(graphs, labels)
            
            # Update best config if score is better
            if score > best_score:
                best_score = score
                best_config = config
        
        # Update current config
        self.config = best_config
        self.graph_builder = GraphPresets.custom_preset(self.config)
        
        return best_config