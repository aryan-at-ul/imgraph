"""
Presets for common graph configurations.
"""

from imgraph.data.node_creation import (
    slic_superpixel_nodes,
    felzenszwalb_superpixel_nodes,
    regular_patch_nodes,
    pixel_nodes
)
from imgraph.data.node_features import (
    mean_color_features,
    mean_std_color_features,
    histogram_color_features,
    lbp_features,
    glcm_features,
    position_features,
    normalized_position_features,
    pretrained_cnn_features
)
from imgraph.data.edge_creation import (
    region_adjacency_edges,
    grid_4_edges,
    grid_8_edges,
    distance_threshold_edges,
    k_nearest_edges,
    feature_similarity_edges
)
from imgraph.data.edge_features import (
    feature_difference,
    color_difference,
    distance_features,
    angle_features,
    boundary_strength
)

import torch
from imgraph.data.make_graph import GraphBuilder, combine_features, combine_edge_features

class GraphPresets:
    """
    A class providing preset configurations for graph construction.
    """
    
    @staticmethod
    def slic_mean_color():
        """
        Create a graph with SLIC superpixel nodes and mean color features.
        
        Returns
        -------
        GraphBuilder
            Graph builder with the specified configuration
        """
        node_creator = lambda image: slic_superpixel_nodes(image, n_segments=100, compactness=10)
        node_featurizer = lambda image, node_info: mean_color_features(image, node_info)
        edge_creator = lambda node_info, image: region_adjacency_edges(node_info)
        
        return GraphBuilder(node_creator, node_featurizer, edge_creator)
    
    @staticmethod
    def slic_color_position():
        """
        Create a graph with SLIC superpixel nodes, mean-std color and position features.
        
        Returns
        -------
        GraphBuilder
            Graph builder with the specified configuration
        """
        node_creator = lambda image: slic_superpixel_nodes(image, n_segments=100, compactness=10)
        
        def node_featurizer(image, node_info):
            color_feat = mean_std_color_features(image, node_info)
            pos_feat = normalized_position_features(image, node_info)
            return torch.cat([color_feat, pos_feat], dim=1)
        
        edge_creator = lambda node_info, image: region_adjacency_edges(node_info)
        edge_featurizer = lambda image, node_info, edge_index: color_difference(image, node_info, edge_index)
        
        return GraphBuilder(node_creator, node_featurizer, edge_creator, edge_featurizer)
    
    @staticmethod
    def slic_texture():
        """
        Create a graph with SLIC superpixel nodes and texture features.
        
        Returns
        -------
        GraphBuilder
            Graph builder with the specified configuration
        """
        node_creator = lambda image: slic_superpixel_nodes(image, n_segments=100, compactness=10)
        node_featurizer = lambda image, node_info: lbp_features(image, node_info)
        edge_creator = lambda node_info, image: region_adjacency_edges(node_info)
        
        return GraphBuilder(node_creator, node_featurizer, edge_creator)
    
    @staticmethod
    def patches_color():
        """
        Create a graph with regular patch nodes and color features.
        
        Returns
        -------
        GraphBuilder
            Graph builder with the specified configuration
        """
        node_creator = lambda image: regular_patch_nodes(image, patch_size=16)
        node_featurizer = lambda image, node_info: mean_std_color_features(image, node_info)
        edge_creator = lambda node_info, image: grid_4_edges(node_info)
        
        return GraphBuilder(node_creator, node_featurizer, edge_creator)
    
    @staticmethod
    def patches_cnn():
        """
        Create a graph with regular patch nodes and CNN features.
        
        Returns
        -------
        GraphBuilder
            Graph builder with the specified configuration
        """
        node_creator = lambda image: regular_patch_nodes(image, patch_size=32)
        node_featurizer = lambda image, node_info: pretrained_cnn_features(image, node_info, model_name='resnet18')
        edge_creator = lambda node_info, image: grid_4_edges(node_info)
        
        return GraphBuilder(node_creator, node_featurizer, edge_creator)
    
    @staticmethod
    def superpixel_comprehensive():
        """
        Create a comprehensive graph with superpixel nodes and multiple features.
        
        Returns
        -------
        GraphBuilder
            Graph builder with the specified configuration
        """
        node_creator = lambda image: slic_superpixel_nodes(image, n_segments=150, compactness=15)
        
        def node_featurizer(image, node_info):
            feature_methods = [
                lambda img, info: mean_std_color_features(img, info),
                lambda img, info: lbp_features(img, info),
                lambda img, info: normalized_position_features(img, info)
            ]
            return combine_features(feature_methods, image, node_info)
        
        edge_creator = lambda node_info, image: region_adjacency_edges(node_info)
        
        def edge_featurizer(image, node_info, edge_index):
            feature_methods = [
                lambda img, info, edges: color_difference(img, info, edges),
                lambda img, info, edges: distance_features(info, edges),
                lambda img, info, edges: boundary_strength(img, info, edges)
            ]
            return combine_edge_features(feature_methods, image, node_info, edge_index)
        
        return GraphBuilder(node_creator, node_featurizer, edge_creator, edge_featurizer)
    
    @staticmethod
    def tiny_graph():
        """
        Create a small graph with few superpixels for quick testing.
        
        Returns
        -------
        GraphBuilder
            Graph builder with the specified configuration
        """
        node_creator = lambda image: slic_superpixel_nodes(image, n_segments=20, compactness=10)
        node_featurizer = lambda image, node_info: mean_std_color_features(image, node_info)
        edge_creator = lambda node_info, image: region_adjacency_edges(node_info)
        
        return GraphBuilder(node_creator, node_featurizer, edge_creator)
    
    @staticmethod
    def medical_preset():
        """
        Create a graph optimized for medical image analysis.
        
        Returns
        -------
        GraphBuilder
            Graph builder with the specified configuration
        """
        # More superpixels and lower compactness for medical images
        node_creator = lambda image: slic_superpixel_nodes(image, n_segments=200, compactness=5)
        
        def node_featurizer(image, node_info):
            feature_methods = [
                lambda img, info: mean_std_color_features(img, info),
                lambda img, info: histogram_color_features(img, info, bins=12),
                lambda img, info: lbp_features(img, info),
                lambda img, info: normalized_position_features(img, info)
            ]
            return combine_features(feature_methods, image, node_info)
        
        # Use both adjacency and distance for more connections
        def edge_creator(node_info, image):
            adj_edges = region_adjacency_edges(node_info)
            dist_edges = k_nearest_edges(node_info, k=3)
            
            # Combine edges
            combined_edges = torch.cat([adj_edges, dist_edges], dim=1)
            
            # Remove duplicates
            combined_edges = torch.unique(combined_edges, dim=1)
            
            return combined_edges
        
        edge_featurizer = lambda image, node_info, edge_index: distance_features(node_info, edge_index)
        
        return GraphBuilder(node_creator, node_featurizer, edge_creator, edge_featurizer)
    
    @staticmethod
    def custom_preset(config):
        """
        Create a graph with custom configuration.
        
        Parameters
        ----------
        config : dict
            Dictionary with configuration parameters
        
        Returns
        -------
        GraphBuilder
            Graph builder with the specified configuration
        """
        # Node creation
        node_creation_config = config.get('node_creation', {})
        node_method = node_creation_config.get('method', 'slic_superpixel_nodes')
        node_params = node_creation_config.get('params', {})
        
        if node_method == 'slic_superpixel_nodes':
            node_creator = lambda image: slic_superpixel_nodes(image, **node_params)
        elif node_method == 'felzenszwalb_superpixel_nodes':
            node_creator = lambda image: felzenszwalb_superpixel_nodes(image, **node_params)
        elif node_method == 'regular_patch_nodes':
            node_creator = lambda image: regular_patch_nodes(image, **node_params)
        elif node_method == 'pixel_nodes':
            node_creator = lambda image: pixel_nodes(image, **node_params)
        else:
            raise ValueError(f"Unknown node creation method: {node_method}")
        
        # Node features
        node_features_config = config.get('node_features', {})
        node_feat_method = node_features_config.get('method', 'mean_std_color_features')
        node_feat_params = node_features_config.get('params', {})
        
        if node_feat_method == 'mean_color_features':
            node_featurizer = lambda image, node_info: mean_color_features(image, node_info, **node_feat_params)
        elif node_feat_method == 'mean_std_color_features':
            node_featurizer = lambda image, node_info: mean_std_color_features(image, node_info, **node_feat_params)
        elif node_feat_method == 'histogram_color_features':
            node_featurizer = lambda image, node_info: histogram_color_features(image, node_info, **node_feat_params)
        elif node_feat_method == 'lbp_features':
            node_featurizer = lambda image, node_info: lbp_features(image, node_info, **node_feat_params)
        elif node_feat_method == 'glcm_features':
            node_featurizer = lambda image, node_info: glcm_features(image, node_info, **node_feat_params)
        elif node_feat_method == 'position_features':
            node_featurizer = lambda image, node_info: position_features(image, node_info, **node_feat_params)
        elif node_feat_method == 'normalized_position_features':
            node_featurizer = lambda image, node_info: normalized_position_features(image, node_info, **node_feat_params)
        elif node_feat_method == 'pretrained_cnn_features':
            node_featurizer = lambda image, node_info: pretrained_cnn_features(image, node_info, **node_feat_params)
        else:
            raise ValueError(f"Unknown node feature method: {node_feat_method}")
        
        # Edge creation
        edge_creation_config = config.get('edge_creation', {})
        edge_method = edge_creation_config.get('method', 'region_adjacency_edges')
        edge_params = edge_creation_config.get('params', {})
        
        if edge_method == 'region_adjacency_edges':
            edge_creator = lambda node_info, image: region_adjacency_edges(node_info, **edge_params)
        elif edge_method == 'grid_4_edges':
            edge_creator = lambda node_info, image: grid_4_edges(node_info, **edge_params)
        elif edge_method == 'grid_8_edges':
            edge_creator = lambda node_info, image: grid_8_edges(node_info, **edge_params)
        elif edge_method == 'distance_threshold_edges':
            edge_creator = lambda node_info, image: distance_threshold_edges(node_info, **edge_params)
        elif edge_method == 'k_nearest_edges':
            edge_creator = lambda node_info, image: k_nearest_edges(node_info, **edge_params)
        elif edge_method == 'feature_similarity_edges':
            # Feature similarity needs node features, so we have to adapt
            edge_creator = lambda node_info, image: feature_similarity_edges(
                node_featurizer(image, node_info), **edge_params
            )
        else:
            raise ValueError(f"Unknown edge creation method: {edge_method}")
        
        # Edge features
        edge_features_config = config.get('edge_features', {})
        edge_feat_method = edge_features_config.get('method', None)
        edge_feat_params = edge_features_config.get('params', {})
        
        if edge_feat_method is None:
            edge_featurizer = None
        elif edge_feat_method == 'color_difference':
            edge_featurizer = lambda image, node_info, edge_index: color_difference(image, node_info, edge_index, **edge_feat_params)
        elif edge_feat_method == 'distance_features':
            edge_featurizer = lambda image, node_info, edge_index: distance_features(node_info, edge_index, **edge_feat_params)
        elif edge_feat_method == 'angle_features':
            edge_featurizer = lambda image, node_info, edge_index: angle_features(node_info, edge_index, **edge_feat_params)
        elif edge_feat_method == 'boundary_strength':
            edge_featurizer = lambda image, node_info, edge_index: boundary_strength(image, node_info, edge_index, **edge_feat_params)
        else:
            raise ValueError(f"Unknown edge feature method: {edge_feat_method}")
        
        return GraphBuilder(node_creator, node_featurizer, edge_creator, edge_featurizer)
