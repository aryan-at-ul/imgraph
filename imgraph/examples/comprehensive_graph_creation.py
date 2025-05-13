"""
Comprehensive example for creating and analyzing graphs from images with custom configurations.
"""

import os
import cv2
import numpy as np
import torch

# Set non-interactive backend to avoid GTK errors
import matplotlib
matplotlib.use('Agg')  # Use the non-interactive backend
import matplotlib.pyplot as plt

from imgraph import GraphPresets, GraphPipeline
from imgraph.data.node_creation import slic_superpixel_nodes, regular_patch_nodes
from imgraph.data.node_features import mean_std_color_features, lbp_features, normalized_position_features
from imgraph.data.edge_creation import region_adjacency_edges, k_nearest_edges
from imgraph.data.edge_features import color_difference, boundary_strength
from imgraph.visualization.graph_plots import visualize_graph_with_features, plot_node_feature_distribution

def create_custom_graph(image):
    """
    Creates a custom graph with specific configuration.
    
    Parameters
    ----------
    image : numpy.ndarray
        Input image
        
    Returns
    -------
    torch_geometric.data.Data
        Graph data
    """
    # Create custom configuration
    config = {
        'node_creation': {
            'method': 'slic_superpixel_nodes',
            'params': {
                'n_segments': 75,
                'compactness': 15,
                'sigma': 1.0
            }
        },
        'node_features': {
            'method': 'mean_std_color_features',
            'params': {
                'color_space': 'hsv',
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
                'color_space': 'hsv',
                'normalize': True
            }
        }
    }
    
    # Create graph pipeline
    pipeline = GraphPipeline(config)
    
    # Process image
    return pipeline.process(image)

def create_multi_feature_graph(image):
    """
    Creates a graph with multiple node and edge features.
    
    Parameters
    ----------
    image : numpy.ndarray
        Input image
        
    Returns
    -------
    torch_geometric.data.Data
        Graph data
    """
    # Create nodes
    node_info = slic_superpixel_nodes(image, n_segments=50, compactness=10)
    
    # Extract multiple node features
    color_features = mean_std_color_features(image, node_info)
    texture_features = lbp_features(image, node_info)
    position_features = normalized_position_features(image, node_info)
    
    # Combine node features
    node_features = torch.cat([color_features, texture_features, position_features], dim=1)
    
    # Create edges
    edge_index = region_adjacency_edges(node_info)
    
    # Extract edge features
    edge_feat1 = color_difference(image, node_info, edge_index)
    edge_feat2 = boundary_strength(image, node_info, edge_index)
    
    # Combine edge features
    edge_attr = torch.cat([edge_feat1, edge_feat2], dim=1)
    
    # Create graph data
    from torch_geometric.data import Data
    
    graph_data = Data(
        x=node_features,
        edge_index=edge_index,
        edge_attr=edge_attr,
        num_nodes=len(node_features),
        image_size=torch.tensor(image.shape[:2])
    )
    
    # Store node info for visualization
    graph_data.node_info = node_info
    
    return graph_data

def analyze_graphs(image, graphs, names):
    """
    Analyzes and visualizes graphs.
    
    Parameters
    ----------
    image : numpy.ndarray
        Input image
    graphs : list
        List of graph data
    names : list
        List of graph names
    """
    os.makedirs("outputs/analysis", exist_ok=True)
    
    for i, (graph, name) in enumerate(zip(graphs, names)):
        print(f"Analyzing {name} graph...")
        
        try:
            # Visualize graph with node feature heatmap
            if graph.x is not None and graph.x.shape[1] > 0:
                for feature_idx in range(min(3, graph.x.shape[1])):
                    fig = visualize_graph_with_features(image, graph, node_feature_idx=feature_idx)
                    fig.suptitle(f"{name} - Node Feature {feature_idx}")
                    fig.savefig(f"outputs/analysis/{name}_node_feature_{feature_idx}.png")
                    plt.close(fig)  # Close figure to free memory
            
            # Visualize graph with edge feature heatmap
            if hasattr(graph, 'edge_attr') and graph.edge_attr is not None and graph.edge_attr.shape[1] > 0:
                for feature_idx in range(min(2, graph.edge_attr.shape[1])):
                    fig = visualize_graph_with_features(image, graph, edge_feature_idx=feature_idx)
                    fig.suptitle(f"{name} - Edge Feature {feature_idx}")
                    fig.savefig(f"outputs/analysis/{name}_edge_feature_{feature_idx}.png")
                    plt.close(fig)  # Close figure to free memory
            
            # Plot node feature distributions
            if graph.x is not None and graph.x.shape[1] > 0:
                fig = plot_node_feature_distribution(graph)
                fig.suptitle(f"{name} - Node Feature Distribution")
                fig.savefig(f"outputs/analysis/{name}_node_feature_distribution.png")
                plt.close(fig)  # Close figure to free memory
        except Exception as e:
            print(f"  Warning: Error visualizing {name} graph: {e}")
        
        # Print graph statistics
        print(f"  Nodes: {graph.num_nodes}")
        print(f"  Edges: {graph.edge_index.shape[1]}")
        print(f"  Node feature dimensions: {graph.x.shape[1]}")
        if hasattr(graph, 'edge_attr') and graph.edge_attr is not None:
            print(f"  Edge feature dimensions: {graph.edge_attr.shape[1]}")
        print()

def create_test_image(width=300, height=300):
    """Create a synthetic test image with various shapes."""
    image = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    # Draw some shapes
    cv2.rectangle(image, (50, 50), (100, 100), (255, 0, 0), -1)
    cv2.rectangle(image, (150, 50), (200, 100), (0, 255, 0), -1)
    cv2.rectangle(image, (100, 150), (200, 200), (0, 0, 255), -1)
    cv2.circle(image, (75, 200), 25, (255, 255, 0), -1)
    cv2.circle(image, (200, 75), 25, (255, 0, 255), -1)
    
    return image

def main():
    try:
        # Create output directories
        os.makedirs("outputs", exist_ok=True)
        os.makedirs("outputs/analysis", exist_ok=True)
        
        # Load sample image
        try:
            # Try to load a sample image
            image_path = "sample_image.jpg"
            if os.path.exists(image_path):
                image = cv2.imread(image_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                print(f"Using image from {image_path}")
            else:
                # If no file exists, create a synthetic test image
                print("No sample image found. Creating a synthetic test image...")
                image = create_test_image()
        except Exception as e:
            print(f"Error loading image: {e}. Creating a synthetic test image...")
            image = create_test_image()
        
        # Save original image
        try:
            plt.figure(figsize=(8, 8))
            plt.imshow(image)
            plt.title("Original Image")
            plt.axis("off")
            plt.savefig("outputs/original_image.png")
            plt.close()  # Close the figure to free memory
        except Exception as e:
            print(f"Warning: Unable to display original image: {e}")
            # Save image directly using OpenCV as a fallback
            cv2.imwrite("outputs/original_image.png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        
        # Create graphs with different methods
        graphs = []
        names = []
        
        print("Creating graphs with different methods...")
        
        # 1. Basic SLIC graph
        try:
            graph_builder = GraphPresets.slic_mean_color()
            slic_graph = graph_builder(image)
            graphs.append(slic_graph)
            names.append("slic_basic")
            print("  ✓ Created SLIC basic graph")
        except Exception as e:
            print(f"  ✗ Error creating SLIC basic graph: {e}")
        
        # 2. Patch-based graph
        try:
            graph_builder = GraphPresets.patches_color()
            patch_graph = graph_builder(image)
            graphs.append(patch_graph)
            names.append("patches")
            print("  ✓ Created patches graph")
        except Exception as e:
            print(f"  ✗ Error creating patches graph: {e}")
        
        # 3. Custom graph
        try:
            custom_graph = create_custom_graph(image)
            graphs.append(custom_graph)
            names.append("custom")
            print("  ✓ Created custom graph")
        except Exception as e:
            print(f"  ✗ Error creating custom graph: {e}")
        
        # 4. Multi-feature graph
        try:
            multi_feature_graph = create_multi_feature_graph(image)
            graphs.append(multi_feature_graph)
            names.append("multi_feature")
            print("  ✓ Created multi-feature graph")
        except Exception as e:
            print(f"  ✗ Error creating multi-feature graph: {e}")
        
        # 5. Graph with specific feature combination
        try:
            graph_builder = GraphPresets.superpixel_comprehensive()
            comprehensive_graph = graph_builder(image)
            graphs.append(comprehensive_graph)
            names.append("comprehensive")
            print("  ✓ Created comprehensive graph")
        except Exception as e:
            print(f"  ✗ Error creating comprehensive graph: {e}")
        
        # Visualize all graphs
        print("\nVisualizing graphs...")
        for i, (graph, name) in enumerate(zip(graphs, names)):
            try:
                # Create a graph builder for visualization
                temp_builder = GraphPresets.slic_mean_color()
                
                # Visualize graph
                fig = temp_builder.visualize_graph(image, graph)
                fig.suptitle(f"Graph with {name} configuration")
                fig.savefig(f"outputs/graph_{name}.png")
                plt.close(fig)  # Close figure to free memory
                
                print(f"  ✓ Visualized {name} graph with {graph.num_nodes} nodes and {graph.edge_index.shape[1]} edges")
            except Exception as e:
                print(f"  ✗ Error visualizing {name} graph: {e}")
        
        # Analyze graphs
        print("\nAnalyzing graphs...")
        analyze_graphs(image, graphs, names)
        
        print("\nGraph visualizations and analysis saved to 'outputs' directory.")
    
    except Exception as e:
        print(f"Critical error in main function: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()