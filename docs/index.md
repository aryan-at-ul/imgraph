# imgraph: Graph Neural Networks for Image Processing

[![Python Versions](https://img.shields.io/badge/python-3.7%20%7C%203.8%20%7C%203.9%20%7C%203.10-blue)](https://pypi.org/project/imgraph/)
[![PyPI Version](https://img.shields.io/badge/pypi-v0.0.9-blue)](https://pypi.org/project/imgraph/)
[![License](https://img.shields.io/badge/license-MIT-green)](https://opensource.org/licenses/MIT)

`imgraph` is a Python library for converting images to graph representations and applying Graph Neural Networks (GNNs) to image analysis tasks. Built on top of PyTorch and PyTorch Geometric, it provides an easy-to-use interface for a variety of image-to-graph conversion methods and GNN architectures. The library supports multiple methods for node creation, feature extraction, and edge construction to enable the most effective graph representation for your specific computer vision task.

## 🔑 Key Features

- **Diverse Graph Representations**: Convert images to graphs using multiple methods:
  - SLIC superpixels
  - Felzenszwalb superpixels
  - Regular grid patches
  - Pixel-level nodes
- **Comprehensive Feature Extraction**: Extract rich node and edge features:
  - Color features (mean, std, histograms)
  - Texture features (LBP, GLCM, Gabor)
  - Position features
  - CNN-based features
  - Boundary and geometric features
- **Flexible Edge Construction**: Connect nodes using various strategies:
  - Region adjacency
  - Distance-based connections
  - Grid-based connections (4-connected, 8-connected)
  - Feature similarity
- **Pre-built GNN Models**: Includes implementations of GCN, GAT, GIN, and GraphSAGE
- **Easy Visualization**: Visualize graph representations with intuitive plotting functions
- **Ready-to-Use Presets**: Common graph creation configurations available as presets
- **Training Pipeline**: Complete training and evaluation pipeline for graph-based image classification

## 🔧 Installation

```bash
# Basic installation
pip install imgraph

# Full installation with all dependencies
pip install imgraph[full]  # testing and under dev

# Developer installation
pip install imgraph[dev]  
```

## 📚 Quick Start

### Basic Graph Creation

```python
import cv2
import matplotlib.pyplot as plt
from imgraph import GraphPresets

# Load an image
image = cv2.imread('sample_image.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Create a graph using a preset
graph_builder = GraphPresets.slic_mean_color()
graph = graph_builder(image)

# Visualize the graph
fig = graph_builder.visualize_graph(image, graph)
plt.show()

# Access graph properties
print(f"Number of nodes: {graph.num_nodes}")
print(f"Number of edges: {graph.edge_index.shape[1]}")
print(f"Node feature dimensions: {graph.x.shape[1]}")
```

### Creating Custom Graphs

```python
from imgraph import GraphBuilder
from imgraph.data.node_creation import slic_superpixel_nodes
from imgraph.data.node_features import mean_std_color_features
from imgraph.data.edge_creation import region_adjacency_edges
from imgraph.data.edge_features import color_difference

# Create a custom graph builder
graph_builder = GraphBuilder(
    node_creation_method=lambda img: slic_superpixel_nodes(img, n_segments=100, compactness=10),
    node_feature_method=lambda img, nodes: mean_std_color_features(img, nodes, color_space='hsv'),
    edge_creation_method=lambda nodes, img: region_adjacency_edges(nodes, connectivity=2),
    edge_feature_method=lambda img, nodes, edges: color_difference(img, nodes, edges)
)

# Process an image
graph = graph_builder(image)
```

### Using the Pipeline API

```python
from imgraph import GraphPipeline

# Define a custom configuration
config = {
    'node_creation': {
        'method': 'slic_superpixel_nodes',
        'params': {
            'n_segments': 100,
            'compactness': 10,
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

# Create the custom graph
pipeline = GraphPipeline(config)
graph = pipeline.process(image)
```

### Training a GNN Model

```python
import torch
from torch_geometric.loader import DataLoader
from imgraph.datasets import ImageFolderGraphDataset
from imgraph.models import GCN
from imgraph.training import Trainer, EarlyStopping

# Create dataset from a folder of images
dataset = ImageFolderGraphDataset(
    root='dataset_directory',
    preset='slic_mean_color',
    force_reload=False
)

# Split dataset
train_dataset, test_dataset = dataset.get_train_test_split(train_ratio=0.8)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=4)

# Create model
num_features = train_dataset[0].x.shape[1]
num_classes = len(dataset.classes)
model = GCN(num_features, 64, num_classes, num_layers=3)

# Setup training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()
early_stopping = EarlyStopping(patience=10)

# Train the model
trainer = Trainer(model, optimizer, criterion, device, early_stopping)
history = trainer.fit(train_loader, test_loader, epochs=50)

# Evaluate
accuracy = trainer.evaluate(test_loader)
print(f"Test accuracy: {accuracy:.4f}")
```

## 🧩 Available Methods

### Node Creation Methods
```python
from imgraph.data.node_creation import (
    slic_superpixel_nodes,           # SLIC superpixel segmentation
    felzenszwalb_superpixel_nodes,   # Felzenszwalb's algorithm
    regular_patch_nodes,             # Regular grid patches
    pixel_nodes                      # Individual pixels as nodes
)
```

### Node Feature Extraction Methods
```python
from imgraph.data.node_features import (
    # Color features
    mean_color_features,             # Mean color in various color spaces
    mean_std_color_features,         # Mean and standard deviation of colors
    histogram_color_features,        # Color histograms
    
    # Texture features
    lbp_features,                    # Local Binary Patterns
    glcm_features,                   # Gray-Level Co-occurrence Matrix
    gabor_features,                  # Gabor filter responses
    
    # Position features
    position_features,               # Raw position coordinates
    normalized_position_features,    # Normalized position coordinates
    
    # Deep features
    pretrained_cnn_features          # Features from pre-trained CNNs
)
```

### Edge Creation Methods
```python
from imgraph.data.edge_creation import (
    # Grid-based connections
    grid_4_edges,                    # 4-connected grid
    grid_8_edges,                    # 8-connected grid
    
    # Distance-based connections
    distance_threshold_edges,        # Edges based on distance threshold
    k_nearest_edges,                 # K-nearest neighbor connections
    
    # Region-based connections
    region_adjacency_edges,          # Connect adjacent regions
    
    # Feature-based connections
    feature_similarity_edges         # Connect nodes with similar features
)
```

### Edge Feature Extraction Methods
```python
from imgraph.data.edge_features import (
    # Feature differences
    feature_difference,              # Difference between node features
    color_difference,                # Color differences between nodes
    
    # Geometric features
    distance_features,               # Distance between nodes
    angle_features,                  # Angle between nodes
    
    # Boundary features
    boundary_strength,               # Strength of boundaries between regions
    boundary_orientation             # Orientation of boundaries
)
```

## 🧩 Available Presets

The library includes several presets for common graph creation configurations:

- `slic_mean_color()`: SLIC superpixels with mean color features
- `slic_color_position()`: SLIC superpixels with color and position features
- `patches_color()`: Regular grid patches with color features
- `tiny_graph()`: Small-scale graph with minimal nodes
- `superpixel_comprehensive()`: Detailed superpixel representation with multiple feature types

## 📐 Architecture

The package is organized into several modules:

- `data`: Graph creation and feature extraction
  - `node_creation`: Methods for creating nodes from images
  - `node_features`: Methods for extracting node features
  - `edge_creation`: Methods for creating edges between nodes
  - `edge_features`: Methods for extracting edge features
- `models`: GNN model implementations (GCN, GAT, GIN, GraphSAGE)
- `datasets`: Dataset classes for common benchmarks and custom data
- `training`: Training utilities for model training and evaluation
- `visualization`: Tools for visualizing graphs and features
- `pipeline`: End-to-end processing pipelines

The core functionality is built around the `GraphBuilder` class, which provides a modular way to construct graphs from images by combining node creation, feature extraction, and edge construction methods.

## 🚀 Advanced Usage

### Combining Multiple Feature Types

```python
from imgraph import GraphBuilder
from imgraph.data.node_creation import slic_superpixel_nodes
from imgraph.data.node_features import mean_color_features, normalized_position_features
from imgraph.data.edge_creation import region_adjacency_edges
from imgraph.data.make_graph import combine_features

# Combine multiple feature extraction methods
def combined_node_features(image, node_info):
    return combine_features(
        [
            lambda img, nodes: mean_color_features(img, nodes, color_space='rgb'),
            lambda img, nodes: normalized_position_features(img, nodes)
        ],
        image, 
        node_info
    )

# Create a graph builder with combined features
graph_builder = GraphBuilder(
    node_creation_method=slic_superpixel_nodes,
    node_feature_method=combined_node_features,
    edge_creation_method=region_adjacency_edges
)

# Process an image
graph = graph_builder(image)
```

### Creating Multiple Graph Representations

```python
from imgraph import MultiGraphBuilder, GraphPresets

# Create multiple graph builders with different configurations
multi_builder = MultiGraphBuilder([
    GraphPresets.slic_mean_color(),
    GraphPresets.patches_color(),
    GraphPresets.superpixel_comprehensive()
])

# Process an image to get multiple graph representations
graphs = multi_builder(image)

# Access individual graphs
slic_graph = graphs[0]
patches_graph = graphs[1]
comprehensive_graph = graphs[2]
```

## 📝 Citation

If you use `imgraph` in your research, please cite:

```
@software{imgraph2023,
  author = {Singh, Aryan},
  title = {imgraph: Graph Neural Networks for Image Processing},
  url = {https://github.com/aryan-at-ul/imgraph},
  version = {0.0.9},
  year = {2023},
}
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.