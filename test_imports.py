"""
Comprehensive test script to verify that all imports are working correctly.
"""

import sys

def test_imports():
    """Test that all important modules can be imported successfully."""
    results = []
    
    # Core components
    try:
        from imgraph import GraphBuilder, GraphPresets, GraphPipeline
        results.append(("Core components", True))
    except ImportError as e:
        results.append(("Core components", False, str(e)))
    
    # Data components
    try:
        from imgraph.data import image_transform_slic, make_edges, graph_generator
        results.append(("Legacy data components", True))
    except ImportError as e:
        results.append(("Legacy data components", False, str(e)))
    
    # Node creation methods
    try:
        from imgraph.data.node_creation import slic_superpixel_nodes, regular_patch_nodes
        results.append(("Node creation methods", True))
    except ImportError as e:
        results.append(("Node creation methods", False, str(e)))
    
    # Node feature methods
    try:
        from imgraph.data.node_features import mean_color_features, lbp_features
        results.append(("Node feature methods", True))
    except ImportError as e:
        results.append(("Node feature methods", False, str(e)))
    
    # Edge creation methods
    try:
        from imgraph.data.edge_creation import region_adjacency_edges, grid_4_edges
        results.append(("Edge creation methods", True))
    except ImportError as e:
        results.append(("Edge creation methods", False, str(e)))
    
    # Edge feature methods
    try:
        from imgraph.data.edge_features import color_difference, boundary_strength
        results.append(("Edge feature methods", True))
    except ImportError as e:
        results.append(("Edge feature methods", False, str(e)))
    
    # Models
    try:
        from imgraph.models import GCN, GAT, GIN, GraphSAGE
        results.append(("Models", True))
    except ImportError as e:
        results.append(("Models", False, str(e)))
    
    # Training utilities
    try:
        from imgraph.training import Trainer, EarlyStopping
        results.append(("Training utilities", True))
    except ImportError as e:
        results.append(("Training utilities", False, str(e)))
    
    # Datasets
    try:
        from imgraph.datasets import ImageFolderGraphDataset, MNISTGraphDataset
        results.append(("Dataset classes", True))
    except ImportError as e:
        results.append(("Dataset classes", False, str(e)))
    
    # Dataset functions
    try:
        from imgraph.datasets import get_mnist_dataset  # Correct spelling!
        results.append(("Dataset functions", True))
    except ImportError as e:
        results.append(("Dataset functions", False, str(e)))
    
    # Visualization
    try:
        from imgraph.visualization import visualize_graph, visualize_graph_with_features
        results.append(("Visualization tools", True))
    except ImportError as e:
        results.append(("Visualization tools", False, str(e)))
    
    # Print results
    print("\n=== Import Test Results ===\n")
    all_passed = True
    
    for result in results:
        if len(result) == 2:
            module, success = result
            if success:
                print(f"✓ {module} imported successfully")
            else:
                all_passed = False
                print(f"❌ {module} import failed")
        else:
            module, success, error = result
            all_passed = False
            print(f"❌ {module} import failed: {error}")
    
    if all_passed:
        print("\n✅ All imports successful! Your package should now work correctly.")
    else:
        print("\n❌ Some imports failed. Please fix the issues listed above.")
    
    return all_passed

if __name__ == "__main__":
    print(f"Testing imports for Python {sys.version}")
    print(f"Current directory: {sys.path[0]}")
    success = test_imports()
    sys.exit(0 if success else 1)