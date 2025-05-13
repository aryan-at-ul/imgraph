# # """
# # Simple test script to verify that imports are working correctly.
# # """

# # def test_imports():
# #     """Test that all important modules can be imported successfully."""
# #     try:
# #         # Core components
# #         from imgraph import GraphBuilder, GraphPresets, GraphPipeline
# #         print("✓ Core components imported successfully")
        
# #         # Data components
# #         from imgraph.data import image_transform_slic, make_edges, graph_generator
# #         print("✓ Legacy data components imported successfully")
        
# #         # Node creation methods
# #         from imgraph.data.node_creation import slic_superpixel_nodes, regular_patch_nodes
# #         print("✓ Node creation methods imported successfully")
        
# #         # Node feature methods
# #         from imgraph.data.node_features import mean_color_features, lbp_features
# #         print("✓ Node feature methods imported successfully")
        
# #         # Edge creation methods
# #         from imgraph.data.edge_creation import region_adjacency_edges, grid_4_edges
# #         print("✓ Edge creation methods imported successfully")
        
# #         # Edge feature methods
# #         from imgraph.data.edge_features import color_difference, boundary_strength
# #         print("✓ Edge feature methods imported successfully")
        
# #         # Models
# #         from imgraph.models import GCN, GAT, GIN, GraphSAGE
# #         print("✓ Models imported successfully")
        
# #         # Training utilities
# #         from imgraph.training import Trainer, EarlyStopping
# #         print("✓ Training utilities imported successfully")
        
# #         # Datasets
# #         from imgraph.datasets import ImageFolderGraphDataset, MNISTGraphDataset, get_mnist_dataset  # Corrected
# #         print("✓ Datasets imported successfully")
        
# #         # Visualization
# #         from imgraph.visualization import visualize_graph, visualize_graph_with_features
# #         print("✓ Visualization tools imported successfully")
        
# #         print("\nAll imports successful! Your package should now work correctly.")
# #         return True
    
# #     except ImportError as e:
# #         print(f"❌ Import error: {e}")
# #         return False

# # if __name__ == "__main__":
# #     test_imports()



# # test_imports_debug.py (create a new file)
# import inspect
# import sys

# def test_imports():
#     """Test that all important modules can be imported successfully."""
#     try:
#         import imgraph.datasets
#         print(f"Contents of imgraph.datasets: {dir(imgraph.datasets)}")
#         print(f"File path: {imgraph.datasets.__file__}")
        
#         # Check what get_mnist_dataset should be
#         if hasattr(imgraph.datasets, 'get_mnist_dataset'):
#             print("'get_mnist_dataset' exists?!")
#         elif hasattr(imgraph.datasets, 'get_mnist_dataset'):
#             print("'get_mnist_dataset' exists but we're asking for 'get_mnist_dataset'")
#         else:
#             print("Neither function exists in the module")
            
#         # Try to import manually
#         from imgraph.datasets import get_mnist_dataset
#         print("Import succeeded unexpectedly")
#     except ImportError as e:
#         print(f"❌ Import error: {e}")
        
#         # Print source code lines to see what's actually being imported
#         import traceback
#         tb = traceback.extract_tb(sys.exc_info()[2])
#         filename, line, func, text = tb[-1]
#         print(f"Error at: {filename}:{line}")
#         print(f"Code: {text}")
#         return False

# if __name__ == "__main__":
#     test_imports()


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