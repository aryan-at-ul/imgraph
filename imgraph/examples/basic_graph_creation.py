"""
Example script for basic graph creation and visualization with robust error handling.
"""

import os
import cv2
import numpy as np
import traceback

# Set non-interactive backend first to avoid display issues
import matplotlib
matplotlib.use('Agg')  # Use the non-interactive backend
import matplotlib.pyplot as plt

from imgraph import GraphPresets

def create_test_image(width=300, height=300):
    """Create a synthetic test image with various shapes."""
    image = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    # Draw some shapes
    cv2.circle(image, (width//2, height//2), min(width, height)//3, (255, 0, 0), -1)
    cv2.rectangle(image, (width//6, height//6), (width*5//6, height*5//6), (0, 255, 0), 5)
    cv2.line(image, (0, 0), (width, height), (0, 0, 255), 3)
    
    return image

def main():
    # Create output directory for visualizations
    os.makedirs("outputs", exist_ok=True)
    
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
        plt.figure(figsize=(10, 8))
        plt.imshow(image)
        plt.title("Original Image")
        plt.axis("off")
        plt.savefig("outputs/original_image.png")
        plt.close()  # Close the figure to free memory
    except Exception as e:
        print(f"Warning: Unable to display original image: {e}")
        # Save image directly using OpenCV as a fallback
        cv2.imwrite("outputs/original_image.png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    
    # Create graph builders with different presets
    presets = [
        ("slic_mean_color", GraphPresets.slic_mean_color()),
        ("slic_color_position", GraphPresets.slic_color_position()),
        ("patches_color", GraphPresets.patches_color()),
        ("tiny_graph", GraphPresets.tiny_graph()),
        ("superpixel_comprehensive", GraphPresets.superpixel_comprehensive())
    ]
    
    # Process the image with each preset and visualize the resulting graph
    for preset_name, graph_builder in presets:
        print(f"Processing with {preset_name} preset...")
        
        try:
            # Create graph
            graph = graph_builder(image)
            
            # Print graph statistics
            print(f"  Nodes: {graph.num_nodes}")
            print(f"  Edges: {graph.edge_index.shape[1]}")
            print(f"  Node feature dimensions: {graph.x.shape[1]}")
            if hasattr(graph, 'edge_attr') and graph.edge_attr is not None:
                print(f"  Edge feature dimensions: {graph.edge_attr.shape[1]}")
            
            # Try to visualize and save the graph
            try:
                fig = graph_builder.visualize_graph(image, graph)
                fig.suptitle(f"Graph with {preset_name} preset")
                fig.savefig(f"outputs/graph_{preset_name}.png")
                plt.close(fig)  # Close figure to free memory
            except Exception as e:
                print(f"  Warning: Unable to visualize graph: {e}")
        except Exception as e:
            print(f"  Error processing with {preset_name} preset: {e}")
            traceback.print_exc()
        
        print()
    
    print("Processed all presets. Results saved to 'outputs' directory.")

if __name__ == "__main__":
    main()