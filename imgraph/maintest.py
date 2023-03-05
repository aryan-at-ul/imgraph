from imgraph.pipeline import create_graph_pipleline




path = "/Users/aryansingh/projects/image_segmentation/chest_xray"

create_graph_pipleline(path, 'classification', 'rag', 'resnet18', 10)