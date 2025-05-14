import os

modules = [
    "imgraph.data.edge_creation.grid_edges",
    "imgraph.data.edge_creation.distance_edges",
    "imgraph.data.edge_creation.region_adjacency",
    "imgraph.data.edge_creation.similarity_edges",
    "imgraph.data.edge_features.boundary_features",
    "imgraph.data.edge_features.feature_diff",
    "imgraph.data.edge_features.geometric_features",
    "imgraph.data.node_features.cnn_features",
    "imgraph.data.node_features.color_features",
    "imgraph.data.node_features.position_features",
    "imgraph.data.node_features.texture_features",
    "imgraph.data.node_creation.patch_nodes",
    "imgraph.data.node_creation.pixel_nodes",
    "imgraph.data.node_creation.superpixel_nodes",
    "imgraph.data.feature_extractor",
    "imgraph.data.make_graph",
    "imgraph.data.legacy",
    "imgraph.data.presets",
    "imgraph.data.transform_graph",
    "imgraph.datasets.mnist_dataset",
    "imgraph.datasets.medmnist_dataset",
    "imgraph.datasets.pneumonia_dataset",
    "imgraph.datasets.image_folder",
    "imgraph.datasets.standard",
    "imgraph.datasets.url_config",
    "imgraph.models.gcn",
    "imgraph.models.gat",
    "imgraph.models.gin",
    "imgraph.models.sage",
    "imgraph.models.base",
    "imgraph.pipeline.folder_pipeline",
    "imgraph.pipeline.from_dataset",
    "imgraph.pipeline.config_pipeline",
    "imgraph.reader.read_directory",
    "imgraph.reader.read_files",
    "imgraph.writer.makedirs",
    "imgraph.writer.write_files",
    "imgraph.training.trainer",
    "imgraph.training.utils",
    "imgraph.utils.feature_size",
    "imgraph.visualization.graph_plots",
    "imgraph.examples.basic_graph_creation",
    "imgraph.examples.comprehensive_graph_creation",
    "imgraph.examples.training_image_folder_example",
    "imgraph.examples.training_synthetic_example",
]


structure = {
    "index.md": "# ImGraph\n\nWelcome to the **ImGraph** documentation.\n\n> A Python library for converting images to graphs and applying Graph Neural Networks (GNNs)."
}

for module in modules:
    file_path = f"{module.replace('.', '/')}.md"
    content = f"## `{module}`\n\n::: {module}\n"
    structure[file_path] = content


for path, content in structure.items():
    full_path = os.path.join("docs", path)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    with open(full_path, "w") as f:
        f.write(content)

print("✅ All .md files written using correct Python module paths (dot notation) for mkdocstrings.")
