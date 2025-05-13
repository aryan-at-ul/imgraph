# """
# Data module for image graph processing.
# """

# from imgraph.data.make_graph import GraphBuilder, MultiGraphBuilder, combine_features, combine_edge_features

# __all__ = [
#     'GraphBuilder',
#     'MultiGraphBuilder',
#     'combine_features',
#     'combine_edge_features'
# ]

"""
Data module for image graph processing.
"""

from imgraph.data.make_graph import GraphBuilder, MultiGraphBuilder, combine_features, combine_edge_features
from imgraph.data.legacy import image_transform_slic, make_edges, graph_generator

__all__ = [
    'GraphBuilder',
    'MultiGraphBuilder',
    'combine_features',
    'combine_edge_features',
    # Legacy functions
    'image_transform_slic',
    'make_edges',
    'graph_generator'
]