"""
Edge creation methods for connecting graph nodes.
"""

from imgraph.data.edge_creation.grid_edges import (
    grid_4_edges,
    grid_8_edges
)
from imgraph.data.edge_creation.distance_edges import (
    distance_threshold_edges,
    k_nearest_edges
)
from imgraph.data.edge_creation.region_adjacency import (
    region_adjacency_edges
)
from imgraph.data.edge_creation.similarity_edges import (
    feature_similarity_edges
)

__all__ = [
    'grid_4_edges',
    'grid_8_edges',
    'distance_threshold_edges',
    'k_nearest_edges',
    'region_adjacency_edges',
    'feature_similarity_edges'
]
