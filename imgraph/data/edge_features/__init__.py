"""
Edge feature extraction methods for image graph edges.
"""

from imgraph.data.edge_features.feature_diff import (
    feature_difference,
    color_difference
)
from imgraph.data.edge_features.geometric_features import (
    distance_features,
    angle_features
)
from imgraph.data.edge_features.boundary_features import (
    boundary_strength,
    boundary_orientation
)

__all__ = [
    'feature_difference',
    'color_difference',
    'distance_features',
    'angle_features',
    'boundary_strength',
    'boundary_orientation'
]
