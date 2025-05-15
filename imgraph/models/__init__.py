"""
Model implementations for graph neural networks.
"""

from imgraph.models.base import BaseGNN
from imgraph.models.gcn import GCN, GCNWithEdgeFeatures
from imgraph.models.gat import GAT, GATWithEdgeFeatures
from imgraph.models.gin import GIN, GINWithEdgeFeatures
from imgraph.models.sage import GraphSAGE, GraphSAGEWithEdgeFeatures
from imgraph.models.custom_gnn import CustomGNN, CustomGNNWithEdgeFeatures

__all__ = [
    'BaseGNN',
    'GCN',
    'GCNWithEdgeFeatures',
    'GAT',
    'GATWithEdgeFeatures',
    'GIN',
    'GINWithEdgeFeatures',
    'GraphSAGE',
    'GraphSAGEWithEdgeFeatures',
    'CustomGNN',
    'CustomGNNWithEdgeFeatures'
]