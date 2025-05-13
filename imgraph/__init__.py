"""
imgraph: A library for converting images to graph representations for graph neural networks
"""

from imgraph.data.make_graph import GraphBuilder, MultiGraphBuilder
from imgraph.data.presets import GraphPresets
from imgraph.pipeline.config_pipeline import GraphPipeline

__version__ = '0.0.4'

# Expose the main classes/functions at the package level
__all__ = [
    'GraphBuilder',
    'MultiGraphBuilder',
    'GraphPresets',
    'GraphPipeline'
]