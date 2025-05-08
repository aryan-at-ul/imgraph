# import imgraph.reader
# import imgraph.writer
# # import imgraph.models
# import imgraph.data
# # import imgraph.utils
# import imgraph.pipeline



# # from .seed import seed_everything
# from .home import get_home_dir, set_home_dir

# __version__ = '0.0.7'

# __all__ = [
#     # 'seed_everything',
#     'get_home_dir',
#     'set_home_dir',
#     'imgraph',
#     '__version__',
# ]


"""
imgraph: A Modular Framework for Image-to-Graph Conversion and GNN Benchmarking

This package provides tools for:
1. Converting images to graphs using various methods
2. Training and evaluating Graph Neural Networks
3. Benchmarking different graph creation methods and GNN architectures
4. Visualizing graphs and results
"""

__version__ = "0.1.0"

import os

# Set home directory for cache
if "IMGRAPH_HOME" in os.environ:
    IMGRAPH_HOME = os.environ["IMGRAPH_HOME"]
else:
    IMGRAPH_HOME = os.path.expanduser("~/.cache/imgraph")
    os.makedirs(IMGRAPH_HOME, exist_ok=True)

# Import main components for convenience
from imgraph.data.make_graph import GraphBuilder
from imgraph.data.presets import GraphPresets
from imgraph.pipeline.config_pipeline import GraphPipeline

# Make these classes directly available
__all__ = ['GraphBuilder', 'GraphPresets', 'GraphPipeline', 'IMGRAPH_HOME']
