import networkx as nx
import torch
import os
import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
import sys


def load_and_transform(gobject,name,type ,task = 'classification'):
    """
    Args: gobject: networkx graph object
            name: name of the graph
            task: classification or regression
            type: type of the graph train/test
    Returns: torch_geometric.data.Data object
    """
    data = Data()
    try:
        data = from_networkx(gobject)
    except Exception as e:
        print("error while nx to data transformation",e,"for image",name)
        #logging error here.
        # pass
    # yy = [0]
    # if type:
    #     data.y = [1]
    #     yy = [1]
    # else:
    data.y = [type]
    data.x = torch.Tensor([torch.flatten(val).tolist() for val in data.x])
    data.name = name
    print(data)
    return data


