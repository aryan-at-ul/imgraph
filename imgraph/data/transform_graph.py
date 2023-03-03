import networkx as nx
import torch
import os
import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx




def load_and_transform(gobject,name,task = 'classification',type = True):
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
    except:
        #logging error here.
        pass
    yy = [0]
    if type:
        data.y = [1]
        yy = [1]
    else:
        data.y = [0]
    k+= 1
    data.x = torch.Tensor([torch.flatten(val).tolist() for val in data.x])
    data.name = name
    return data


