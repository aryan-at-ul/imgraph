import torch
import torch.nn as nn
from torch.nn import Sequential, Linear, ReLU, BatchNorm1d
from torch_geometric.nn import GINConv
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool
import torch.nn.functional as F
from typing import Optional




class GIN(torch.nn.Module):
    """GIN"""
    def __init__(self, input_dim, dim_h=64, n_layers=3):
        torch.manual_seed(12345)
        super(GIN, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(GINConv(Sequential(Linear(input_dim, dim_h),
                                             BatchNorm1d(dim_h), ReLU(),
                                             Linear(dim_h, dim_h), ReLU())))
        for i in range(n_layers-1):
            self.convs.append(GINConv(Sequential(Linear(dim_h, dim_h), BatchNorm1d(dim_h), ReLU(),
                                                  Linear(dim_h, dim_h), ReLU())))
        self.lin1 = Linear(dim_h*n_layers, 64)
        self.lin2 = Linear(64, 2)

    def forward(self, x, edge_index, batch):
        # Node embeddings
        h = x
        for conv in self.convs:
            h = conv(h, edge_index)

        # Graph-level readout
        h_list = []
        for i in range(len(self.convs)):
            h_i = global_add_pool(h, batch)
            h_list.append(h_i)
        h = torch.cat(h_list, dim=1)

        # Classifier
        h = self.lin1(h)
        h = h.relu()
        h = F.dropout(h, p=0.5, training=self.training)
        h = self.lin2(h)

        return F.log_softmax(h, dim=1)


def gin_model(input_dim : int, dim_h : int, n_layers  : Optional[int] = 4, task = 'classification'):
    """
    Args:
        input_dim (int): The number of input channels.
        dim_h (int): The number of hidden channels.
        n_layers (int, optional): The number of GIN layers.
        task (str, optional): The task to be solved. (default: :obj:`"classification"`)
    Returns:
        The GIN model.
    """
    if task == 'classification':
        return GIN(input_dim, dim_h, n_layers)
    else:
        raise NotImplementedError("Only classification task is supported")