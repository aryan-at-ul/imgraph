import torch
import torch.nn as nn
from torch.nn import Sequential, Linear, ReLU, BatchNorm1d
from torch_geometric.nn import GATConv
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool
import torch.nn.functional as F
from typing import Optional



class GAT(torch.nn.Module):
    """GAT"""
    def __init__(self, hidden_channels, n_layers):
        super(GAT, self).__init__()
        torch.manual_seed(12345)
        self.hid = 8
        self.in_head = 8
        self.out_head = 1
        
        self.convs = torch.nn.ModuleList()
        self.convs.append(GATConv(hidden_channels, self.hid, heads=self.in_head, dropout=0.3))
        for i in range(n_layers - 1):
            self.convs.append(GATConv(self.hid * self.in_head, self.hid, heads=self.in_head, dropout=0.3))
        self.conv_final = GATConv(self.hid * self.in_head, 32, concat=False,
                                  heads=self.out_head, dropout=0.3)
        self.lin1 = Linear(32, 2)

    def forward(self, x, edge_index, batch):
        # Dropout before the GAT layer is used to avoid overfitting
        x = F.dropout(x, p=0.3, training=self.training)
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=0.3, training=self.training)
        x = self.conv_final(x, edge_index)
        x = global_mean_pool(x, batch)
        x = self.lin1(x)

        return x


def gat_model(hidden_channels : int, n_layers  : Optional[int] = 4, task = 'classification'):
    """
    Args:
        hidden_channels (int): The number of hidden channels.
        n_layers (int, optional): The number of GAT layers.
        task (str, optional): The task to be solved. (default: :obj:`"classification"`)
    Returns:
        The GAT model.
    """
    if task == 'classification':
        return GAT(hidden_channels, n_layers)
    else:
        raise NotImplementedError("Only classification task is supported")