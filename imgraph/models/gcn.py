import torch
import torch.nn as nn
from torch.nn import Sequential, Linear, ReLU, BatchNorm1d
from torch_geometric.nn import Sequential  as Seq, GCNConv
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool
import torch.nn.functional as F
from typing import Optional



class GCN(torch.nn.Module):
    """
    GCN model
    """
    def __init__(self, input_channels,hidden_channels, n_layers):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(input_channels, hidden_channels))
        for i in range(n_layers-1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.lin1 = Linear(hidden_channels, 32)
        self.lin = Linear(32, 2)

    def forward(self, x, edge_index, batch):
        #  Obtain node embeddings
        for conv in self.convs:
            x = conv(x, edge_index)
            x = x.relu()

        #  Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        #  Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin1(x)
        x = x.relu()
        x = self.lin(x)
        
        # Final graph representation
        return x
    
def gcn_model(input_channels : int,hidden_channels : int, n_layers  : Optional[int] = 4, task = 'classification'):
    """
    Args:
        input_channels (int): The number of input channels.
        hidden_channels (int): The number of hidden channels.
        n_layers (int, optional): The number of GCN layers.
        task (str, optional): The task to be solved. (default: :obj:`"classification"`)
    Returns:
        The GCN model.
    """

    if task == 'classification':
        return GCN(input_channels,hidden_channels, n_layers)
    else:
        raise NotImplementedError("Only classification task is supported")
