"""
Graph Attention Network (GAT) model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GATv2Conv
from imgraph.models.base import BaseGNN

class GAT(BaseGNN):
    """
    Graph Attention Network (GAT) model.
    
    Parameters
    ----------
    input_dim : int
        Input feature dimension
    hidden_dim : int
        Hidden layer dimension
    output_dim : int
        Output dimension
    num_layers : int, optional
        Number of layers, by default 2
    dropout : float, optional
        Dropout rate, by default 0.0
    pool : str, optional
        Global pooling method, by default 'mean'
    heads : int, optional
        Number of attention heads, by default 8
    concat : bool, optional
        Whether to concatenate attention heads, by default True
    use_v2 : bool, optional
        Whether to use GATv2, by default False
    """
    
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.0, 
                 pool='mean', heads=8, concat=True, use_v2=False):
        """Initialize the GAT model."""
        self.heads = heads
        self.concat = concat
        self.use_v2 = use_v2
        super(GAT, self).__init__(input_dim, hidden_dim, output_dim, num_layers, dropout, pool)
    
    def init_layers(self):
        """Initialize GAT layers."""
        self.convs = nn.ModuleList()
        
        # Choose GAT implementation
        GATLayer = GATv2Conv if self.use_v2 else GATConv
        
        # Input layer
        self.convs.append(GATLayer(
            in_channels=self.input_dim,
            out_channels=self.hidden_dim // self.heads if self.concat else self.hidden_dim,
            heads=self.heads,
            concat=self.concat,
            dropout=self.dropout
        ))
        
        # Hidden layers
        for i in range(self.num_layers - 1):
            if i < self.num_layers - 2:
                # Hidden layer with multiple heads
                self.convs.append(GATLayer(
                    in_channels=self.hidden_dim,
                    out_channels=self.hidden_dim // self.heads if self.concat else self.hidden_dim,
                    heads=self.heads,
                    concat=self.concat,
                    dropout=self.dropout
                ))
            else:
                # Last layer often uses a single head
                self.convs.append(GATLayer(
                    in_channels=self.hidden_dim,
                    out_channels=self.hidden_dim,
                    heads=1,
                    concat=False,
                    dropout=self.dropout
                ))
    
    def forward(self, data):
        """
        Forward pass.
        
        Parameters
        ----------
        data : torch_geometric.data.Data
            Input graph data
            
        Returns
        -------
        torch.Tensor
            Output tensor
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # If batch is None, assume a single graph
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        # Convolutional layers
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Global pooling
        x = self.pool(x, batch)
        
        # Readout
        x = self.readout(x)
        
        return x

class GATWithEdgeFeatures(BaseGNN):
    """
    Graph Attention Network (GAT) with edge features.
    
    Parameters
    ----------
    input_dim : int
        Input feature dimension
    edge_dim : int
        Edge feature dimension
    hidden_dim : int
        Hidden layer dimension
    output_dim : int
        Output dimension
    num_layers : int, optional
        Number of layers, by default 2
    dropout : float, optional
        Dropout rate, by default 0.0
    pool : str, optional
        Global pooling method, by default 'mean'
    heads : int, optional
        Number of attention heads, by default 8
    """
    
    def __init__(self, input_dim, edge_dim, hidden_dim, output_dim, num_layers=2, 
                 dropout=0.0, pool='mean', heads=8):
        """Initialize the GAT model with edge features."""
        self.edge_dim = edge_dim
        self.heads = heads
        super(GATWithEdgeFeatures, self).__init__(input_dim, hidden_dim, output_dim, num_layers, dropout, pool)
    
    def init_layers(self):
        """Initialize GAT layers with edge features."""
        self.convs = nn.ModuleList()
        
        # Input layer
        self.convs.append(GATConv(
            in_channels=self.input_dim,
            out_channels=self.hidden_dim // self.heads,
            heads=self.heads,
            concat=True,
            dropout=self.dropout,
            edge_dim=self.edge_dim
        ))
        
        # Hidden layers
        for i in range(self.num_layers - 1):
            if i < self.num_layers - 2:
                # Hidden layer with multiple heads
                self.convs.append(GATConv(
                    in_channels=self.hidden_dim,
                    out_channels=self.hidden_dim // self.heads,
                    heads=self.heads,
                    concat=True,
                    dropout=self.dropout,
                    edge_dim=self.edge_dim
                ))
            else:
                # Last layer often uses a single head
                self.convs.append(GATConv(
                    in_channels=self.hidden_dim,
                    out_channels=self.hidden_dim,
                    heads=1,
                    concat=False,
                    dropout=self.dropout,
                    edge_dim=self.edge_dim
                ))
    
    def forward(self, data):
        """
        Forward pass.
        
        Parameters
        ----------
        data : torch_geometric.data.Data
            Input graph data
            
        Returns
        -------
        torch.Tensor
            Output tensor
        """
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        # If batch is None, assume a single graph
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        # Convolutional layers
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index, edge_attr=edge_attr)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Global pooling
        x = self.pool(x, batch)
        
        # Readout
        x = self.readout(x)
        
        return x