"""
Graph Convolutional Network (GCN) model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from imgraph.models.base import BaseGNN

class GCN(BaseGNN):
    """
    Graph Convolutional Network (GCN) model.
    
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
    use_batch_norm : bool, optional
        Whether to use batch normalization, by default True
    """
    
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.0, pool='mean', use_batch_norm=True):
        """Initialize the GCN model."""
        self.use_batch_norm = use_batch_norm
        super(GCN, self).__init__(input_dim, hidden_dim, output_dim, num_layers, dropout, pool)
    
    def init_layers(self):
        """Initialize GCN layers."""
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        # Input layer
        self.convs.append(GCNConv(self.input_dim, self.hidden_dim))
        if self.use_batch_norm:
            self.bns.append(nn.BatchNorm1d(self.hidden_dim))
        
        # Hidden layers
        for i in range(self.num_layers - 1):
            self.convs.append(GCNConv(self.hidden_dim, self.hidden_dim))
            if self.use_batch_norm:
                self.bns.append(nn.BatchNorm1d(self.hidden_dim))
    
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
            
            if self.use_batch_norm:
                x = self.bns[i](x)
            
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Global pooling
        x = self.pool(x, batch)
        
        # Readout
        x = self.readout(x)
        
        return x

class GCNWithEdgeFeatures(BaseGNN):
    """
    Graph Convolutional Network (GCN) with edge features.
    
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
    """
    
    def __init__(self, input_dim, edge_dim, hidden_dim, output_dim, num_layers=2, dropout=0.0, pool='mean'):
        """Initialize the GCN model with edge features."""
        self.edge_dim = edge_dim
        super(GCNWithEdgeFeatures, self).__init__(input_dim, hidden_dim, output_dim, num_layers, dropout, pool)
    
    def init_layers(self):
        """Initialize GCN layers with edge features."""
        from torch_geometric.nn import GCNConv
        
        self.convs = nn.ModuleList()
        self.edge_embeddings = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        # Input layer
        self.convs.append(GCNConv(self.input_dim, self.hidden_dim))
        self.edge_embeddings.append(nn.Linear(self.edge_dim, 1))
        self.bns.append(nn.BatchNorm1d(self.hidden_dim))
        
        # Hidden layers
        for i in range(self.num_layers - 1):
            self.convs.append(GCNConv(self.hidden_dim, self.hidden_dim))
            self.edge_embeddings.append(nn.Linear(self.edge_dim, 1))
            self.bns.append(nn.BatchNorm1d(self.hidden_dim))
    
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
            # Compute edge weights from edge features
            if edge_attr is not None:
                edge_weights = torch.sigmoid(self.edge_embeddings[i](edge_attr)).squeeze(-1)
            else:
                edge_weights = None
            
            # Apply convolution
            x = self.convs[i](x, edge_index, edge_weight=edge_weights)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Global pooling
        x = self.pool(x, batch)
        
        # Readout
        x = self.readout(x)
        
        return x