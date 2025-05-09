"""
Graph Isomorphism Network (GIN) model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, GINEConv
from imgraph.models.base import BaseGNN

class GIN(BaseGNN):
    """
    Graph Isomorphism Network (GIN) model.
    
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
    eps : float, optional
        Epsilon value for GIN, by default 0.0
    train_eps : bool, optional
        Whether to train epsilon, by default False
    use_batch_norm : bool, optional
        Whether to use batch normalization, by default True
    """
    
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.0, 
                 pool='mean', eps=0.0, train_eps=False, use_batch_norm=True):
        """Initialize the GIN model."""
        self.eps = eps
        self.train_eps = train_eps
        self.use_batch_norm = use_batch_norm
        super(GIN, self).__init__(input_dim, hidden_dim, output_dim, num_layers, dropout, pool)
    
    def init_layers(self):
        """Initialize GIN layers."""
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList() if self.use_batch_norm else None
        
        # Input layer
        nn1 = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        self.convs.append(GINConv(nn1, eps=self.eps, train_eps=self.train_eps))
        if self.use_batch_norm:
            self.bns.append(nn.BatchNorm1d(self.hidden_dim))
        
        # Hidden layers
        for i in range(self.num_layers - 1):
            nn_layer = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim)
            )
            self.convs.append(GINConv(nn_layer, eps=self.eps, train_eps=self.train_eps))
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

class GINWithEdgeFeatures(BaseGNN):
    """
    Graph Isomorphism Network (GIN) with edge features (GINE).
    
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
    eps : float, optional
        Epsilon value for GIN, by default 0.0
    train_eps : bool, optional
        Whether to train epsilon, by default False
    use_batch_norm : bool, optional
        Whether to use batch normalization, by default True
    """
    
    def __init__(self, input_dim, edge_dim, hidden_dim, output_dim, num_layers=2, dropout=0.0,
                 pool='mean', eps=0.0, train_eps=False, use_batch_norm=True):
        """Initialize the GINE model."""
        self.edge_dim = edge_dim
        self.eps = eps
        self.train_eps = train_eps
        self.use_batch_norm = use_batch_norm
        super(GINWithEdgeFeatures, self).__init__(input_dim, hidden_dim, output_dim, num_layers, dropout, pool)
    
    def init_layers(self):
        """Initialize GINE layers."""
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList() if self.use_batch_norm else None
        
        # Input layer
        nn1 = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        self.convs.append(GINEConv(nn1, eps=self.eps, train_eps=self.train_eps, edge_dim=self.edge_dim))
        if self.use_batch_norm:
            self.bns.append(nn.BatchNorm1d(self.hidden_dim))
        
        # Hidden layers
        for i in range(self.num_layers - 1):
            nn_layer = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim)
            )
            self.convs.append(GINEConv(nn_layer, eps=self.eps, train_eps=self.train_eps, edge_dim=self.edge_dim))
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
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        # If batch is None, assume a single graph
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        # Convolutional layers
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index, edge_attr)
            
            if self.use_batch_norm:
                x = self.bns[i](x)
            
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Global pooling
        x = self.pool(x, batch)
        
        # Readout
        x = self.readout(x)
        
        return x