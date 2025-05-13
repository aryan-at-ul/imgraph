"""
GraphSAGE model implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from imgraph.models.base import BaseGNN

class GraphSAGE(BaseGNN):
    """
    GraphSAGE model.
    
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
    aggr : str, optional
        Aggregation method, by default 'mean'
    use_batch_norm : bool, optional
        Whether to use batch normalization, by default True
    """
    
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.0, 
                 pool='mean', aggr='mean', use_batch_norm=True):
        """Initialize the GraphSAGE model."""
        self.aggr = aggr
        self.use_batch_norm = use_batch_norm
        super(GraphSAGE, self).__init__(input_dim, hidden_dim, output_dim, num_layers, dropout, pool)
    
    def init_layers(self):
        """Initialize GraphSAGE layers."""
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList() if self.use_batch_norm else None
        
        # Input layer
        self.convs.append(SAGEConv(self.input_dim, self.hidden_dim, aggr=self.aggr))
        if self.use_batch_norm:
            self.bns.append(nn.BatchNorm1d(self.hidden_dim))
        
        # Hidden layers
        for i in range(self.num_layers - 1):
            self.convs.append(SAGEConv(self.hidden_dim, self.hidden_dim, aggr=self.aggr))
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

class GraphSAGEWithEdgeFeatures(BaseGNN):
    """
    GraphSAGE model with edge features.
    
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
    aggr : str, optional
        Aggregation method, by default 'mean'
    """
    
    def __init__(self, input_dim, edge_dim, hidden_dim, output_dim, num_layers=2, 
                 dropout=0.0, pool='mean', aggr='mean'):
        """Initialize the GraphSAGE model with edge features."""
        self.edge_dim = edge_dim
        self.aggr = aggr
        super(GraphSAGEWithEdgeFeatures, self).__init__(input_dim, hidden_dim, output_dim, num_layers, dropout, pool)
    
    def init_layers(self):
        """Initialize GraphSAGE layers with edge features."""
        self.convs = nn.ModuleList()
        self.edge_embeddings = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        # Input layer
        self.convs.append(SAGEConv(self.input_dim, self.hidden_dim, aggr=self.aggr))
        self.edge_embeddings.append(nn.Linear(self.edge_dim, 1))
        self.bns.append(nn.BatchNorm1d(self.hidden_dim))
        
        # Hidden layers
        for i in range(self.num_layers - 1):
            self.convs.append(SAGEConv(self.hidden_dim, self.hidden_dim, aggr=self.aggr))
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
        
        # Since SAGEConv doesn't support edge features directly, we'll use edge weights instead
        for i in range(self.num_layers):
            # Compute edge weights from edge features
            if edge_attr is not None:
                edge_weights = torch.sigmoid(self.edge_embeddings[i](edge_attr)).squeeze(-1)
                
                # Apply weighted message passing (this is a simplification, since SAGEConv
                # doesn't directly support edge weights, we apply the weights to the features)
                src, dst = edge_index
                src_features = x[src] * edge_weights.unsqueeze(-1)
                
                # Store original features
                original_x = x.clone()
                
                # Update source features in x (for weighted aggregation)
                for j in range(len(src)):
                    x[src[j]] = src_features[j]
                
                # Apply convolution
                x = self.convs[i](x, edge_index)
                
                # Restore original features
                x = x + original_x
            else:
                # Standard convolution without edge features
                x = self.convs[i](x, edge_index)
            
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Global pooling
        x = self.pool(x, batch)
        
        # Readout
        x = self.readout(x)
        
        return x