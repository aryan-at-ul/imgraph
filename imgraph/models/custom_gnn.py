import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from typing import List, Dict, Union, Callable, Optional, Tuple, Any
from imgraph.models.base import BaseGNN


class CustomGNN(BaseGNN):
    """
    A flexible GNN model that accepts arbitrary PyG GNN layers, pooling methods, and custom layers.
    
    Compatible with the ImGraph training pipeline and follows the structure of other GNN models in the library.
    
    Parameters:
    -----------
    num_features : int
        Number of input features
    hidden_dim : int
        Number of hidden features
    num_classes : int
        Number of output classes/values
    num_layers : int
        Number of GNN layers
    gnn_layer_cls : Union[Callable, List[Callable]]
        GNN layer class(es) from PyG or custom implementation
    pooling_method : Union[str, Callable]
        Method to pool node features to graph representation
        Supported string values: 'mean', 'max', 'sum'
    dropout : float
        Dropout probability
    activation : Callable
        Activation function to use between layers
    kwargs : Dict
        Additional keyword arguments to pass to the GNN layers
    """
    
    def __init__(
        self,
        num_features: int,
        hidden_dim: int,
        num_classes: int,
        num_layers: int = 3,
        gnn_layer_cls: Union[Callable, List[Callable]] = None,
        pooling_method: Union[str, Callable] = 'mean',
        dropout: float = 0.5,
        activation: Callable = F.relu,
        **kwargs
    ):
        # Call the parent class constructor with required parameters
        super().__init__(
            input_dim=num_features,
            hidden_dim=hidden_dim,
            output_dim=num_classes,
            num_layers=num_layers,
            dropout=dropout,
            pool=pooling_method if isinstance(pooling_method, str) else 'mean'
        )
        
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.dropout = dropout
        self.activation = activation
        
        # Handle single layer class or list of layer classes
        if gnn_layer_cls is None:
            raise ValueError("gnn_layer_cls must be provided")
        
        # Convert single layer class to list of the same class
        if not isinstance(gnn_layer_cls, list):
            gnn_layer_cls = [gnn_layer_cls] * num_layers
        
        # Ensure we have enough layer classes
        if len(gnn_layer_cls) < num_layers:
            gnn_layer_cls.extend([gnn_layer_cls[-1]] * (num_layers - len(gnn_layer_cls)))
        
        # Store the custom pooling method if it's a callable (not a string)
        if not isinstance(pooling_method, str):
            self.pool = pooling_method
        
        # Create GNN layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # First layer: num_features to hidden_dim
        self.convs.append(gnn_layer_cls[0](num_features, hidden_dim, **kwargs))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Hidden layers: hidden_dim to hidden_dim
        for i in range(1, num_layers - 1):
            self.convs.append(gnn_layer_cls[i](hidden_dim, hidden_dim, **kwargs))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Last layer: hidden_dim to hidden_dim (not to num_classes yet)
        # We'll use the base class's readout layers for the final classification
        if num_layers > 1:
            self.convs.append(gnn_layer_cls[-1](hidden_dim, hidden_dim, **kwargs))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
    
    def init_layers(self):
        """
        Initialize layers - this is already handled in __init__ so we provide an empty implementation
        to satisfy the abstract method requirement.
        """
        pass
    
    def forward(self, data):
        """
        Forward pass of the CustomGNN model.
        
        Parameters:
        -----------
        data : torch_geometric.data.Data
            The input graph data
        
        Returns:
        --------
        torch.Tensor
            Output predictions (logits)
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Process through GNN layers
        for i, conv in enumerate(self.convs):
            # Check if the layer supports edge attributes
            if hasattr(data, 'edge_attr') and hasattr(conv, 'supports_edge_attr') and conv.supports_edge_attr:
                x = conv(x, edge_index, data.edge_attr)
            else:
                x = conv(x, edge_index)
            
            x = self.batch_norms[i](x)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Apply pooling if batch information is available
        if batch is not None:
            x = self.pool(x, batch)
        
        # Use the base class's readout layers for final classification
        out = self.readout(x)
        
        return out


class CustomGNNWithEdgeFeatures(CustomGNN):
    """
    A version of CustomGNN that explicitly handles edge features.
    
    This class is provided for API consistency with other models in the package.
    """
    
    def __init__(
        self,
        num_features: int,
        hidden_dim: int,
        num_classes: int,
        edge_dim: int,
        num_layers: int = 3,
        gnn_layer_cls: Union[Callable, List[Callable]] = None,
        pooling_method: Union[str, Callable] = 'mean',
        dropout: float = 0.5,
        activation: Callable = F.relu,
        **kwargs
    ):
        super().__init__(
            num_features=num_features,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            num_layers=num_layers,
            gnn_layer_cls=gnn_layer_cls,
            pooling_method=pooling_method,
            dropout=dropout,
            activation=activation,
            edge_dim=edge_dim,
            **kwargs
        )
        
        # Mark edge features as supported for layers that don't explicitly declare it
        for conv in self.convs:
            if hasattr(conv, 'supports_edge_attr'):
                continue
            setattr(conv, 'supports_edge_attr', True)