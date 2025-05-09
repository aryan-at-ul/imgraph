"""
Base model class for graph neural networks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool

class BaseGNN(nn.Module):
    """
    Base class for Graph Neural Networks.
    
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
    """
    
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.0, pool='mean'):
        """Initialize the BaseGNN."""
        super(BaseGNN, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Set pooling function
        if pool == 'mean':
            self.pool = global_mean_pool
        elif pool == 'max':
            self.pool = global_max_pool
        elif pool == 'sum':
            self.pool = global_add_pool
        else:
            raise ValueError(f"Unknown pooling method: {pool}")
        
        # Initialize layers (to be implemented by subclasses)
        self.init_layers()
        
        # Readout layers for graph classification
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def init_layers(self):
        """Initialize layers (to be implemented by subclasses)."""
        raise NotImplementedError("Subclasses must implement init_layers method")
    
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
        raise NotImplementedError("Subclasses must implement forward method")
    
    def predict(self, data):
        """
        Make predictions.
        
        Parameters
        ----------
        data : torch_geometric.data.Data
            Input graph data
            
        Returns
        -------
        torch.Tensor
            Predicted class probabilities or logits
        """
        self.eval()
        with torch.no_grad():
            return self.forward(data)
    
    def reset_parameters(self):
        """Reset model parameters."""
        # Reset parameters of all layers
        for layer in self.modules():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
    
    def create_explanations(self, data):
        """
        Create explanations for model predictions.
        
        Parameters
        ----------
        data : torch_geometric.data.Data
            Input graph data
            
        Returns
        -------
        dict
            Explanation data
        """
        # Basic implementation - subclasses can override for model-specific explanations
        try:
            from torch_geometric.explain import Explainer, ModelConfig
            
            # Try using PyG's explainer
            model_config = ModelConfig(
                mode="classification",
                task_level="graph",
                return_type="log_probs"
            )
            
            explainer = Explainer(
                model=self,
                algorithm=None,  # Will be set during explanation
                model_config=model_config,
            )
            
            # Use GNNExplainer algorithm
            from torch_geometric.explain.algorithm import GNNExplainer
            explanation = explainer(
                data.x, 
                data.edge_index, 
                algorithm=GNNExplainer(), 
                index=0  # Explain first node by default
            )
            
            return {
                "node_importances": explanation.node_mask.detach(),
                "edge_importances": explanation.edge_mask.detach(),
                "explained_prediction": explanation.prediction.detach()
            }
        
        except (ImportError, Exception) as e:
            # Fallback to simpler approach if PyG explainer is not available
            print(f"Warning: GNN explanation failed ({str(e)}). Using fallback.")
            
            # Get node importances using a simple gradient-based approach
            data = data.clone()
            data.x.requires_grad_(True)
            out = self.forward(data)
            
            if out.shape[0] > 1:  # Graph has multiple nodes or batched
                # For graph classification, use pooled output
                pred_class = out.argmax(dim=1)
                out[range(out.shape[0]), pred_class].sum().backward()
            else:
                # For single prediction
                out.max().backward()
            
            node_importances = data.x.grad.abs().sum(dim=1)
            
            return {
                "node_importances": node_importances.detach(),
                "explained_prediction": out.detach()
            }
    
    def save(self, path):
        """
        Save model to file.
        
        Parameters
        ----------
        path : str
            Path to save model
        """
        torch.save(self.state_dict(), path)
    
    def load(self, path):
        """
        Load model from file.
        
        Parameters
        ----------
        path : str
            Path to load model from
        """
        self.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
        self.eval()
    
    @staticmethod
    def get_optimizer(model, lr=0.01, weight_decay=5e-4, optimizer='adam'):
        """
        Get optimizer for the model.
        
        Parameters
        ----------
        model : nn.Module
            Model to optimize
        lr : float, optional
            Learning rate, by default 0.01
        weight_decay : float, optional
            Weight decay, by default 5e-4
        optimizer : str, optional
            Optimizer type, by default 'adam'
            
        Returns
        -------
        torch.optim.Optimizer
            Optimizer
        """
        if optimizer == 'adam':
            return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer == 'sgd':
            return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer == 'adamw':
            return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer}")