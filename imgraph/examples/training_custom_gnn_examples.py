"""
Example script for training a CustomGNN model on synthetic data.
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from torch_geometric.nn import GCNConv, GATConv, GINConv, SAGEConv
from torch_geometric.nn import TopKPooling, SAGPooling, ASAPooling
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Import your library components
from imgraph import GraphPresets
from imgraph.models import CustomGNN, CustomGNNWithEdgeFeatures
from imgraph.training.trainer import Trainer
from imgraph.training.utils import EarlyStopping

# Import the create_synthetic_dataset and plot_results functions from your existing script
from training_synthetic_example import (
    create_synthetic_dataset, 
    create_graph_dataset, 
    evaluate_model,
    plot_results
)


def create_custom_model(model_type, num_features, hidden_dim, num_classes, edge_dim=None, device=None):
    """
    Creates a CustomGNN model with the specified GNN layer type.
    
    Parameters
    ----------
    model_type : str
        Type of GNN layer to use ('gcn', 'gat', 'gin', 'sage', 'mix')
    num_features : int
        Number of input features
    hidden_dim : int
        Number of hidden features
    num_classes : int
        Number of output classes
    edge_dim : int, optional
        Dimension of edge features, by default None
    device : torch.device, optional
        Device to create the model on, by default None
        
    Returns
    -------
    torch.nn.Module
        Created model
    """
    # Define layer mapping
    layer_map = {
        'gcn': GCNConv,
        'gat': lambda in_c, out_c, **kwargs: GATConv(in_c, out_c, heads=4, concat=False, **kwargs),
        'gin': lambda in_c, out_c, **kwargs: GINConv(
            nn.Sequential(
                nn.Linear(in_c, out_c),
                nn.BatchNorm1d(out_c),
                nn.ReLU(),
                nn.Linear(out_c, out_c)
            ),
            **kwargs
        ),
        'sage': SAGEConv
    }
    
    # Get layer class
    if model_type == 'mix':
        # Mixed model with different layer types
        gnn_layers = [
            layer_map['gcn'],          # First layer: GCN
            layer_map['gat'],          # Second layer: GAT
            layer_map['sage']          # Third layer: SAGE
        ]
    else:
        gnn_layers = layer_map[model_type]
    
    # Create model
    if edge_dim is not None:
        model = CustomGNNWithEdgeFeatures(
            num_features=num_features,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            edge_dim=edge_dim,
            num_layers=3,
            gnn_layer_cls=gnn_layers,
            pooling_method='mean',
            dropout=0.3
        )
    else:
        model = CustomGNN(
            num_features=num_features,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            num_layers=3,
            gnn_layer_cls=gnn_layers,
            pooling_method='mean',
            dropout=0.3
        )
    
    # Move to device if specified
    if device is not None:
        model = model.to(device)
    
    return model


def train_custom_model(model, train_loader, val_loader, device, lr=0.001, epochs=100, patience=10):
    """
    Trains a CustomGNN model.
    
    Parameters
    ----------
    model : torch.nn.Module
        Model to train
    train_loader : torch_geometric.loader.DataLoader
        Training data loader
    val_loader : torch_geometric.loader.DataLoader
        Validation data loader
    device : torch.device
        Device to train on
    lr : float, optional
        Learning rate, by default 0.001
    epochs : int, optional
        Number of epochs, by default 100
    patience : int, optional
        Patience for early stopping, by default 10
        
    Returns
    -------
    tuple
        (trainer, history)
    """
    # Create optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()
    
    # Create early stopping
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        early_stopping=early_stopping
    )
    
    # Train model
    history = trainer.fit(train_loader, val_loader, epochs=epochs)
    
    return trainer, history


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train CustomGNN models on synthetic data')
    parser.add_argument('--output_dir', type=str, default='results_custom', help='Output directory for results')
    parser.add_argument('--model', type=str, default='mix', choices=['gcn', 'gat', 'gin', 'sage', 'mix'], 
                        help='GNN layer type to use')
    parser.add_argument('--preset', type=str, default='slic_color_position', 
                       choices=['slic_mean_color', 'slic_color_position', 'patches_color', 'tiny_graph', 'superpixel_comprehensive'],
                       help='Graph preset to use')
    parser.add_argument('--use_edge_features', action='store_true', help='Use edge features')
    parser.add_argument('--num_samples', type=int, default=200, help='Number of samples to generate')
    parser.add_argument('--image_size', type=int, default=64, help='Size of the images')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--patience', type=int, default=10, help='Patience for early stopping')
    parser.add_argument('--random_seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate synthetic dataset
    print("Generating synthetic dataset...")
    images, labels, class_names = create_synthetic_dataset(
        n_samples=args.num_samples,
        image_size=args.image_size
    )
    
    # Create graph dataset
    print(f"Creating graph dataset with {args.preset} preset...")
    dataset = create_graph_dataset(images, labels, preset=args.preset)
    
    # Get number of features and classes
    num_features = dataset[0].x.shape[1]
    num_classes = len(class_names)
    edge_dim = dataset[0].edge_attr.shape[1] if args.use_edge_features and hasattr(dataset[0], 'edge_attr') else None
    
    print(f"Number of samples: {len(dataset)}")
    print(f"Number of features: {num_features}")
    print(f"Number of classes: {num_classes}")
    if edge_dim:
        print(f"Edge dimension: {edge_dim}")
    
    # Split dataset into train, validation, and test
    train_dataset, test_dataset = train_test_split(dataset, test_size=0.2, random_state=args.random_seed)
    train_dataset, val_dataset = train_test_split(train_dataset, test_size=0.2, random_state=args.random_seed)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    print(f"Creating CustomGNN model with {args.model} layers...")
    hidden_dim = 64
    model = create_custom_model(
        model_type=args.model,
        num_features=num_features,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        edge_dim=edge_dim,
        device=device
    )
    
    # Print model architecture
    print(model)
    
    # Train model
    print(f"Training CustomGNN model...")
    trainer, history = train_custom_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        lr=args.lr,
        epochs=args.epochs,
        patience=args.patience
    )
    
    # Evaluate model on test set
    print("Evaluating model on test set...")
    test_accuracy, test_predictions, test_targets = evaluate_model(model, test_loader, device)
    print(f"Test accuracy: {test_accuracy:.4f}")
    
    # Plot results
    plot_results(
        history=history,
        predictions=test_predictions,
        targets=test_targets,
        class_names=class_names,
        output_dir=args.output_dir
    )
    
    # Save model
    torch.save(model.state_dict(), os.path.join(args.output_dir, f"custom_{args.model}_model.pt"))
    
    print(f"Results saved to {args.output_dir}")


# Example of a custom class to be used with CustomGNN
class CustomEdgeGNNLayer(torch.nn.Module):
    """
    A custom GNN layer that explicitly handles edge features.
    
    This is an example of how to create a custom layer to use with CustomGNN.
    """
    def __init__(self, in_channels, out_channels, edge_dim=None, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_dim = edge_dim
        
        # Node feature transformation
        self.lin = nn.Linear(in_channels, out_channels)
        
        # Edge feature transformation if edge features are provided
        if edge_dim is not None:
            self.edge_lin = nn.Linear(edge_dim, out_channels)
            self.supports_edge_attr = True
    
    def forward(self, x, edge_index, edge_attr=None):
        # Process node features
        x = self.lin(x)
        
        # Process edge features if provided
        if edge_attr is not None and hasattr(self, 'edge_lin'):
            edge_features = self.edge_lin(edge_attr)
            # Apply edge features (this is a simplified example)
            # In a real implementation, you'd use a proper message passing scheme
        
        return x


if __name__ == "__main__":
    main()