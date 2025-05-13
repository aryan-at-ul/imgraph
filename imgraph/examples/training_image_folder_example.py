"""
Example script for training GNN models on a folder of images.

This script demonstrates:
1. How to use ImageFolderGraphDataset to create graphs from images
2. How to train and evaluate different GNN models (GCN, GAT, GIN, GraphSAGE)
3. How to visualize and save results

Usage:
    python training_image_folder_example.py --input_dir <path_to_image_folder> [options]

Options:
    --input_dir: Directory containing images organized in class folders
    --output_dir: Directory for saving processed graphs
    --results_dir: Directory for saving training results
    --model: GNN model to train (gcn, gat, gin, sage)
    --preset: Graph representation preset (see choices below)
    --batch_size: Batch size for training
    --epochs: Number of training epochs
    --lr: Learning rate
    --patience: Early stopping patience
    --random_seed: Random seed for reproducibility
"""

import os
import torch
import numpy as np
import traceback
import shutil

# Set non-interactive backend to avoid display issues
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from torch_geometric.loader import DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import argparse

from imgraph.datasets import ImageFolderGraphDataset
from imgraph.models import GCN, GAT, GIN, GraphSAGE
from imgraph.training.trainer import Trainer
from imgraph.training.utils import EarlyStopping

def train_model(model_name, train_loader, val_loader, num_features, num_classes, 
                device, lr=0.001, epochs=100, patience=10):
    """
    Trains a GNN model.
    
    Parameters
    ----------
    model_name : str
        Name of the model to train (gcn, gat, gin, sage)
    train_loader : torch_geometric.loader.DataLoader
        Training data loader
    val_loader : torch_geometric.loader.DataLoader
        Validation data loader
    num_features : int
        Number of input features
    num_classes : int
        Number of output classes
    device : torch.device
        Device to train on (cuda or cpu)
    lr : float, optional
        Learning rate, by default 0.001
    epochs : int, optional
        Number of epochs, by default 100
    patience : int, optional
        Patience for early stopping, by default 10
        
    Returns
    -------
    tuple
        (model, trainer, history)
    """
    # Create model
    hidden_dim = 64
    if model_name.lower() == 'gcn':
        model = GCN(num_features, hidden_dim, num_classes, num_layers=3)
    elif model_name.lower() == 'gat':
        model = GAT(num_features, hidden_dim, num_classes, num_layers=2, heads=4)
    elif model_name.lower() == 'gin':
        model = GIN(num_features, hidden_dim, num_classes, num_layers=3)
    elif model_name.lower() == 'sage':
        model = GraphSAGE(num_features, hidden_dim, num_classes, num_layers=3)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Move model to device
    model = model.to(device)
    
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
    
    return model, trainer, history

def evaluate_model(model, loader, device):
    """
    Evaluates the model on the given data loader.
    
    Parameters
    ----------
    model : torch.nn.Module
        Model to evaluate
    loader : torch_geometric.loader.DataLoader
        Data loader
    device : torch.device
        Device to evaluate on
        
    Returns
    -------
    tuple
        (accuracy, predictions, targets)
    """
    model.eval()
    predictions = []
    targets = []
    
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            pred = out.argmax(dim=1)
            
            predictions.extend(pred.cpu().numpy())
            targets.extend(data.y.cpu().numpy())
    
    accuracy = accuracy_score(targets, predictions)
    
    return accuracy, predictions, targets

def plot_results(history, predictions, targets, class_names=None, output_dir='results'):
    """
    Plots the training history and confusion matrix.
    
    Parameters
    ----------
    history : dict
        Training history
    predictions : list
        Model predictions
    targets : list
        Ground truth targets
    class_names : list, optional
        List of class names, by default None
    output_dir : str, optional
        Output directory, by default 'results'
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot training history
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Validation')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train')
    plt.plot(history['val_acc'], label='Validation')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_history.png'))
    plt.close()  # Close the figure to free memory
    
    # Plot confusion matrix
    cm = confusion_matrix(targets, predictions)
    plt.figure(figsize=(10, 8))
    
    if class_names is not None:
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    else:
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()  # Close the figure to free memory
    
    # Print classification report
    if class_names is not None:
        report = classification_report(targets, predictions, target_names=class_names)
    else:
        report = classification_report(targets, predictions)
    
    print("\nClassification Report:")
    print(report)
    
    # Save report to file
    with open(os.path.join(output_dir, 'classification_report.txt'), 'w') as f:
        f.write(report)

def main():
    try:
        # Parse arguments
        parser = argparse.ArgumentParser(description='Train GNN models on an image folder')
        parser.add_argument('--input_dir', type=str, required=True, help='Input directory containing images')
        parser.add_argument('--output_dir', type=str, default='output', help='Output directory for processed graphs')
        parser.add_argument('--results_dir', type=str, default='results', help='Results directory for training results')
        parser.add_argument('--model', type=str, default='gcn', choices=['gcn', 'gat', 'gin', 'sage'], help='GNN model to train')
        parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
        parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
        parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
        parser.add_argument('--patience', type=int, default=10, help='Patience for early stopping')
        parser.add_argument('--preset', type=str, default='slic_mean_color', 
                          choices=['slic_mean_color', 'slic_color_position', 'patches_color', 'tiny_graph', 'superpixel_comprehensive'],
                          help='Graph preset to use')
        parser.add_argument('--force_reload', action='store_true', help='Force reload the dataset')
        parser.add_argument('--random_seed', type=int, default=42, help='Random seed')
        args = parser.parse_args()
        
        print(f"Starting training with preset: {args.preset}, model: {args.model}")
        print(f"Input directory: {args.input_dir}")
        print(f"Output directory: {args.output_dir}")
        print(f"Results directory: {args.results_dir}")
        
        # Set random seed
        torch.manual_seed(args.random_seed)
        np.random.seed(args.random_seed)
        
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(args.results_dir, exist_ok=True)
        
        # Setup dataset directory structure
        dataset_root = os.path.join(args.output_dir, 'dataset')
        os.makedirs(dataset_root, exist_ok=True)
        
        # Create raw directory
        raw_dir = os.path.join(dataset_root, 'raw')
        os.makedirs(raw_dir, exist_ok=True)
        
        # Create processed directory
        processed_dir = os.path.join(dataset_root, 'processed')
        os.makedirs(processed_dir, exist_ok=True)
        
        # Check if input_dir has train, test subfolders
        has_train_test_dirs = os.path.isdir(os.path.join(args.input_dir, 'train'))
        
        # If input directory has train, test subfolders, prepare class folders in raw dir
        if has_train_test_dirs:
            # Find all class names first
            class_names = set()
            for split in ['train', 'test', 'val']:
                split_dir = os.path.join(args.input_dir, split)
                if os.path.isdir(split_dir):
                    for class_dir in os.listdir(split_dir):
                        class_path = os.path.join(split_dir, class_dir)
                        if os.path.isdir(class_path):
                            class_names.add(class_dir)
            
            # Create class folders in raw dir
            for class_name in class_names:
                os.makedirs(os.path.join(raw_dir, class_name), exist_ok=True)
            
            # Copy images from train and test to class folders in raw dir
            for split in ['train', 'test', 'val']:
                split_dir = os.path.join(args.input_dir, split)
                if os.path.isdir(split_dir):
                    for class_name in class_names:
                        src_dir = os.path.join(split_dir, class_name)
                        dst_dir = os.path.join(raw_dir, class_name)
                        
                        if os.path.isdir(src_dir):
                            # Create symbolic links to all images
                            for img in os.listdir(src_dir):
                                src_path = os.path.join(src_dir, img)
                                dst_path = os.path.join(dst_dir, f"{split}_{img}")  # Add prefix to avoid collisions
                                
                                if os.path.isfile(src_path) and not os.path.exists(dst_path):
                                    try:
                                        os.symlink(src_path, dst_path)
                                    except:
                                        # If symlink fails, copy the file
                                        shutil.copy2(src_path, dst_path)
            
            print(f"Prepared dataset with {len(class_names)} classes.")
        else:
            # If input directory is already in ImageFolder format, just create a symlink or copy
            for item in os.listdir(args.input_dir):
                src_path = os.path.join(args.input_dir, item)
                dst_path = os.path.join(raw_dir, item)
                
                if os.path.isdir(src_path) and not os.path.exists(dst_path):
                    try:
                        os.symlink(src_path, dst_path)
                    except:
                        # If symlink fails, copy the directory structure
                        os.makedirs(dst_path, exist_ok=True)
                        print(f"Created directory {dst_path}")
        
        # Create dataset
        print("Creating graph dataset...")
        dataset = ImageFolderGraphDataset(
            root=dataset_root,
            preset=args.preset,
            force_reload=args.force_reload
        )
        
        print(f"Dataset created with {len(dataset)} samples.")
        print(f"Classes: {dataset.classes}")
        
        # Split dataset into train, validation, and test
        print("\nSplitting dataset...")
        train_dataset, test_dataset = dataset.get_train_test_split(
            train_ratio=0.8, 
            stratify=True, 
            random_seed=args.random_seed
        )
        
        # Further split test into validation and test
        indices = list(range(len(test_dataset)))
        from sklearn.model_selection import train_test_split
        val_indices, test_indices = train_test_split(
            indices, 
            test_size=0.5, 
            random_state=args.random_seed
        )
        
        from torch.utils.data import Subset
        val_dataset = Subset(test_dataset, val_indices)
        test_dataset = Subset(test_dataset, test_indices)
        
        print(f"  Training samples: {len(train_dataset)}")
        print(f"  Validation samples: {len(val_dataset)}")
        print(f"  Test samples: {len(test_dataset)}")
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
        
        # Get feature dimensions
        sample = dataset[0]
        num_features = sample.x.shape[1]
        num_classes = len(dataset.classes)
        
        print("\nFeature information:")
        print(f"  Number of node features: {num_features}")
        print(f"  Number of classes: {num_classes}")
        
        if hasattr(sample, 'edge_attr') and sample.edge_attr is not None:
            print(f"  Edge feature dimensions: {sample.edge_attr.shape[1]}")
        else:
            print("  No edge features")
        
        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"\nUsing device: {device}")
        
        # Train model
        print(f"\nTraining {args.model.upper()} model...")
        model, trainer, history = train_model(
            model_name=args.model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_features=num_features,
            num_classes=num_classes,
            device=device,
            lr=args.lr,
            epochs=args.epochs,
            patience=args.patience
        )
        
        # Evaluate model on test set
        print("\nEvaluating model on test set...")
        test_accuracy, test_predictions, test_targets = evaluate_model(model, test_loader, device)
        print(f"Test accuracy: {test_accuracy:.4f}")
        
        # Plot results
        print("\nGenerating evaluation plots and reports...")
        plot_results(
            history=history,
            predictions=test_predictions,
            targets=test_targets,
            class_names=dataset.classes,
            output_dir=args.results_dir
        )
        
        # Save model
        model_path = os.path.join(args.results_dir, f"{args.model}_model.pt")
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")
        
        # Save preset information
        preset_path = os.path.join(args.results_dir, "preset_info.txt")
        with open(preset_path, 'w') as f:
            f.write(f"Model: {args.model}\n")
            f.write(f"Preset: {args.preset}\n")
            f.write(f"Number of samples: {len(dataset)}\n")
            f.write(f"Number of node features: {num_features}\n")
            f.write(f"Number of classes: {num_classes}\n")
            f.write(f"Classes: {', '.join(dataset.classes)}\n")
        
        print(f"\nAll results saved to {args.results_dir}")
        print("Training completed successfully!")
    
    except Exception as e:
        print(f"Error in main function: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()