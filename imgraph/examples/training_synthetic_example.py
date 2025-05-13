"""
Example script for training GNN models on synthetic data.
"""

import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
from tqdm import tqdm
import time
import argparse

import matplotlib
matplotlib.use('Agg')  # Use the non-interactive backend
import matplotlib.pyplot as plt

from imgraph import GraphPresets, GraphPipeline
from imgraph.models import GCN, GAT, GIN, GraphSAGE
from imgraph.training.trainer import Trainer
from imgraph.training.utils import EarlyStopping

def create_synthetic_dataset(n_samples=100, image_size=64, shapes=None, colors=None):
    """
    Creates a synthetic dataset of images with different shapes.
    
    Parameters
    ----------
    n_samples : int, optional
        Number of samples to generate, by default 100
    image_size : int, optional
        Size of the images, by default 64
    shapes : list, optional
        List of shapes to use, by default None
        Options: 'circle', 'rectangle', 'triangle', 'cross', 'ellipse'
    colors : list, optional
        List of colors to use, by default None
        
    Returns
    -------
    tuple
        (images, labels, class_names)
    """
    # Default shapes and colors
    if shapes is None:
        shapes = ['circle', 'rectangle', 'triangle', 'cross', 'ellipse']
    
    if colors is None:
        colors = [
            (255, 0, 0),    # Red
            (0, 255, 0),    # Green
            (0, 0, 255),    # Blue
            (255, 255, 0),  # Yellow
            (255, 0, 255)   # Magenta
        ]
    
    # Generate samples
    images = []
    labels = []
    
    for i in range(n_samples):
        # Create blank image
        img = np.ones((image_size, image_size, 3), dtype=np.uint8) * 255
        
        # Choose a random shape
        shape_idx = np.random.randint(0, len(shapes))
        shape = shapes[shape_idx]
        
        # Choose a random color
        color = colors[np.random.randint(0, len(colors))]
        
        # Calculate center and size
        center = (np.random.randint(image_size // 4, 3 * image_size // 4),
                  np.random.randint(image_size // 4, 3 * image_size // 4))
        size = np.random.randint(image_size // 6, image_size // 3)
        
        # Draw shape
        if shape == 'circle':
            cv2.circle(img, center, size, color, -1)
        elif shape == 'rectangle':
            top_left = (center[0] - size, center[1] - size)
            bottom_right = (center[0] + size, center[1] + size)
            cv2.rectangle(img, top_left, bottom_right, color, -1)
        elif shape == 'triangle':
            points = np.array([
                [center[0], center[1] - size],
                [center[0] - size, center[1] + size],
                [center[0] + size, center[1] + size]
            ], np.int32)
            cv2.fillPoly(img, [points], color)
        elif shape == 'cross':
            thickness = size // 3
            cv2.line(img, (center[0] - size, center[1]), (center[0] + size, center[1]), color, thickness)
            cv2.line(img, (center[0], center[1] - size), (center[0], center[1] + size), color, thickness)
        elif shape == 'ellipse':
            axes = (size, size // 2)
            angle = np.random.randint(0, 180)
            cv2.ellipse(img, center, axes, angle, 0, 360, color, -1)
        
        # Add noise
        noise = np.random.randint(0, 20, img.shape).astype(np.uint8)
        img = cv2.add(img, noise)
        
        # Add to dataset
        images.append(img)
        labels.append(shape_idx)
    
    return np.array(images), np.array(labels), shapes

def create_graph_dataset(images, labels, preset='slic_mean_color'):
    """
    Creates a graph dataset from images.
    
    Parameters
    ----------
    images : numpy.ndarray
        Array of images
    labels : numpy.ndarray
        Array of labels
    preset : str, optional
        Graph preset to use, by default 'slic_mean_color'
        
    Returns
    -------
    list
        List of graph data objects
    """
    # Create graph builder
    if preset == 'slic_mean_color':
        graph_builder = GraphPresets.slic_mean_color()
    elif preset == 'slic_color_position':
        graph_builder = GraphPresets.slic_color_position()
    elif preset == 'patches_color':
        graph_builder = GraphPresets.patches_color()
    elif preset == 'tiny_graph':
        graph_builder = GraphPresets.tiny_graph()
    elif preset == 'superpixel_comprehensive':
        graph_builder = GraphPresets.superpixel_comprehensive()
    else:
        raise ValueError(f"Unknown preset: {preset}")
    
    # Create graphs
    graphs = []
    for i, (image, label) in enumerate(tqdm(zip(images, labels), total=len(images), desc="Creating graphs")):
        # Convert image to graph
        graph = graph_builder(image)
        
        # Add label
        graph.y = torch.tensor(label, dtype=torch.long)
        
        graphs.append(graph)
    
    return graphs

def train_model(model_name, train_loader, val_loader, num_features, num_classes, 
                device, lr=0.001, epochs=100, patience=10):
    """
    Trains a GNN model.
    
    Parameters
    ----------
    model_name : str
        Name of the model to train
    train_loader : torch_geometric.loader.DataLoader
        Training data loader
    val_loader : torch_geometric.loader.DataLoader
        Validation data loader
    num_features : int
        Number of input features
    num_classes : int
        Number of output classes
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

def plot_results(history, predictions, targets, class_names, output_dir='results'):
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
    class_names : list
        List of class names
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
    
    # Plot confusion matrix
    cm = confusion_matrix(targets, predictions)
    plt.figure(figsize=(10, 8))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    
    # Print classification report
    report = classification_report(targets, predictions, target_names=class_names)
    print("\nClassification Report:")
    print(report)
    
    # Save report to file
    with open(os.path.join(output_dir, 'classification_report.txt'), 'w') as f:
        f.write(report)

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train GNN models on synthetic data')
    parser.add_argument('--output_dir', type=str, default='results', help='Output directory for results')
    parser.add_argument('--model', type=str, default='gcn', choices=['gcn', 'gat', 'gin', 'sage'], help='GNN model to train')
    parser.add_argument('--preset', type=str, default='slic_color_position', 
                       choices=['slic_mean_color', 'slic_color_position', 'patches_color', 'tiny_graph', 'superpixel_comprehensive'],
                       help='Graph preset to use')
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
    os.makedirs(os.path.join(args.output_dir, 'samples'), exist_ok=True)
    
    # Generate synthetic dataset
    print("Generating synthetic dataset...")
    images, labels, class_names = create_synthetic_dataset(
        n_samples=args.num_samples,
        image_size=args.image_size
    )
    
    # Save some sample images
    for i in range(min(10, len(images))):
        plt.figure(figsize=(3, 3))
        plt.imshow(images[i])
        plt.title(class_names[labels[i]])
        plt.axis('off')
        plt.savefig(os.path.join(args.output_dir, 'samples', f'sample_{i}.png'))
        plt.close()
    
    # Create graph dataset
    print(f"Creating graph dataset with {args.preset} preset...")
    dataset = create_graph_dataset(images, labels, preset=args.preset)
    
    # Get number of features and classes
    num_features = dataset[0].x.shape[1]
    num_classes = len(class_names)
    
    print(f"Number of samples: {len(dataset)}")
    print(f"Number of features: {num_features}")
    print(f"Number of classes: {num_classes}")
    
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
    
    # Train model
    print(f"Training {args.model.upper()} model...")
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
    torch.save(model.state_dict(), os.path.join(args.output_dir, f"{args.model}_model.pt"))
    
    print(f"Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()
