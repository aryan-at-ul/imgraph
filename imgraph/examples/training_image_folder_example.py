"""
Example script for training GNN models on a folder of images.

This script demonstrates:
1. How to use ImageFolderGraphDataset to create graphs from images
2. How to train and evaluate different GNN models (GCN, GAT, GIN, GraphSAGE)

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
import shutil
import traceback
import matplotlib.pyplot as plt

from torch_geometric.loader import DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import argparse

from imgraph.datasets import ImageFolderGraphDataset
from imgraph.models import GCN, GAT, GIN, GraphSAGE
from imgraph.training.trainer import Trainer
from imgraph.training.utils import EarlyStopping

def prepare_image_folder_structure(input_dir, dataset_dir):
    """Prepares the dataset directory structure."""
    # Create raw and processed directories
    raw_dir = os.path.join(dataset_dir, 'raw')
    processed_dir = os.path.join(dataset_dir, 'processed')
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)
    
    # Check if input has train/test/val structure
    if os.path.isdir(os.path.join(input_dir, 'train')):
        # Find all class names
        class_names = set()
        for split in ['train', 'test', 'val']:
            split_dir = os.path.join(input_dir, split)
            if os.path.isdir(split_dir):
                for cls in os.listdir(split_dir):
                    cls_dir = os.path.join(split_dir, cls)
                    if os.path.isdir(cls_dir):
                        class_names.add(cls)
        
        # Create class directories
        for cls in class_names:
            os.makedirs(os.path.join(raw_dir, cls), exist_ok=True)
        
        # Create symbolic links or copy files
        for split in ['train', 'test', 'val']:
            split_dir = os.path.join(input_dir, split)
            if os.path.isdir(split_dir):
                for cls in class_names:
                    cls_dir = os.path.join(split_dir, cls)
                    if os.path.isdir(cls_dir):
                        for img in os.listdir(cls_dir):
                            src_path = os.path.join(cls_dir, img)
                            if os.path.isfile(src_path):
                                # Create a unique filename to avoid collisions
                                dst_name = f"{split}_{img}"
                                dst_path = os.path.join(raw_dir, cls, dst_name)
                                
                                if not os.path.exists(dst_path):
                                    try:
                                        os.symlink(os.path.abspath(src_path), dst_path)
                                    except OSError:
                                        shutil.copy2(src_path, dst_path)
        
        print(f"Prepared dataset with {len(class_names)} classes")
    else:
        # Direct copy/link of the directory structure
        for item in os.listdir(input_dir):
            src_path = os.path.join(input_dir, item)
            if os.path.isdir(src_path):
                dst_path = os.path.join(raw_dir, item)
                if not os.path.exists(dst_path):
                    try:
                        os.symlink(os.path.abspath(src_path), dst_path)
                    except OSError:
                        # Create directory and copy files
                        os.makedirs(dst_path, exist_ok=True)
                        for file in os.listdir(src_path):
                            file_path = os.path.join(src_path, file)
                            if os.path.isfile(file_path):
                                shutil.copy2(file_path, os.path.join(dst_path, file))

def main():
    try:
        # Parse arguments
        parser = argparse.ArgumentParser(description='Train GNN models on an image folder')
        parser.add_argument('--input_dir', type=str, required=True, help='Input directory containing images')
        parser.add_argument('--output_dir', type=str, default='output', help='Output directory for processed graphs')
        parser.add_argument('--results_dir', type=str, default='results', help='Results directory for training results')
        parser.add_argument('--model', type=str, default='gcn', choices=['gcn', 'gat', 'gin', 'sage'], help='GNN model to train')
        parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
        parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
        parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
        parser.add_argument('--patience', type=int, default=10, help='Patience for early stopping')
        parser.add_argument('--preset', type=str, default='slic_mean_color', 
                          choices=['slic_mean_color', 'slic_color_position', 'patches_color', 
                                   'tiny_graph', 'superpixel_comprehensive'],
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
        
        # Create output directories
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(args.results_dir, exist_ok=True)
        
        # Prepare dataset directory
        dataset_dir = os.path.join(args.output_dir, 'dataset')
        prepare_image_folder_structure(args.input_dir, dataset_dir)
        
        # Create dataset
        print("\nCreating graph dataset...")
        dataset = ImageFolderGraphDataset(
            root=dataset_dir,
            preset=args.preset,
            force_reload=args.force_reload
        )
        
        print(f"Dataset created with {len(dataset)} samples")
        print(f"Classes: {dataset.classes}")
        
        # Split dataset into train, validation, and test
        print("\nSplitting dataset...")
        train_dataset, test_dataset = dataset.get_train_test_split(
            train_ratio=0.8, 
            stratify=True, 
            random_seed=args.random_seed
        )
        
        # Further split test into validation and test
        from torch.utils.data import Subset
        from sklearn.model_selection import train_test_split
        
        test_indices = list(range(len(test_dataset)))
        val_indices, test_indices = train_test_split(
            test_indices, 
            test_size=0.5, 
            random_state=args.random_seed
        )
        
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
        
        # Create model
        if args.model == 'gcn':
            model = GCN(num_features, 64, num_classes, num_layers=3)
        elif args.model == 'gat':
            model = GAT(num_features, 64, num_classes, num_layers=2, heads=4)
        elif args.model == 'gin':
            model = GIN(num_features, 64, num_classes, num_layers=3)
        elif args.model == 'sage':
            model = GraphSAGE(num_features, 64, num_classes, num_layers=3)
        else:
            raise ValueError(f"Unknown model: {args.model}")
        
        model = model.to(device)
        
        # Create optimizer and loss function
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
        criterion = torch.nn.CrossEntropyLoss()
        
        # Create early stopping
        early_stopping = EarlyStopping(patience=args.patience, verbose=True)
        
        # Create trainer
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            early_stopping=early_stopping
        )
        
        # Train model
        print(f"\nTraining {args.model.upper()} model...")
        history = trainer.fit(train_loader, val_loader, epochs=args.epochs)
        
        # Evaluate model
        model.eval()
        predictions = []
        targets = []
        
        with torch.no_grad():
            for data in test_loader:
                data = data.to(device)
                out = model(data)
                pred = out.argmax(dim=1)
                
                predictions.extend(pred.cpu().numpy())
                targets.extend(data.y.cpu().numpy())
        
        accuracy = accuracy_score(targets, predictions)
        print(f"\nTest accuracy: {accuracy:.4f}")
        
        # Print classification report
        report = classification_report(targets, predictions, target_names=dataset.classes)
        print("\nClassification Report:")
        print(report)
        
        # Save model
        model_path = os.path.join(args.results_dir, f"{args.model}_model.pt")
        torch.save(model.state_dict(), model_path)
        print(f"\nModel saved to {model_path}")
        
        # Also save the report
        with open(os.path.join(args.results_dir, "classification_report.txt"), 'w') as f:
            f.write(f"Model: {args.model}\n")
            f.write(f"Preset: {args.preset}\n")
            f.write(f"Test accuracy: {accuracy:.4f}\n\n")
            f.write(report)
        
        print("Training completed successfully!")
    
    except Exception as e:
        print(f"Error in main function: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()