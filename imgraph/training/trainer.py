"""
Trainer class for graph neural networks.
"""

import torch
import numpy as np
from tqdm import tqdm
import time

class Trainer:
    """
    Trainer class for graph neural networks.
    
    Parameters
    ----------
    model : torch.nn.Module
        Model to train
    optimizer : torch.optim.Optimizer
        Optimizer for training
    criterion : callable
        Loss function
    device : torch.device
        Device to train on
    early_stopping : EarlyStopping, optional
        Early stopping object, by default None
    """
    
    def __init__(self, model, optimizer, criterion, device, early_stopping=None):
        """Initialize the Trainer."""
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.early_stopping = early_stopping
    
    def train_epoch(self, loader):
        """
        Trains the model for one epoch.
        
        Parameters
        ----------
        loader : torch_geometric.loader.DataLoader
            Data loader
            
        Returns
        -------
        tuple
            (loss, accuracy)
        """
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        # Progress bar
        pbar = tqdm(loader, desc="Training", leave=False)
        
        # Iterate over batches
        for data in pbar:
            # Move data to device
            data = data.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            out = self.model(data)
            
            # Compute loss
            loss = self.criterion(out, data.y)
            
            # Backward pass
            loss.backward()
            
            # Update weights
            self.optimizer.step()
            
            # Update statistics
            total_loss += loss.item() * data.num_graphs
            
            # Compute accuracy
            pred = out.argmax(dim=1)
            correct += pred.eq(data.y).sum().item()
            total += data.num_graphs
            
            # Update progress bar
            pbar.set_postfix(loss=loss.item(), acc=correct/total)
        
        # Close progress bar
        pbar.close()
        
        # Compute epoch statistics
        epoch_loss = total_loss / len(loader.dataset)
        epoch_acc = correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self, loader):
        """
        Validates the model on the given data loader.
        
        Parameters
        ----------
        loader : torch_geometric.loader.DataLoader
            Data loader
            
        Returns
        -------
        tuple
            (loss, accuracy)
        """
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        # Disable gradients
        with torch.no_grad():
            # Iterate over batches
            for data in loader:
                # Move data to device
                data = data.to(self.device)
                
                # Forward pass
                out = self.model(data)
                
                # Compute loss
                loss = self.criterion(out, data.y)
                
                # Update statistics
                total_loss += loss.item() * data.num_graphs
                
                # Compute accuracy
                pred = out.argmax(dim=1)
                correct += pred.eq(data.y).sum().item()
                total += data.num_graphs
        
        # Compute epoch statistics
        epoch_loss = total_loss / len(loader.dataset)
        epoch_acc = correct / total
        
        return epoch_loss, epoch_acc
    
    def fit(self, train_loader, val_loader, epochs=100):
        """
        Trains the model for the given number of epochs.
        
        Parameters
        ----------
        train_loader : torch_geometric.loader.DataLoader
            Training data loader
        val_loader : torch_geometric.loader.DataLoader
            Validation data loader
        epochs : int, optional
            Number of epochs, by default 100
            
        Returns
        -------
        dict
            Training history
        """
        # Initialize history
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
        
        # Initialize best model state
        best_val_loss = float('inf')
        best_model_state = None
        
        # Start timer
        start_time = time.time()
        
        # Training loop
        for epoch in range(epochs):
            # Train epoch
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_acc = self.validate(val_loader)
            
            # Update history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)
            
            # Print progress
            print(f"Epoch {epoch+1}/{epochs} - "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # Check if this is the best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = self.model.state_dict().copy()
            
            # Early stopping
            if self.early_stopping is not None:
                self.early_stopping(val_loss)
                if self.early_stopping.early_stop:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        # End timer
        end_time = time.time()
        
        # Print training time
        training_time = end_time - start_time
        print(f"Training completed in {training_time:.2f} seconds")
        
        # Load best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        return history
    
    def predict(self, loader):
        """
        Makes predictions with the model.
        
        Parameters
        ----------
        loader : torch_geometric.loader.DataLoader
            Data loader
            
        Returns
        -------
        tuple
            (predictions, targets)
        """
        self.model.eval()
        predictions = []
        targets = []
        
        # Disable gradients
        with torch.no_grad():
            # Iterate over batches
            for data in loader:
                # Move data to device
                data = data.to(self.device)
                
                # Forward pass
                out = self.model(data)
                
                # Get predictions
                pred = out.argmax(dim=1)
                
                # Add to lists
                predictions.extend(pred.cpu().numpy())
                targets.extend(data.y.cpu().numpy())
        
        return np.array(predictions), np.array(targets)
    
    def evaluate(self, loader):
        """
        Evaluates the model on the given data loader.
        
        Parameters
        ----------
        loader : torch_geometric.loader.DataLoader
            Data loader
            
        Returns
        -------
        dict
            Evaluation metrics
        """
        # Get predictions
        predictions, targets = self.predict(loader)
        
        # Compute accuracy
        accuracy = np.mean(predictions == targets)
        
        # Return metrics
        return {
            'accuracy': accuracy,
            'predictions': predictions,
            'targets': targets
        }
