"""
Utility functions for training graph neural networks.
"""

import numpy as np
import torch

class EarlyStopping:
    """
    Early stopping to prevent overfitting.
    
    Parameters
    ----------
    patience : int, optional
        Number of epochs with no improvement after which training will be stopped, by default 7
    verbose : bool, optional
        Whether to print messages, by default False
    delta : float, optional
        Minimum change in the monitored quantity to qualify as an improvement, by default 0.0
    path : str, optional
        Path for the checkpoint to be saved to, by default 'checkpoint.pt'
    """
    
    def __init__(self, patience=7, verbose=False, delta=0.0, path='checkpoint.pt'):
        """Initialize EarlyStopping."""
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
    
    def __call__(self, val_loss, model=None):
        """
        Call EarlyStopping to check if training should be stopped.
        
        Parameters
        ----------
        val_loss : float
            Validation loss
        model : torch.nn.Module, optional
            Model to save, by default None
        """
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
    
    def save_checkpoint(self, val_loss, model=None):
        """
        Saves model when validation loss decreases.
        
        Parameters
        ----------
        val_loss : float
            Validation loss
        model : torch.nn.Module, optional
            Model to save, by default None
        """
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...')
        if model is not None:
            torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

def count_parameters(model):
    """
    Counts the number of trainable parameters in a model.
    
    Parameters
    ----------
    model : torch.nn.Module
        Model to count parameters
        
    Returns
    -------
    int
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def learning_rate_scheduler(optimizer, epoch, initial_lr=0.01, lr_decay_factor=0.1, lr_decay_epochs=[30, 60, 90]):
    """
    Adjusts the learning rate during training.
    
    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        Optimizer
    epoch : int
        Current epoch
    initial_lr : float, optional
        Initial learning rate, by default 0.01
    lr_decay_factor : float, optional
        Factor by which to decay the learning rate, by default 0.1
    lr_decay_epochs : list, optional
        Epochs at which to decay the learning rate, by default [30, 60, 90]
        
    Returns
    -------
    float
        New learning rate
    """
    # Calculate lr_decay_factor^(number of times epoch is in lr_decay_epochs)
    decay = lr_decay_factor ** sum([epoch >= decay_epoch for decay_epoch in lr_decay_epochs])
    
    # Set learning rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = initial_lr * decay
    
    return initial_lr * decay

class LRWarmup:
    """
    Learning rate warmup.
    
    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        Optimizer
    warmup_epochs : int, optional
        Number of epochs for warmup, by default 5
    target_lr : float, optional
        Target learning rate after warmup, by default 0.01
    """
    
    def __init__(self, optimizer, warmup_epochs=5, target_lr=0.01):
        """Initialize LRWarmup."""
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.target_lr = target_lr
        self.initial_lrs = [group['lr'] for group in optimizer.param_groups]
    
    def step(self, epoch):
        """
        Adjust learning rate.
        
        Parameters
        ----------
        epoch : int
            Current epoch
        """
        if epoch >= self.warmup_epochs:
            # Warmup complete, set target lr
            for i, param_group in enumerate(self.optimizer.param_groups):
                param_group['lr'] = self.target_lr
        else:
            # During warmup, linearly increase lr
            warmup_percent = epoch / self.warmup_epochs
            for i, param_group in enumerate(self.optimizer.param_groups):
                param_group['lr'] = self.initial_lrs[i] + (self.target_lr - self.initial_lrs[i]) * warmup_percent

def set_seed(seed):
    """
    Sets random seed for reproducibility.
    
    Parameters
    ----------
    seed : int
        Random seed
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_data_loaders(dataset, batch_size=32, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_seed=42):
    """
    Splits a dataset into train, validation, and test sets and creates DataLoader objects.
    
    Parameters
    ----------
    dataset : torch_geometric.data.Dataset
        Dataset to split
    batch_size : int, optional
        Batch size, by default 32
    train_ratio : float, optional
        Training set ratio, by default 0.7
    val_ratio : float, optional
        Validation set ratio, by default 0.15
    test_ratio : float, optional
        Test set ratio, by default 0.15
    random_seed : int, optional
        Random seed, by default 42
        
    Returns
    -------
    tuple
        (train_loader, val_loader, test_loader)
    """
    from torch_geometric.loader import DataLoader
    
    # Set random seed
    np.random.seed(random_seed)
    
    # Get number of samples
    num_samples = len(dataset)
    
    # Create indices
    indices = list(range(num_samples))
    np.random.shuffle(indices)
    
    # Calculate split sizes
    train_size = int(train_ratio * num_samples)
    val_size = int(val_ratio * num_samples)
    
    # Split indices
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    # Create subset datasets
    train_dataset = [dataset[i] for i in train_indices]
    val_dataset = [dataset[i] for i in val_indices]
    test_dataset = [dataset[i] for i in test_indices]
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    return train_loader, val_loader, test_loader
