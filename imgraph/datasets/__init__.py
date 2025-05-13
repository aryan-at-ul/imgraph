from imgraph.datasets.image_folder import ImageFolderGraphDataset
from imgraph.datasets.mnist_dataset import MNISTGraphDataset, get_mnist_dataset  # Add this function
from imgraph.datasets.medmnist_dataset import MedMNISTGraphDataset
from imgraph.datasets.standard import CIFAR10GraphDataset, CIFAR100GraphDataset

__all__ = [
    'ImageFolderGraphDataset',
    'MNISTGraphDataset',
    'MedMNISTGraphDataset',
    'CIFAR10GraphDataset',
    'CIFAR100GraphDataset',
    'get_mnist_dataset'  # Add this function to __all__
]