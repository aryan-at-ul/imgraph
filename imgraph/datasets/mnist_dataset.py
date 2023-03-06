import os
import os.path as osp
from imgraph.writer import download_from_url,makedirs
import pickle
import torch
import torch.nn.functional as F
from torchvision.datasets import MNIST as MNISTImage
from torchvision.transforms import ToTensor
from torch_geometric.data import InMemoryDataset, Data, DataLoader


ENV_IMGRAPH_HOME = 'IMGRAPH_HOME'
DEFAULT_CACHE_DIR = osp.join('~', '.cache', 'imgraph')

#https://drive.google.com/file/d/1lORmY8srDFTm6a8yqzUOLrrvPOkoUZQ0/view?usp=sharing
#https://drive.google.com/file/d/1fwkebInpfzHHv9M60zvH3YRPbEVBg53L/view?usp=sharing
def get_minst_dataset():
    """
    Args:
        None
    Returns:
        A train_dataset and a test_dataset
    """
    print("Getting minst dataset")
    train_loader_url = "https://drive.google.com/uc?export=download&confirm=yes&id=1lORmY8srDFTm6a8yqzUOLrrvPOkoUZQ0"
    test_loader_url = "https://drive.google.com/uc?export=download&confirm=yes&id=1fwkebInpfzHHv9M60zvH3YRPbEVBg53L"
    path = osp.join(DEFAULT_CACHE_DIR, 'output')
    if os.environ.get(ENV_IMGRAPH_HOME):
        path  = osp.join(os.environ.get(ENV_IMGRAPH_HOME), 'output')
    train_filename = osp.expanduser(osp.join(path, 'mnist_trainloader.pkl'))
    test_filename = osp.expanduser(osp.join(path, 'mnist_testloader.pkl'))

    path = osp.expanduser(path)

    # os.system(f"curl -o {train_filename} -L '{train_loader_url}'")
    # os.system(f"curl -o {test_filename} -L '{test_loader_url}'")
    #url : str, path : str, filename : Optional[str] = None
    download_from_url(train_loader_url,path,'mnist_trainloader.pkl')
    download_from_url(test_loader_url,path,'mnist_testloader.pkl')
    #filepath = osp.join(path, filename)
    train_loader = None
    test_loader = None
    print("Loading minst dataset")
    with open(train_filename, 'rb') as f:
        train_loader = pickle.load(f)

    with open(test_filename, 'rb') as f:
        test_loader = pickle.load(f)

    return train_loader.dataset,test_loader.dataset

