
import os 
import os.path as osp
from .url_config import pneumonia_dataset_url
from imgraph.writer import download_from_url
import pickle

ENV_IMGRAPH_HOME = 'IMGRAPH_HOME'
DEFAULT_CACHE_DIR = osp.join('~', '.cache', 'imgraph')


def get_pneumonia_dataset(super_pixels : int, feature_extractor : str) -> list:
    """
    Download the pneumonia datasets
    Args:
        super_pixels (int): The number of super pixels to be used in the graph
        feature_extractor (str): The feature extractor to be used to create the graph. Currently only 'resnet18/efficientnet/densenet121' is supported

    Returns:
        Pneumonia dataset
    """


    train_loader_url = pneumonia_dataset_url[f"train_dataloader_{super_pixels}_{feature_extractor}"]
    test_loader_url = pneumonia_dataset_url[f"test_dataloader_{super_pixels}_{feature_extractor}"]


    path = osp.join(DEFAULT_CACHE_DIR, 'output')
    if os.environ.get(ENV_IMGRAPH_HOME):
        path = osp.join(os.environ.get(ENV_IMGRAPH_HOME), 'output')
    
    path = osp.expanduser(path)

    train_filename = osp.expanduser(osp.join(path, f'train_dataloader_{super_pixels}_{feature_extractor}.pkl'))
    test_filename = osp.expanduser(osp.join(path, f'test_dataloader_{super_pixels}_{feature_extractor}.pkl'))


    download_from_url(train_loader_url, path, f'train_dataloader_{super_pixels}_{feature_extractor}.pkl')
    download_from_url(test_loader_url, path, f'test_dataloader_{super_pixels}_{feature_extractor}.pkl')
    #filepath = osp.join(path, filename)
    
    train_loader = None
    test_loader = None

    print("Loading Pneumonia dataset")
    with open(train_filename, 'rb') as f:
        train_loader = pickle.load(f)

    with open(test_filename, 'rb') as f:
        test_loader = pickle.load(f)

    return train_loader.dataset,test_loader.dataset

