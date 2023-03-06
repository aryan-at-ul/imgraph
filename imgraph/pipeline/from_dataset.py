import os 
import os.path as osp  
from typing import Optional
from imgraph.datasets import get_minst_dataset





def load_saved_datasets(dataset_name: str, root: Optional[str] = None) -> None:
    r"""Loads the dataset from the local cache.

    Args:
        dataset_name (str): The name of the dataset.
        root (str, optional): The root directory where the dataset should be saved.
            (default: :obj:`None`)
    Returns:
        The dataset object.
    """

    if dataset_name.lower() == 'mnist':
        train_dataset, test_dataset = get_minst_dataset()
        return train_dataset, test_dataset
    


    