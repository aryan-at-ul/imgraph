import os 
import os.path as osp
import ssl
import sys
import urllib
from typing import Optional
from .makedirs import makedirs
import pickle
import networkx as nx
from PIL import Image
import torch


def download_from_url(url : str, path : str, filename : Optional[str] = None):
    """Downloads a file from a URL.
    Args:
        url (str): The URL to download from.
        path (str): The path to save the file to.
    """
    if filename is None:
        filename = url.rpartition('/')[2]
        filename = filename if filename[0] == '?' else filename.split('?')[0]

    filepath = osp.join(path, filename)

    if osp.exists(filepath):  
        return filepath

    makedirs(path)

    context = ssl._create_unverified_context()
    data = urllib.request.urlopen(url, context=context)

    with open(filepath, 'wb') as f:
        # workaround for https://bugs.python.org/issue42853
        while True:
            chunk = data.read(10 * 1024 * 1024)
            if not chunk:
                break
            f.write(chunk)

    return filepath


def write_pickle_file(obj, path):
    """Writes a pickle file to a file.
    Args:
        obj (object): The object to write.
        path (str): The path to a local pickle file.
    """
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

def write_graph(graph, path):
    """Writes a graph to a file.
    Args:
        graph (networkx graph): The graph to write.
        path (str): The path to a local graph file.
    """
    print("writing graph to file", path)
    nx.write_gpickle(graph, osp.expanduser(path))


def write_pyg_data(data, path):
    """Writes a PyG data object to a file.
    Args:
        data (PyG data object): The data object to write.
        path (str): The path to a local PyG data file.
    """
    torch.save(data, osp.expanduser(path))

def write_image(img, path):
    """Writes an image to a file.
    Args:
        img (PIL image): The image to write.
        path (str): The path to a local image file.
    Todo: support other image formats.
    """
    img.save(path)

def write_dataloader(dataloader, path):
    """Writes a dataloader to a file.
    Args:
        dataloader (torch dataloader): The dataloader to write.
        path (str): The path to a local dataloader file.
    """
    with open(path, 'wb') as f:
        pickle.dump(dataloader, f)

#todo for deleting files, for now overwrite it