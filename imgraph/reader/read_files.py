import os 
import os.path as osp
from PIL import Image
import cv2
import numpy as np
import pickle
import networkx as nx

def read_image(path, backend='PIL', **kwargs):
    """Reads an image from a file.
    Args:
        path (str): The path to a local image file.
    Returns:
        A PIL image.
    """
    img = None
    if backend == 'PIL':
    
        img =  Image.open(path).convert('RGB')
    else:
        img =  cv2.imread(path)

    if 'upchannel' in kwargs and kwargs['upchannel'] and backend == 'PIL':
        img = img.convert('RGB')
    elif 'upchannel' in kwargs and kwargs['upchannel'] and backend == 'cv2':
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.stack((img,)*3, axis=-1)

    if 'resize' in kwargs and backend == 'PIL':
        img = img.resize(kwargs['resize'])
    elif 'resize' in kwargs and backend == 'cv2':
        img = cv2.resize(img, kwargs['resize'])

    return img

def read_pickle_file(path):
    """Reads a pickle file from a file.
    Args:
        path (str): The path to a local pickle file.
    Returns:
        A pickle file.
    """
    
    with open(path, 'rb') as f:
        return pickle.load(f)
    
def read_graph(path):
    """Reads a graph from a file.
    Args:
        path (str): The path to a local graph file.
    Returns:
        A networkx graph.
    """
    return nx.read_gpickle(path)



#todo 
def read_csv_file(path):
    """Reads a csv file from a file.
    Args:
        path (str): The path to a local csv file.
    Returns:
        A csv file.
    """
    pass


    
    