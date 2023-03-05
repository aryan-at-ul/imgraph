import os 
import os.path as osp
import torch
# from PIL import Image
import cv2
import numpy as np
import pickle
import networkx as nx
from skimage.io import imread

def read_image(path, name : str ,backend='PIL', **kwargs):
    """Reads an image from a file.
    Args:
        path (str): The path to a local image file.
    Returns:
        A PIL image.
    """
    image = imread(path)
    height, widht = 0,0
    if len(image.shape) >= 3:
        height, width, channel = image.shape
    else:
        height,width = image.shape
    #height, width = image.shape
    # image = image[0:height, 10:width-10]
    try:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (500, 500))
    except Exception as e:
        print("Error in resizing image: ", name, " with error: ", e)
    # image = img_as_float(image)
    return image,name
    # img = None
    # if backend == 'PIL':
    
    #     img =  Image.open(path).convert('RGB')
    # else:
    #     img =  cv2.imread(path)

    # if 'upchannel' in kwargs and kwargs['upchannel'] and backend == 'PIL':
    #     img = img.convert('RGB')
    # elif 'upchannel' in kwargs and kwargs['upchannel'] and backend == 'cv2':
    #     # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #     img = np.stack((img,)*3, axis=-1)

    # if 'resize' in kwargs and backend == 'PIL':
    #     img = img.resize(kwargs['resize'])
    # elif 'resize' in kwargs and backend == 'cv2':
    #     img = cv2.resize(img, kwargs['resize'])

    # return img

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


    
    