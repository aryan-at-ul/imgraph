import os 
import os.path as osp
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage import data, segmentation
from skimage import io, color
from skimage.io import imread
from skimage.util import img_as_float
from skimage.future import graph
from skimage.measure import regionprops
import numpy as np
import networkx as nx
import time
import cv2
from .feature_extractor import get_feture_extractor_model,feature_from_img
from .transform_graph import load_and_transform
import  errno
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
import time
import traceback
from imgraph.reader import read_image
from imgraph.writer import write_graph



def image_transform_slic(img : np.ndarray, n_segments = 10, compactness = 10, sigma = 0, multichannel = True):
    """
    Args: img: numpy array of the image
            n_segments: number of segments
            compactness: compactness of the segments
            sigma: sigma for the filter
            multichannel: if the image is multichannel
    Returns: numpy array of the image/ segments of the image
    """
    segments = slic(img, n_segments = n_segments, compactness = compactness, sigma = sigma)
    return segments


def make_edges(img : np.ndarray, segments : np.ndarray, task = 'classification', type = True):
    """
    Args: img: numpy array of the image
            segments: numpy array of the segments of the image
            name: name of the graph
            task: classification or regression
            type: type of the graph train/test
    Returns: RAG recency graph
    """
    rag = graph.rag_mean_color(img, segments,mode='distance')
    return rag

def make_graph(img : np.ndarray, name : str, n_segments = 10, compactness = 10, sigma = 1, multichannel = True, task = 'classification', type = True):
    """
    Args: img: numpy array of the image
            segments: numpy array of the segments of the image
            name: name of the graph
            task: classification or regression
            type: type of the graph train/test
    Returns: networkx graph
    """
    segments = image_transform_slic(img, n_segments, compactness, sigma, multichannel)
    G = make_edges(img, segments, task, type)
    return G, segments


def graph_generator(img : np.ndarray, model_name : str, name : str, class_name : str, class_map : dict, n_segments = 10, compactness = 10, sigma = 1, multichannel = True, task = 'classification', type = True):
    """
    Args: img: numpy array of the image
            segments: numpy array of the segments of the image
            name: name of the graph
            task: classification or regression
            type: type of the graph train/test
    Returns: networkx graph
    """
    # print("Generating graph for image: ", name, " with model: ", model_name)
    start_time = time.time()
    G2 = nx.Graph()
    model,feature_extractor = get_feture_extractor_model(model_name)
    if len(img.shape) < 3:
        img = np.stack((img,)*3, axis=-1)
    seg_imgs = []
    G, segments  = make_graph(img, name, n_segments, compactness, sigma, multichannel, task, type)
    for (i, segVal) in enumerate(np.unique(segments)):
        mask = np.zeros(img.shape[:2], dtype = "uint8")
        mask[segments == segVal] = 255
        segimg = cv2.cvtColor(cv2.bitwise_and(img, img, mask = mask), cv2.COLOR_BGR2RGB)
        segimg = cv2.bitwise_and(img, img, mask = mask)
        gray = cv2.cvtColor(segimg, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)[1]
        cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        seg = segimg.copy()
        for c in cnts:
            x,y,w,h = cv2.boundingRect(c)
            seg = img[y:y+h, x:x+w]
            break
        seg = cv2.cvtColor(seg, cv2.COLOR_BGR2RGB)
        G.nodes[segVal]['img'] = seg
        seg_imgs.append([seg,segVal])
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(feature_from_img, seg_img[0],model, feature_extractor ,seg_img[1])  for  i,seg_img in enumerate(seg_imgs)]
        for future in concurrent.futures.as_completed(futures):
            try:
                img_fet,i = future.result()
                G2.add_node(i,x = img_fet)
            except Exception as exc:
                print(f'generated an exception: {exc}')
                print(traceback.format_exc())
   
    end_time = time.time()
    # print(f"{name} total time take per image is {end_time - start_time}")
    edges = G.edges
    for e in edges:
        G2.add_weighted_edges_from([(e[0],e[1],G[e[0]][e[1]]['weight'])])

    data = None
    if task == 'classification':
        data = load_and_transform(G2, name, class_map[class_name])

    return G2,name,data


def graph_generator_from_path(img_path : str, model_name : str, name : str, n_segments = 10, compactness = 10, sigma = 1, multichannel = True, task = 'classification', type = True):   
    """
    Args: img_path: path of the image
            segments: numpy array of the segments of the image
            name: name of the graph
            task: classification or regression
            type: type of the graph train/test
    Returns: PYG Data object
    """
    img = read_image(img_path)
    G,segments = graph_generator(img, model_name, name, n_segments, compactness, sigma, multichannel, task, type)
    # write_graph(G, name)
    data = None
    if task == 'classification':
        data = load_and_transform(G, name, type)
    return data if task == 'classification' else G