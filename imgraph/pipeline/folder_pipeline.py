import os
import os.path as osp
from typing import Optional
from imgraph.reader import get_directories_from_path,get_file_categoryies_from_path,get_files_from_path,read_image
from imgraph.data import graph_generator
from imgraph.writer import makedirs,write_graph,write_pyg_data
# from imgraph.writer import write_graph
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
import traceback
from tqdm import tqdm
import torch
import torchvision.datasets as datasets
from sklearn.preprocessing import LabelEncoder
import networkx as nx




ENV_IMGRAPH_HOME = 'IMGRAPH_HOME'
DEFAULT_CACHE_DIR = osp.join('~', '.cache', 'imgraph')
label_encoder = LabelEncoder()
# path = "/Users/aryansingh/projects/image_segmentation/chest_xray"


def read_images_in_parallel(image_path_dictionary):
    """
    Args:
        image_path_dictionary (dict): A dictionary of image paths
    Returns:
        A list of images
    """
    print("Reading images in parallel")

    for train_test_val in image_path_dictionary.keys():
        for class_name in image_path_dictionary[train_test_val].keys():
            print("reading images for class: ", class_name, " and ", train_test_val, " set")
            img_names = [elem.split('/')[-1].split('.')[0] for elem in image_path_dictionary[train_test_val][class_name]]
            image_name_path_list = zip(image_path_dictionary[train_test_val][class_name],img_names)
            image_path_dictionary[train_test_val][class_name] = {}
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                futures = [executor.submit(read_image, img_info[0], img_info[1])  for  i,img_info in enumerate(image_name_path_list)]
                for future in concurrent.futures.as_completed(futures):
                    try:
                        img,name = future.result()
                        image_path_dictionary[train_test_val][class_name][name] = img
                    except Exception as exc:
                        print(f'generated an exception: {exc}')
                        print(traceback.format_exc())

    return image_path_dictionary

def get_graph(image_dictionary,class_map,cnn_method = 'resnet18', node_count = 10):
    """
    Args: 
        image_dictionary (dict): A dictionary of images
    Returns:    
        A dictionary with the segment of the image
    """
    print("Getting segment of image")
    for train_test_val in image_dictionary.keys():
        for class_name in image_dictionary[train_test_val].keys():
            img_names =  image_dictionary[train_test_val][class_name].keys()
            imgs = image_dictionary[train_test_val][class_name].values()
            img_name_img = zip(img_names, imgs)
            image_dictionary[train_test_val][class_name] = {}
            print("running for class: ", class_name, " and ", train_test_val, " set")
            makedirs(osp.join(output_dir, train_test_val, class_name))
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                futures = [executor.submit(graph_generator, img_and_name[1], cnn_method,img_and_name[0], class_name, class_map,node_count) for img_and_name in img_name_img]   
                for future in tqdm(concurrent.futures.as_completed(futures)):
                    try:
                        G,name,data = future.result()
                        image_dictionary[train_test_val][class_name][name] = G
                        # print("for file: ", name, " graph generated")
                        # write_graph(G, osp.join(output_dir, train_test_val, class_name, name + '.gpickle'))
                        write_pyg_data(data, osp.join(output_dir, train_test_val, class_name, name + '.pt'))
                        # nx.write_gpickle(G, osp.join(output_dir, train_test_val, class_name, name + '.gpickle'))
                    except Exception as exc:
                        print(f'generated an exception: {exc}')
                        print(traceback.format_exc())
                # segment = get_segment(img, node_count)
                # image_path_dictionary[train_test_val][class_name][img_name] = segment




def create_graph_pipleline(path : str, task : str, graph_method : str, feature_extractor, node_count = 10,  **kwargs):
    
    """ Create a graph pipeline from a folder of images
    Args:
        path (str): Path to the folder containing the images
        task (str): The task to be performed on the images. Currently only classification is supported
        graph_method (str): The method to be used to create the graph. Currently only 'SLIC' is supported
        feature_extractor (str): The feature extractor to be used to create the graph. Currently only 'resnet18/efficientnet/densenet121' is supported
        node_count (int): The number of nodes in the graph
        kwargs: Additional arguments to be passed to the graph creation method, like the number of prcoessors to be used
    Returns:
        A dictionary of the images with the graph segment of the image, defaut ~/.cache/imgraph/output else as specified in the environment variable IMGRAPH_HOME

    """


    print("Creating graph pipeline")
    global output_dir
    global class_map
    output_dir = osp.join(DEFAULT_CACHE_DIR, 'output')
    if os.environ.get(ENV_IMGRAPH_HOME):
        makedirs(os.environ.get(ENV_IMGRAPH_HOME))
        output_dir = osp.join(os.environ.get(ENV_IMGRAPH_HOME), 'output')
    else:
        makedirs(DEFAULT_CACHE_DIR)


    if task == 'classification':
        image_path = path 
        image_files = get_directories_from_path(image_path)
        image_categories = get_file_categoryies_from_path(image_files)
        file_path_dictionary = {}
        for train_test_val in image_categories:
            cat_path = osp.join(image_path, train_test_val)
            class_dirs = get_directories_from_path(cat_path)
            classes = get_file_categoryies_from_path(class_dirs)
            label_encoder.fit(classes)
            class_map = dict(zip(classes,label_encoder.transform(classes)))
            if train_test_val not in file_path_dictionary.keys():
                file_path_dictionary[train_test_val] = {}
            for class_name in classes:
                class_files = []
                class_path = osp.join(cat_path, class_name)
                class_files = get_files_from_path(class_path)
                if class_name not in file_path_dictionary[train_test_val].keys():
                    file_path_dictionary[train_test_val][class_name] = {}
                file_path_dictionary[train_test_val][class_name] = class_files
        img_and_name = read_images_in_parallel(file_path_dictionary)
        graph_and_name = get_graph(img_and_name, class_map,feature_extractor, node_count)
                    
    #     return ClassificationPipeline(path, graph_method, feature_extractor, **kwargs)
    # elif task == 'segmentation':
    #     from imgraph.pipeline.segmentation_pipeline import SegmentationPipeline
    #     return SegmentationPipeline(path, graph_method, feature_extractor, **kwargs)
    # elif task == 'detection':
    #     from imgraph.pipeline.detection_pipeline import DetectionPipeline
    #     return DetectionPipeline(path, graph_method, feature_extractor, **kwargs)
    else:
        raise ValueError(f'Invalid task: {task}')


