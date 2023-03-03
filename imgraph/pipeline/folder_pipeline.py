import os
import os.path as osp
from typing import Optional
from imgraph.reader import get_directories_from_path,get_file_categoryies_from_path,get_files_from_path
# from imgraph.writer import write_graph



ENV_IMGRAPH_HOME = 'IMGRAPH_HOME'
DEFAULT_CACHE_DIR = osp.join('~', '.cache', 'imgraph')

path = "/Users/aryansingh/projects/image_segmentation/chest_xray"
def create_graph_pipleline(path : str, task : str, graph_method : str, feature_extractor, node_count = 10,  **kwargs):
    print("Creating graph pipeline")
    if task == 'classification':
        image_path = path 
        image_files = get_directories_from_path(image_path)
        image_categories = get_file_categoryies_from_path(image_files)
        file_path_dictionary = {}
        for train_test_val in image_categories:  
            cat_path = osp.join(image_path, train_test_val)
            class_dirs = get_directories_from_path(cat_path)
            classes = get_file_categoryies_from_path(class_dirs)
            for class_name in classes:
                class_files = []
                class_path = osp.join(cat_path, class_name)
                class_files = get_files_from_path(class_path)
                if train_test_val not in file_path_dictionary.keys():
                    file_path_dictionary[train_test_val] = {}
                else:
                    file_path_dictionary[train_test_val][class_name] = class_files
        # print(file_path_dictionary)
                    
    #     return ClassificationPipeline(path, graph_method, feature_extractor, **kwargs)
    # elif task == 'segmentation':
    #     from imgraph.pipeline.segmentation_pipeline import SegmentationPipeline
    #     return SegmentationPipeline(path, graph_method, feature_extractor, **kwargs)
    # elif task == 'detection':
    #     from imgraph.pipeline.detection_pipeline import DetectionPipeline
    #     return DetectionPipeline(path, graph_method, feature_extractor, **kwargs)
    else:
        raise ValueError(f'Invalid task: {task}')


