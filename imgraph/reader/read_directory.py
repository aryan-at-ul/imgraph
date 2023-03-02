import os




def get_directories_from_path(path):
    """Get all directories from a path.
    Args:
        path (str): The path to a local folder. this folder should contain train, test, validation folders of images of graphs 
    Returns:
        A list of directories. The reruned list is sorted used to iterate through each folder.
    """
    return [os.path.join(path, f) for f in os.listdir(path) if os.path.isdir(os.path.join(path, f)) and not f.startswith('__')]



def get_file_categoryies_from_path(path_list):
    """Get all directories from a path.
    Args:
        path (str): The path to a local folder. this folder should contain train, test, validation folders of images of graphs 
    Returns:
        A list of directories. Returns available catgory names, test, train, val, or class names.
    """
    return [f.split('/')[-1] for f in path_list]


def get_files_from_path(folderpath):
    """Get all files from a path.
    Args:
        folderpath (str): The path to a local folder. this folder should contain train, test, validation folders of images of graphs 
    Returns:
        A list of files. The reruned list is sorted used to iterate through each folder.
    """
    return [os.path.join(folderpath, f) for f in os.listdir(folderpath) if os.path.isfile(os.path.join(folderpath, f))]


def get_files_from_path_with_extension(folderpath, extension):
    """Get all files from a path.
    Args:
        folderpath (str): The path to a local folder. this folder should contain train, test, validation folders of images of graphs 
    Returns:
        A list of files. The reruned list is sorted used to iterate through each folder.
    """
    return [os.path.join(folderpath, f) for f in os.listdir(folderpath) if os.path.isfile(os.path.join(folderpath, f)) and f.endswith(extension)]


def get_files_from_path_with_extension_and_prefix(folderpath, extension, prefix):
    """Get all files from a path.
    Args:
        folderpath (str): The path to a local folder. this folder should contain train, test, validation folders of images of graphs 
    Returns:
        A list of files. The reruned list is sorted used to iterate through each folder.
    """
    return [os.path.join(folderpath, f) for f in os.listdir(folderpath) if os.path.isfile(os.path.join(folderpath, f)) and f.endswith(extension) and f.startswith(prefix)]