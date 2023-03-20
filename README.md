# IMGRAPH 

### Used for converting image to graph, uses superpixel method for node creation, extract features from CNN models. 

Example Usage: 

```
    from imgraph.pipeline import create_graph_pipleline

    path = "path/to/image"

    create_graph_pipleline(path, 'classification', 'rag', 'resnet18', 10)

```

### Above code will create a graph from the image and save it in the directory .~/cache/imgraph or directory specified by the user in enviornment variable IMGRAPH_HOME.


### Expected input folder structure: 

```
    image_folder
    ├── test
    │   ├── class1
    │   └── class2
    ├── train
    │   ├── class1
    │   └── class2
    └── val
        ├── class1
        └── class2
```


### The graph will be saved in the PyG Data format or pickle format.

### To install pytorch geometric dependencies, please follow the instructions here: [PyG installation](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html) or use the following code snippet:

    
```
    import torch

    def format_pytorch_version(version):
    return version.split('+')[0]

    TORCH_version = torch.__version__
    TORCH = format_pytorch_version(TORCH_version)

    def format_cuda_version(version):
    return 'cu' + version.replace('.', '')

    CUDA_version = torch.version.cuda
    CUDA = format_cuda_version(CUDA_version)

    !pip install torch-scatter     -f https://pytorch-geometric.com/whl/torch-{TORCH}+{CUDA}.html
    !pip install torch-sparse      -f https://pytorch-geometric.com/whl/torch-{TORCH}+{CUDA}.html
    !pip install torch-cluster     -f https://pytorch-geometric.com/whl/torch-{TORCH}+{CUDA}.html
    !pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-{TORCH}+{CUDA}.html
    !pip install torch-geometric 

```
### To install full dependeciens install using setup.py with full-dependencies flag (its slow, but will install all dependencies)



