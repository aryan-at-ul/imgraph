# IMGRAPH 

### Used for converting image to graph, uses superpixel method for node creation, extract features from CNN models. 

Example Usage: 

```
from imgraph.pipeline import create_graph_pipleline

path = "path/to/image"

create_graph_pipleline(path, 'classification', 'rag', 'resnet18', 10)

```

### Above code will create a graph from the image and save it in the directory .~/cache/imgraph or directory specified by the user in enviornment variable IMGRAPH_HOME.

### The graph will be saved in the PyG Data format or pickle format.




