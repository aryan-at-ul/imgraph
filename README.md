# IMGRAPH 

### Used for converting image to graph, uses superpixel method for node creation, extract features from CNN models. 

Example Usage: 

```
from imgraph import image_to_graph,draw_graph_as_image

#use a test image
#model_name  -> default resnet18, possible value densenet121 and efficientnet-b0
g,seg = image_to_graph('test.jpeg', n_segments = 10, model_name = "densenet121")


draw_graph_as_image(g,seg)
```

