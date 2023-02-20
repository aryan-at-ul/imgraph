from imgraph import image_to_graph,draw_graph_as_image



g,seg = image_to_graph('/Users/aryansingh/projects/imgraph/xray_sample.jpeg')

draw_graph_as_image(g,seg)
