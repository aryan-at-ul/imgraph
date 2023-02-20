import os
import sys
# from feature_extraction import fet_from_img
import networkx as nx
import numpy as np
import cv2
import time
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import traceback
import torch
import torchvision
from torchvision import models
from skimage.util import img_as_float
# from torchvision import transforms
# from torchvision import datasets
# from torchvision import utils
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage import data, segmentation
from skimage import io, color
from skimage.io import imread
from skimage.util import img_as_float
import matplotlib.pyplot as plt
import argparse
from skimage.future import graph
from skimage.measure import regionprops
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image
import torchvision.io as io
import numpy as np
# from IPython.display import Image as dImage
import os
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor

from torchvision import transforms
import torchvision 
import torch

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from random import randint
import pandas as pd
import pickle
# from models import get_feture_extractor_model
from functools import lru_cache


current_file_path = os.path.dirname(os.path.abspath(__file__))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image
import torchvision.io as io
import numpy as np
# from IPython.display import Image as dImage
import os
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor

from torchvision import transforms
import torchvision 
import torch

from sklearn.cluster import KMeans
# from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
# from random import randint
# import pandas as pd
import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


path = os.path.dirname(os.path.abspath(__file__))

def get_feture_extractor_model(model_name):

    # auto_transform = weights.transforms()

    if model_name == 'resnet18':

        model = models.resnet18(pretrained=True).to(device)

    elif model_name == 'efficientnet_b0':
        weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
        model = models.efficientnet_b0(weights = weights).to(device)

    else:
        weights = torchvision.models.DenseNet121_Weights.DEFAULT
        model = models.densenet121(weights = weights).to(device)#torch.load(f"{current_file_path}/saved_models/model_densenet_head.pt").to(device)

    layes_names = get_graph_node_names(model)
    # model.eval()
    feature_extractor = create_feature_extractor(
        model, return_nodes=['flatten'])

    return model, feature_extractor


def draw_graph_as_image(G,segments):
    pos = {c.label-1: np.array([c.centroid[1],c.centroid[0]]) for c in regionprops(segments+1)}
    nx.draw_networkx(G,pos,width=1,edge_color="b",alpha=0.6)
    ax=plt.gca()
    fig=plt.gcf()
    fig.set_size_inches(20, 20)

    trans = ax.transData.transform
    trans2 = fig.transFigure.inverted().transform
    imsize = 0.05
    nodes = list(G.nodes())
    nodes = nodes[::-1]
    for n in nodes:
        (x,y) = pos[n]
        xx,yy = trans((x,y))
        xa,ya = trans2((xx,yy))
        a = plt.axes([xa-imsize/2.0,ya-imsize/2.0, imsize, imsize ])
        a.imshow(cv2.cvtColor(G.nodes[n]['image'], cv2.COLOR_BGR2RGB))
        a.set_aspect('equal')
        a.axis('off')
    plt.show()



def fet_from_img(img, model, feature_extractor, i = 0):

   # print("reaching here")
#    model, feature_extractor = get_model(model_name)
   mean = [0.485, 0.456, 0.406]
   std = [0.229, 0.224, 0.225]

   # print(img.shape)

   mean = [0.485, 0.485, 0.485]
   std = [0.229, 0.229, 0.229]

   transform_norm = transforms.Compose([transforms.ToTensor(),
   transforms.Resize((224,224)),transforms.Normalize(mean, std)])

   transform_norm = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224,224)),
    #transforms.CenterCrop(224),
   #  transforms.RandomRotation(20),
   #  transforms.RandomHorizontalFlip(),
    transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
   # get normalized image
   img_normalized = transform_norm(img).float()
   img_normalized = img_normalized.unsqueeze_(0)
   # input = Variable(image_tensor)
   img_normalized = img_normalized.to(device)
   # print(img_normalized.shape)
   with torch.no_grad():
      model.eval()
      output =model(img_normalized)
      out = feature_extractor(img_normalized)
      # return out['flatten'],i # this is for densenet and effnet
      return out['flatten'],img,i#.cpu().detach().numpy().ravel(),i



def make_graph_from_image(image,n_segments = 10,model_name = 'resnet18'):
    G2 = nx.Graph()
    model,feature_extractor = get_feture_extractor_model(model_name)
    s = time.time()
    if len(image.shape) < 3:
        image = np.stack((image,)*3, axis=-1)
    segments = slic(image, n_segments=n_segments,compactness= 30)#,sigma = 5)
    rag = graph.rag_mean_color(image, segments,mode='distance')
    e = time.time()
    #print(f"segmentation time = {e-s}")
    seg_imgs = []
    start_time = time.time()
    for (i, segVal) in enumerate(np.unique(segments)):
        #print(f"segment {i} of {len(np.unique(segments))}")
        mask = np.zeros(image.shape[:2], dtype = "uint8")
        mask[segments == segVal] = 255
        segimg = cv2.cvtColor(cv2.bitwise_and(image, image, mask = mask), cv2.COLOR_BGR2RGB)
        segimg = cv2.bitwise_and(image, image, mask = mask)
        gray = cv2.cvtColor(segimg, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)[1]
        cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        seg = segimg.copy()
        for c in cnts:
            x,y,w,h = cv2.boundingRect(c)
            seg = image[y:y+h, x:x+w]
            break
        seg = cv2.cvtColor(seg, cv2.COLOR_BGR2RGB)
        rag.nodes[segVal]['image'] = seg
        seg_imgs.append([seg,segVal])
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(fet_from_img, seg_img[0],model, feature_extractor ,seg_img[1])  for  i,seg_img in enumerate(seg_imgs)]
        for future in concurrent.futures.as_completed(futures):
            try:
                img_fet,img,i = future.result()
                G2.add_node(i,x = img_fet, image = img)
            except Exception as exc:
                # print(f'generated an exception: {exc} for seg {i}')
                print(traceback.format_exc())

    end_time = time.time()
    # print(f"{image_name} total time take per image is {end_time - start_time}")
    edges = rag.edges
    for e in edges:
        G2.add_weighted_edges_from([(e[0],e[1],rag[e[0]][e[1]]['weight'])])

    return G2,segments




def image_to_graph(image_path, n_segments = 10, model_name = 'resnet18'):

    print(image_path)
    image_name = image_path.split('/')[-1]

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    # image = img_as_float(image)

    graph_name = image_name.split('.')[-2] + ".gpickle"


    G,segs = make_graph_from_image(image,n_segments,model_name)
    nx.write_gpickle(G, f"/tmp/{graph_name}")

    return G,segs


if __name__ == "__main__":

    print("start")
    if len(sys.argv) < 2:
        image_path = path + '/' + 'xray_sample.jpeg'
        image_path = '/'.join(image_path.split('/')[:-2]) + '/' +image_path.split('/')[-1]
        class_name = 1
        data_typ = 'sample'
        G,seg = main(image_path)
        draw_graph_as_image(G,seg)
    else:
        image_path = sys.argv[1]
        n_segment = sys.argv[2]
        model_name = sys.argv[3]
        image_name = image_path.split('/')[-1]
        G, seg = main(image_path, n_segment, model_name)
        draw_graph_as_image(G,seg)


