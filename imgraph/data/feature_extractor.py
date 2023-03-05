
import os
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor
from functools import lru_cache
from torchvision import transforms
import torchvision 
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@lru_cache(maxsize=100000)
def get_feture_extractor_model(model_name):
    
    # auto_transform = weights.transforms()
    weights = torchvision.models.ResNet18_Weights.DEFAULT 
    model = models.resnet18(weights=weights).to(device)

    if model_name == 'resnet18':
        weights = torchvision.models.ResNet18_Weights.DEFAULT 
        model = models.resnet18(weights=weights).to(device)
        
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


def feature_from_img(img, model, feature_extractor, i = 0):

   mean = [0.485, 0.456, 0.406]
   std = [0.229, 0.224, 0.225]

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

   img_normalized = transform_norm(img).float()
   img_normalized = img_normalized.unsqueeze_(0)
   # input = Variable(image_tensor)
   img_normalized = img_normalized.to(device)
   # print(img_normalized.shape)
   with torch.no_grad():
      model.eval()
      output =model(img_normalized)
      out = feature_extractor(img_normalized)
      
      return out['flatten'],i

