"""
Functions for extracting CNN features from image regions.
"""

import numpy as np
import torch
import warnings

def pretrained_cnn_features(image, node_info, model_name='resnet18', layer=None, transform=None):
    """
    Extracts features from a pretrained CNN model.
    
    Parameters
    ----------
    image : numpy.ndarray
        Input image with shape (H, W, C)
    node_info : dict
        Node information dictionary from node creation
    model_name : str, optional
        Name of the pretrained model to use, by default 'resnet18'
    layer : str, optional
        Name of the layer to extract features from, by default None (use final feature layer)
    transform : callable, optional
        Transformation to apply to image patches, by default None
        
    Returns
    -------
    torch.Tensor
        Node features tensor with shape (N, F) where F depends on the model and layer
    """
    try:
        import torchvision.models as models
        from torchvision import transforms
        import cv2
    except ImportError:
        raise ImportError("torchvision and opencv-python are required for CNN feature extraction.")
    
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load pretrained model
    if model_name == 'resnet18':
        model = models.resnet18(pretrained=True)
        if layer is None:
            layer = 'avgpool'
    elif model_name == 'resnet50':
        model = models.resnet50(pretrained=True)
        if layer is None:
            layer = 'avgpool'
    elif model_name == 'vgg16':
        model = models.vgg16(pretrained=True)
        if layer is None:
            layer = 'features.29'
    elif model_name == 'mobilenet_v2':
        model = models.mobilenet_v2(pretrained=True)
        if layer is None:
            layer = 'features.18'
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
    # Set model to evaluation mode
    model.eval()
    model.to(device)
    
    # Setup feature extraction
    features = []
    activation = {}

    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook
    
    # Register hook to extract features
    try:
        if '.' in layer:
            # For nested modules like 'features.29' in VGG
            main_module, sub_module = layer.split('.', 1)
            sub_module = int(sub_module) if sub_module.isdigit() else sub_module
            getattr(getattr(model, main_module), sub_module).register_forward_hook(get_activation(layer))
        else:
            # For direct modules like 'avgpool' in ResNet
            getattr(model, layer).register_forward_hook(get_activation(layer))
    except Exception as e:
        warnings.warn(f"Error registering hook for layer {layer}: {e}. Using final layer.")
        # Use the final feature layer as fallback
        if hasattr(model, 'avgpool'):
            model.avgpool.register_forward_hook(get_activation('avgpool'))
        else:
            model.features[-1].register_forward_hook(get_activation('features'))
    
    # Define default transform if none provided
    if transform is None:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    # Extract features based on node type
    with torch.no_grad():
        if 'segments' in node_info:  # Superpixel nodes
            masks = node_info['masks']
            bboxes = node_info['bboxes']
            
            for mask, bbox in zip(masks, bboxes):
                if isinstance(bbox, tuple):
                    top, left, bottom, right = bbox
                else:
                    top, left, bottom, right = bbox
                
                # Extract patch
                patch = image[top:bottom, left:right].copy()
                
                # Skip empty patches
                if patch.size == 0 or patch.shape[0] == 0 or patch.shape[1] == 0:
                    # Use zeros as features
                    if len(features) > 0:
                        features.append(np.zeros_like(features[0]))
                    else:
                        # Determine feature size by running a dummy forward pass
                        dummy = np.zeros((224, 224, 3), dtype=np.uint8)
                        dummy_tensor = transform(dummy).unsqueeze(0).to(device)
                        model(dummy_tensor)
                        feat_size = activation[list(activation.keys())[0]].cpu().numpy().flatten().shape[0]
                        features.append(np.zeros(feat_size))
                    continue
                
                # Apply mask within the bbox
                mask_cropped = mask[top:bottom, left:right]
                
                # Create masked patch
                patch_masked = patch.copy()
                if len(patch.shape) == 3:
                    for c in range(patch.shape[2]):
                        patch_masked[:, :, c] = patch[:, :, c] * mask_cropped
                else:
                    patch_masked = patch * mask_cropped
                
                # Resize to model's expected input size
                patch_resized = cv2.resize(patch_masked, (224, 224))
                
                # Convert to tensor and normalize
                patch_tensor = transform(patch_resized).unsqueeze(0).to(device)
                
                # Forward pass
                model(patch_tensor)
                
                # Get features
                key = list(activation.keys())[0]
                feat = activation[key].cpu().numpy().flatten()
                features.append(feat)
        
        elif 'patches' in node_info:  # Patch nodes
            patches = node_info['patches']
            
            for patch in patches:
                # Skip empty patches
                if patch.size == 0 or patch.shape[0] == 0 or patch.shape[1] == 0:
                    # Use zeros as features
                    if len(features) > 0:
                        features.append(np.zeros_like(features[0]))
                    else:
                        # Determine feature size by running a dummy forward pass
                        dummy = np.zeros((224, 224, 3), dtype=np.uint8)
                        dummy_tensor = transform(dummy).unsqueeze(0).to(device)
                        model(dummy_tensor)
                        feat_size = activation[list(activation.keys())[0]].cpu().numpy().flatten().shape[0]
                        features.append(np.zeros(feat_size))
                    continue
                
                # Resize to model's expected input size
                patch_resized = cv2.resize(patch, (224, 224))
                
                # Convert to tensor and normalize
                patch_tensor = transform(patch_resized).unsqueeze(0).to(device)
                
                # Forward pass
                model(patch_tensor)
                
                # Get features
                key = list(activation.keys())[0]
                feat = activation[key].cpu().numpy().flatten()
                features.append(feat)
        
        elif 'values' in node_info:  # Pixel nodes
            # For pixel nodes, we can't extract meaningful CNN features
            # Just return zeros or compute features from small patches around each pixel
            positions = node_info['positions']
            
            # Determine feature size by running a dummy forward pass
            dummy = np.zeros((224, 224, 3), dtype=np.uint8)
            dummy_tensor = transform(dummy).unsqueeze(0).to(device)
            model(dummy_tensor)
            feat_size = activation[list(activation.keys())[0]].cpu().numpy().flatten().shape[0]
            
            features = np.zeros((len(positions), feat_size))
    
    # Convert to tensor
    return torch.tensor(np.array(features), dtype=torch.float)
