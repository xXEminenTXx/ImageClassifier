# python imports
import numpy as np
from PIL import Image
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict

# File containing all of the functions used in the predict program
def load_checkpoint(filepath):

    checkpoint = torch.load(filepath)

    model = eval("models." + checkpoint['arch'] + "(pretrained=True)")

    for param in model.parameters():
        param.requires_grad = False

    model.class_to_idx = checkpoint['class_to_idx']

    # Load classifier from checkpoint
    classifier = checkpoint['classifier']

    model.classifier = classifier

    model.load_state_dict(checkpoint['model_state_dict'])

    return model

def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
    returns an Numpy array
    '''

    # Process a PIL image for use in a PyTorch model

    pil_image = Image.open(image_path)

    # Resize
    if pil_image.size[0] > pil_image.size[1]:
        pil_image.thumbnail((5000, 256))
    else:
        pil_image.thumbnail((256, 5000))

    # Crop 
    left_margin = (pil_image.width-224)/2
    bottom_margin = (pil_image.height-224)/2
    right_margin = left_margin + 224
    top_margin = bottom_margin + 224

    pil_image = pil_image.crop((left_margin, bottom_margin, right_margin, top_margin))

    # Normalize
    np_image = np.array(pil_image)/255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std

    # PyTorch expects the color channel to be the first dimension but it's the third dimension in the PIL image and Numpy array
    # Color channel needs to be first; retain the order of the other two dimensions.
    np_image = np_image.transpose((2, 0, 1))

    return np_image

def predict(image_path, model, topk, gpu):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''

    image = process_image(image_path)

    if gpu:
        model.to('cuda')
        image = torch.from_numpy(image).type(torch.cuda.FloatTensor)
    else:
        image = torch.from_numpy(image).type(torch.FloatTensor)

    # Returns a new tensor with a dimension of size one inserted at the specified position.
    image = image.unsqueeze(0)

    output = model.forward(image)

    probabilities = torch.exp(output)

    # Probabilities and the indices of those probabilities corresponding to the classes
    top_probabilities, top_indices = probabilities.topk(topk)

    # Convert to lists
    top_probabilities = top_probabilities.detach().type(torch.FloatTensor).numpy().tolist()[0] 
    top_indices = top_indices.detach().type(torch.FloatTensor).numpy().tolist()[0] 

    # Convert topk_indices to the actual class labels using class_to_idx
    # Invert the dictionary so you get a mapping from index to class.

    idx_to_class = {value: key for key, value in model.class_to_idx.items()}
    #print(idx_to_class)

    top_classes = [idx_to_class[index] for index in top_indices]

    return top_probabilities, top_classes
