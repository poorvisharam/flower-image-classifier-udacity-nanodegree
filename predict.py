
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import json
import argparse
from train import generate_model

#Gets Argument from CLI
parser = argparse.ArgumentParser()

parser.add_argument('--gpu', default=True, type=bool, help='Device CPU or GPU')
parser.add_argument('--image-file', default='flowers/test/100/image_07896.jpg', type=str, help='test image file for prediction')
parser.add_argument('--class-mapping-file', default='./cat_to_name.json', type=str, help='class to names mapping json')
parser.add_argument('--checkpoint', default='checkpoint.pth', type=str, help='Load the saved checkpoint')
parser.add_argument('--topk', default=5, type=int, help='no. of top K elements')
parser.add_argument('--arch', default='vgg16', type=str, choices=["vgg16", "densenet121"], help='chooses which pretrained model to use')
args = parser.parse_args()

#select device name
device_name = "cpu"
if args.gpu == True:
    if torch.mps.device_count() > 0:
        device_name = "mps"
    elif torch.cuda.is_available():
        device_name = "cuda"

device = torch.device(device_name)

# TODO: Write a function that loads a checkpoint and rebuilds the model
def load_model(file_path, device):
    checkpoint = torch.load(file_path)
    
    arch = checkpoint['arch']
    dropout = checkpoint['dropout']
    # model, criterion, optimizer = generate_model(device, arch, 0.2)
    model = generate_model(device, arch, 0.2)
    
    epochs = checkpoint['epochs']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model

if args.arch not in args.checkpoint:
    checkpoint_file_path = args.arch + "-" + args.checkpoint

saved_model = load_model(checkpoint_file_path, device)
saved_model


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    original_image = Image.open(image)
    
    modifiedImage = transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])
    
    image_value = modifiedImage(original_image)
    
    return image_value

image = process_image(args.image_file)
print(image.shape)


def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

imshow(process_image(args.image_file))

# TODO: Implement the code to predict the class from an image file
def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    image = process_image(image_path)
    image = image.unsqueeze_(0)
    image = image.float()
    
    with torch.no_grad():
        
        logps = model.forward(image.to(device))
        
        prob = F.softmax(logps, dim=1)
        
        
        top_p, top_c = prob.topk(topk, dim = 1)
        top_p = top_p.cpu().numpy()[0]
        top_c = top_c.cpu().numpy().flatten()

        index_to_class = dict()
        for key, value in model.class_to_idx.items():
            index_to_class[value] = key
        top_class = []
        for c in top_c:
            top_class.append(index_to_class[c])
        
    return top_p, top_class


saved_model.to(device)
top_p, top_class = predict(args.image_file, saved_model)
print("the topk probabilty is:" ,top_p)
print("the topk classes is:" ,top_class)

# label mapping
with open(args.class_mapping_file, 'r') as f:
    cat_to_name = json.load(f)
    top_flowers = []
    for f_id in top_class:
        top_flowers.append(cat_to_name[str(f_id)])
    print("the topk flowers is:" ,top_flowers)
