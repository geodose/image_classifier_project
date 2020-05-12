import click
import torch
import json
import time
from torchvision import datasets, transforms, models
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import OrderedDict
from workspace_utils import active_session
from PIL import Image

"""
Predict flower name from an image with predict.py along with the probability of that name. That is, you'll pass in a single image /path/to/image and return the flower name and class probability.

Basic usage: python predict.py /path/to/image checkpoint
Options:
Return top KK most likely classes: python predict.py input checkpoint --top_k 3
Use a mapping of categories to real names: python predict.py input checkpoint --category_names cat_to_name.json
Use GPU for inference: python predict.py input checkpoint --gpu
"""

# Setup CLI functionality

@click.command()

@click.argument('image_path', type=click.Path(exists=True))
@click.argument('checkpoint', type=click.Path(exists=True))

@click.option('--top-k', type=click.IntRange(1,25), default=1, help='Return top K most likely classes. Range between 1 and 25.')
@click.option('--category_names', type=click.Path(exists=True), help='Path to json file with category names.')
@click.option('--gpu', is_flag=True, help='Set this flag to train on GPU, if available.')

def predict(image_path, checkpoint, top_k, category_names, gpu):
    # First, define helper function to process input image
    
    def process_image(image):
        ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
            returns a PyTorch Tensor!!! Not NumPy Array
        '''
    
        im = Image.open(image)
        width, height = im.size
        means = np.array([0.485, 0.456, 0.406])
        stds = np.array([0.229, 0.224, 0.225])
        #select shorter side from im.size, find factor that will give 256 pixels. find resize values, apply thumbnail method

        short_side_len = min(im.size)    
        factor = 256/short_side_len

        resize_w = width*factor
        resize_h = height*factor

        im.copy()

        im.thumbnail((resize_w, resize_h))

        # find coordinates for crop

        left = (resize_w - 224)/2
        top = (resize_h - 224)/2
        right = (resize_w + 224)/2
        bottom = (resize_h + 224)/2

        cropped = im.crop((left, top, right, bottom))

        np_image = np.array(cropped)

        np_image = np_image/255

        np_image = (np_image - means)/stds

        out_array = np_image.transpose((2,0,1))

        tensor = torch.from_numpy(out_array).float()

        return tensor
    
    # Check for GPU, inform user if unavailable
    if gpu:
        if torch.cuda.is_available():
            pass
        else:
            click.secho("No GPU available, training on CPU", fg='red')

    image = process_image(image_path)
    
    # Build up the model structure from scratch based on checkpoint
    
    checkpoint = torch.load(checkpoint)
    
    if checkpoint['arch'] == 'vgg13':
        model = models.vgg13(pretrained=True)
    else:
        model = models.vgg16(pretrained=True)
    
    for parameter in model.parameters():
        parameter.requires_grad = False
    
    # Take hidden_units value from checkpoint and apply it to this classifier
    
    new_classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(4096, checkpoint['hidden_units'])),
                                       ('relu', nn.ReLU()),
                                       ('dropout', nn.Dropout()),
                                       ('fc2', nn.Linear(checkpoint['hidden_units'], 102)),
                                       ('output', nn.LogSoftmax(dim=1))]))
    model.classifier[6] = new_classifier
    
    # Now load checkpoint and apply saved parameters
    
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model, image = model.to(device), image.to(device)

    # Add batch dimension since only passing one image
    
    image.unsqueeze_(0)
    
    model.eval()
    
    log_ps = model(image)
    ps = torch.exp(log_ps)
    probs, classes = ps.topk(top_k, dim=1)
    
    # convert tensor to list
    probs, classes = probs[0].tolist(), classes[0].tolist()
    
    # flip key/values class_to_index    
    idx_to_class = {v: k for k, v in model.class_to_idx.items()}

    # map index to class, then class to name
    top_classes = [idx_to_class[cls] for cls in classes]
    
    if category_names:
        with open('cat_to_name.json', 'r') as f:
            cat_to_name = json.load(f)
            top_names = [cat_to_name[cls] for cls in top_classes]
            
            for index in range(top_k):
                rank = index + 1
                click.secho(f"Rank {rank}: {top_names[index]}, Probability: {probs[index]:.3f}", fg='green')
                
    else:
        for index in range(top_k):
            rank = index + 1
            click.secho(f"Rand {rank}: Class {top_classes[index]}, Probability: {probs[index]:.3f}", fg='green')
    
if __name__ == '__main__':
    predict()