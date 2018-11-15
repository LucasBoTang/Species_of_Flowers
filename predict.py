# Imports here
import numpy as np
from PIL import Image
import argparse
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import argparse


# TODO: Write a function that loads a checkpoint and rebuilds the model
def load_checkpoint(filepath):

    checkpoint = torch.load(filepath)

    classifier = nn.Sequential(nn.Linear(checkpoint['input_size'], checkpoint['hidden_layers'][0]),
                               nn.ReLU(),
                               nn.Dropout(0.3),
                               nn.Linear(checkpoint['hidden_layers'][0], checkpoint['hidden_layers'][1]),
                               nn.ReLU(),
                               nn.Dropout(0.3),
                               nn.Linear(checkpoint['hidden_layers'][1],  checkpoint['output_size']),
                               nn.LogSoftmax(dim=1))

    if checkpoint['densenet']:
        model = models.densenet121(pretrained=True)
    else:
        model = models.vgg16(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    model.classifier = classifier
    model.eval()

    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']

    return model


# TODO: Process a PIL image for use in a PyTorch model
def process_image(image):
    """
    Scales, crops, and normalizes a PIL image for a PyTorch model,
    returns an Numpy array
    """

    # Load
    pl_img = Image.open(image)#.resize((256,256)).crop((16, 16, 240, 240))

    # Resize and maintain ratio
    width, height = pl_img.size
    new_width = width * 256 // max(width, height)
    new_height = height * 256 // max(width, height)
    pl_img = pl_img.resize((new_width, new_height))

    # Center crop
    left = new_width // 2 - 112
    down = new_height // 2 - 112
    right = new_width // 2 + 112
    up = new_height // 2 + 112
    pl_img = pl_img.crop((left, down, right, up))

    # Normalize
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_img = np.array(pl_img) / 255
    norm_img = (np_img - mean) / std

    # Transpose
    img = norm_img.transpose((2,0,1))

    return torch.from_numpy(img)


# TODO: Write a function to predict the class from an image file
def predict(image_path, model, topk):
    """
    Predict the class (or classes) of an image using a trained deep learning model.
    """

    model.eval()
    img = process_image(image_path)
    tensor = img.type(torch.FloatTensor).to(device)
    tensor.resize_([1, 3, 224, 224])

    with torch.no_grad():
        output = model.forward(tensor)
        probs = torch.exp(output)

    probs, classes = probs.topk(topk)[0][0].cpu(), probs.topk(topk)[1][0].cpu()
    return np.array(probs), np.array(classes)


# TODO: Input parameters
parser = argparse.ArgumentParser()
parser.add_argument('-gpu', action = 'store_true',
                    dest = 'gpu',
                    default = False,
                    help='Use GPU or not')
parser.add_argument('-cat', action = 'store',
                    dest = 'cat',
                    type = str,
                    default = 'cat_to_name.json',
                    help = 'JSON file to map categories')
parser.add_argument('-topk', action='store',
                    dest = 'topk',
                    type = int,
                    default = 5,
                    help = 'Top K possibilities')
parser.add_argument('-img', action = 'store',
                    dest = 'img',
                    type = str,
                    default = 'flowers/test/23/image_03382.jpg',
                    help = 'File path of image')
args = parser.parse_args()

# TODO: Set the enviroment
if args.gpu:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print('Using the model with cuda...')
    else:
        print('Gpu is not avaliable! Using the model with cpu...')
else:
    device = torch.device("cpu")
    print('Using the model with cpu...')

# TODO: Load model
print('Loading model from checkpoint...')
model = load_checkpoint('checkpoint.pth')
model.to(device)

# TODO: Map lables
print('Mapping labels...')
import json
with open(args.cat, 'r') as f:
    cat_to_name = json.load(f)
cat_to_idx = model.class_to_idx
idx_to_cat = {v: k for k, v in cat_to_idx.items()}
idx_to_name = {k: cat_to_name[v] for k, v in idx_to_cat.items()}

# TODO: Make predictions
print('Making predictions...')
probs, classes = predict(args.img, model, args.topk)
names = []
for c in classes:
    names.append(idx_to_name[c])
for i in range(args.topk):
    print(names[i]+':', '%.4f%%' % (probs[i] * 100))
