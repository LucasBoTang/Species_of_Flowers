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


# TODO: Train a model with a pre-trained network
def training(model, train_loader, val_loader, epochs, print_every, criterion, optimizer, device):

    steps = 0
    model.to(device)

    for e in range(epochs):
        running_loss = 0

        for _, (inputs, labels) in enumerate(train_loader):

            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            # Forward and backward passes
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                # Make sure network is in eval mode for inference
                model.eval()

                # Turn off gradients for validation, saves memory and computations
                with torch.no_grad():
                    test_loss, accuracy = validation(model, val_loader, criterion)

                print("Epoch: {}/{}... ".format(e+1, epochs),
                      "Training Loss: {:.3f}... ".format(running_loss / print_every),
                      "Validation Loss: {:.3f}... ".format(test_loss),
                      "Validation Accuracy: {:.3f}".format(accuracy))

            running_loss = 0


# TODO: Implement a function for the validation pass
def validation(model, test_loader, criterion):

    test_loss = 0
    accuracy = 0

    for _, (inputs, labels) in enumerate(test_loader):

        inputs, labels = inputs.to(device), labels.to(device)

        output = model.forward(inputs)
        test_loss += criterion(output, labels).item()

        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()

    return test_loss / len(test_loader), accuracy / len(test_loader)


# TODO: Input parameters
parser = argparse.ArgumentParser()
parser.add_argument('-gpu', action = 'store_true',
                    dest = 'gpu',
                    default = False,
                    help='Use GPU or not')
parser.add_argument('-dense', action = 'store_true',
                    dest = 'arch',
                    default = False,
                    help = 'Architecture of Model: DenseNet or VGG')
parser.add_argument('-e', action='store',
                    dest = 'epochs',
                    type = int,
                    default = 5,
                    help = 'Number of epochs')
parser.add_argument('-r', action = 'store',
                    dest = 'learning_rate',
                    type = float,
                    default = 0.0005,
                    help = 'Learning rate of adam')
args = parser.parse_args()


# Load the data
data_dir = 'flowers'
train_dir = data_dir + '/train'
val_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

# TODO: Define your transforms for the training, validation, and testing sets
augm_transforms = transforms.Compose([transforms.RandomRotation(30),
                                      transforms.RandomResizedCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])
norm_transforms = transforms.Compose([transforms.RandomRotation(30),
                                      transforms.RandomResizedCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

# TODO: Load the datasets with ImageFolder
print('Loading the datasets...')
trainset = datasets.ImageFolder(train_dir, transform=augm_transforms)
valset = datasets.ImageFolder(val_dir, transform=norm_transforms)
testset = datasets.ImageFolder(test_dir, transform=norm_transforms)

# TODO: Using the image datasets and the trainforms, define the dataloaders
train_loader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
val_loader = torch.utils.data.DataLoader(valset, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

# TODO: Load densenet121
if args.arch:
    print('Loading DenseNet...')
    model = models.densenet121(pretrained=True)
    # Architecture of classifier
    input_size = 1024
    hidden_sizes = [512, 256]
    output_size = 102
else:
    print('Loading VGG...')
    model = models.vgg16(pretrained=True)
    # Architecture of classifier
    input_size = 25088
    hidden_sizes = [4096, 1024]
    output_size = 102


# TODO: Freeze parameters so we don't backprop through them
for param in model.parameters():
    param.requires_grad = False

# TODO: Set a new classifier
print('Building a new classifier...')
classifier = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                           nn.ReLU(),
                           nn.Dropout(0.3),
                           nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                           nn.ReLU(),
                           nn.Dropout(0.3),
                           nn.Linear(hidden_sizes[1], output_size),
                           nn.LogSoftmax(dim=1))
model.classifier = classifier

# TODO: Train the model
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)
if args.gpu:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print('Training the model with cuda...')
    else:
        print('Gpu is not avaliable! Training the model with cpu...')
else:
    device = torch.device("cpu")
    print('Training the model with cpu...')
epochs = args.epochs
print_every = 32
training(model, train_loader, val_loader, epochs, print_every, criterion, optimizer, device)

# TODO: Do validation on the test set
test_loss, test_accuracy = validation(model, test_loader, criterion)
print("Test Loss: %f.. Test Accuracy : %f " % (test_loss, test_accuracy))

# TODO: Save the checkpoint
print('Saving the checkpoint...')
model.class_to_idx = trainset.class_to_idx
checkpoint = {'densenet': args.arch,
              'input_size': input_size,
              'hidden_layers': hidden_sizes,
              'output_size': output_size,
              'state_dict': model.state_dict(),
              'class_to_idx': model.class_to_idx}
torch.save(checkpoint, 'checkpoint.pth')
print('Checkpoint has been saved!')
