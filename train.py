# Programmer info
# Name: Fahad Moidy
# Date: 5/03/2022

# Example command
# python train.py ./flowers --learning_rate 0.001 --hidden_units 5000 --epochs 15 --gpu --save_dir ./saves

# Python Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import cv2
import json
from collections import OrderedDict

import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

# Program dependacies
from input_arg_handler import train_fetch_input_args
from train_functions import train_classifier, test_accuracy, save_checkpoint



# Define main function
def main():
    # Handle command line args (or lack of)
    args = train_fetch_input_args()
    
    print("Loading data from: %s, Architecture: %s, Learning rate: %.3f, With %d Hidden units and %d epochs. Using GPU: %s. Saving to: %s" % (
        args["data_dir"], args["arch"], args["learning_rate"], args["hidden_units"], args["epochs"], args["gpu"], args["save_dir"]))
    
    data_dir = args["data_dir"]
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # Define transforms for training, validation, and testing sets
    training_transforms = transforms.Compose([transforms.RandomRotation(30),
                                              transforms.RandomResizedCrop(224),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.485, 0.456, 0.406], 
                                                                   [0.229, 0.224, 0.225])])

    validation_transforms = transforms.Compose([transforms.Resize(256),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406], 
                                                                     [0.229, 0.224, 0.225])])

    testing_transforms = transforms.Compose([transforms.Resize(256),
                                             transforms.CenterCrop(224),
                                             transforms.ToTensor(),
                                             transforms.Normalize([0.485, 0.456, 0.406], 
                                                                  [0.229, 0.224, 0.225])])
    
    
    # Load the datasets with ImageFolder
    training_dataset = datasets.ImageFolder(train_dir, transform=training_transforms)
    validation_dataset = datasets.ImageFolder(valid_dir, transform=validation_transforms)
    testing_dataset = datasets.ImageFolder(test_dir, transform=testing_transforms)

    # Using the image datasets and the trainforms, define the dataloaders
    train_loader = torch.utils.data.DataLoader(training_dataset, batch_size=64, shuffle=True)
    validate_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=32)
    test_loader = torch.utils.data.DataLoader(testing_dataset, batch_size=32)
    
    # Load category map
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    
    # Download network
    model = eval("models." + args['arch'] + "(pretrained=True)")
    print(model)
    
    # Freeze pretrained model parameters to avoid backpropogating through them
    for parameter in model.parameters():
        parameter.requires_grad = False


    # Build custom classifier
    classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(25088, args["hidden_units"])),
                                            ('relu', nn.ReLU()),
                                            ('drop', nn.Dropout(p=0.5)),
                                            ('fc2', nn.Linear(args["hidden_units"], 102)),
                                            ('output', nn.LogSoftmax(dim=1))]))

    model.classifier = classifier

    # Loss function and gradient descent

    criterion = nn.NLLLoss()

    optimizer = optim.Adam(model.classifier.parameters(), lr=args["learning_rate"])
    
    # Train the classifyer
    train_classifier(args["gpu"], args["epochs"], criterion, optimizer, model, args["data_dir"]) 

    
    # Test network
    test_accuracy(model, test_loader, args["gpu"])
    
    # Save the checkpoint
    save_checkpoint(model, args["arch"], classifier, args["save_dir"], training_dataset)

    
#   Call main function, starting the program
if __name__ == "__main__":
    main()
