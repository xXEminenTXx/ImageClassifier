# python imports
import numpy as np
import torch
from workspace_utils import active_session
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

# Validates accuracy
def validation(model, validateloader, criterion, gpu):

    val_loss = 0
    accuracy = 0

    for images, labels in iter(validateloader):

        # run on gpu if enabled
        if gpu:
            images, labels = images.to('cuda'), labels.to('cuda')

        output = model.forward(images)
        val_loss += criterion(output, labels).item()

        probabilities = torch.exp(output)

        equality = (labels.data == probabilities.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()

    return val_loss, accuracy

# Trains the network
def train_classifier(gpu, epochs, criterion, optimizer, model, data_dir):
    
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
    
    # Set directory vars
    data_dir = data_dir
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    # Load the datasets with ImageFolder
    training_dataset = datasets.ImageFolder(train_dir, transform=training_transforms)
    validation_dataset = datasets.ImageFolder(valid_dir, transform=validation_transforms)
    testing_dataset = datasets.ImageFolder(test_dir, transform=testing_transforms)
    
    # Using the image datasets and the trainforms, define the dataloaders
    train_loader = torch.utils.data.DataLoader(training_dataset, batch_size=64, shuffle=True)
    validate_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=32)
    test_loader = torch.utils.data.DataLoader(testing_dataset, batch_size=32)

    with active_session():

        steps = 0
        print_every = 40

        if gpu:
            model.to('cuda')

            for e in range(epochs):

                model.train()

                running_loss = 0

                for images, labels in iter(train_loader):

                    steps += 1

                    if gpu:
                        images, labels = images.to('cuda'), labels.to('cuda')

                    optimizer.zero_grad()

                    output = model.forward(images)
                    loss = criterion(output, labels)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()

                    if steps % print_every == 0:

                        model.eval()

                        # Turn off gradients for validation, saves memory and computations
                        with torch.no_grad():
                            validation_loss, accuracy = validation(model, validate_loader, criterion, gpu)

                        print("Epoch: {}/{}.. ".format(e+1, epochs),
                              "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                              "Validation Loss: {:.3f}.. ".format(validation_loss/len(validate_loader)),
                              "Validation Accuracy: {:.3f}".format(accuracy/len(validate_loader)))

                        running_loss = 0
                        model.train()

# Tests the network
def test_accuracy(model, test_loader, gpu):

    model.eval()

    if gpu:
        model.to('cuda')

    with torch.no_grad():

        accuracy = 0

        for images, labels in iter(test_loader):

            if gpu:
                images, labels = images.to('cuda'), labels.to('cuda')

            output = model.forward(images)

            probabilities = torch.exp(output)

            equality = (labels.data == probabilities.max(dim=1)[1])

            accuracy += equality.type(torch.FloatTensor).mean()

        print("Test Accuracy: {}".format(accuracy/len(test_loader)))    
  
# Saves the checkpoint 
def save_checkpoint(model, arch, classifier, save_dir, training_dataset):

    model.class_to_idx = training_dataset.class_to_idx

    checkpoint = {'arch': arch,
                  'class_to_idx': model.class_to_idx,
                  'model_state_dict': model.state_dict(),
                  'classifier' : classifier
                 }

    torch.save(checkpoint, save_dir + '/checkpoint.pth')
