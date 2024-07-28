import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch import tensor
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
import torchvision.models as models
from collections import OrderedDict
import json
import PIL
from PIL import Image
import argparse

arch = {"vgg16":25088,
        "densenet121":1024,
        "alexnet":9216}


def load_data(where  = "./flowers" ):   

    flowers_data_dir = where
    training_data_dir = flowers_data_dir + '/train'
    validation_data_dir = flowers_data_dir + '/valid'
    testing_data_dir = flowers_data_dir + '/test'

    #Apply the required transfomations to the test dataset in order to maximize the efficiency of the learning
    #process

 train_data_transforms = transforms.Compose([
    transforms.Resize(255),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

    # Crop and Resize the data and validation images in order to be able to be fed into the network

test_data_transforms = transforms.Compose([
    transforms.Resize(255),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

valid_data_transforms = transforms.Compose([
    transforms.Resize(255),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


    # TODO: Load the datasets with ImageFolder
train_dataset = datasets.ImageFolder(training_data_dir, transform=train_data_transforms)
valid_dataset = datasets.ImageFolder(validation_data_dir, transform=valid_data_transforms)
test_dataset = datasets.ImageFolder(testing_data_dir, transform=test_data_transforms)

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    # The data loaders are going to use to load the data to the NN(no shit Sherlock)
train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
valid_data_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=64)
test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64)



    return train_data_loader , valid_data_loader, test_data_loader


def nn_setup(model_arch='densenet121', dropout_prob=0.5, hidden_units=120, learning_rate=0.001, device='gpu'):
    if model_arch == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif model_arch == 'densenet121':
        model = models.densenet121(pretrained=True)
    elif model_arch == 'alexnet':
        model = models.alexnet(pretrained=True)
    else:
        print(f"Invalid model architecture: {model_arch}. Choose from 'vgg16', 'densenet121', or 'alexnet'.")

    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(collections.OrderedDict([
        ('dropout', nn.Dropout(dropout_prob)),
        ('fc1', nn.Linear(arch[model_arch], hidden_units)),
        ('relu1', nn.ReLU()),
        ('fc2', nn.Linear(hidden_units, 90)),
        ('relu2', nn.ReLU()),
        ('fc3', nn.Linear(90, 80)),
        ('relu3', nn.ReLU()),
        ('fc4', nn.Linear(80, 102)),
        ('output', nn.LogSoftmax(dim=1))
    ]))

    model.classifier = classifier
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), learning_rate)

    if torch.cuda.is_available() and device == 'gpu':
        model.cuda()

    return model, criterion, optimizer



def train_neural_network(model, criterion, optimizer, epochs=3, print_every=20, data_loader=train_data_loader, device='gpu'):
    step_count = 0
    total_loss = 0

    print("Starting the training process...")

    for epoch in range(epochs):
        total_loss = 0

        for batch_index, (inputs, labels) in enumerate(data_loader):
            step_count += 1

            if torch.cuda.is_available() and device == 'gpu':
                inputs, labels = inputs.to('cuda'), labels.to('cuda')

            optimizer.zero_grad()

            # Forward and backward passes
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if step_count % print_every == 0:
                model.eval()
                validation_loss = 0
                accuracy = 0

                for batch_index, (inputs2, labels2) in enumerate(valid_data_loader):
                    optimizer.zero_grad()

                    if torch.cuda.is_available():
                        inputs2, labels2 = inputs2.to('cuda:0'), labels2.to('cuda:0')
                        model.to('cuda:0')

                    with torch.no_grad():
                        outputs = model.forward(inputs2)
                        validation_loss = criterion(outputs, labels2)
                        probabilities = torch.exp(outputs).data
                        correct_predictions = (labels2.data == probabilities.max(1)[1])
                        accuracy += correct_predictions.type_as(torch.FloatTensor()).mean()

                validation_loss = validation_loss / len(valid_data_loader)
                accuracy = accuracy / len(valid_data_loader)

                print("Epoch: {}/{}... ".format(epoch + 1, epochs),
                      "Loss: {:.4f}".format(total_loss / print_every),
                      "Validation Loss: {:.4f}".format(validation_loss),
                      "Accuracy: {:.4f}".format(accuracy))

                total_loss = 0

    print("Training completed successfully.")
    print("The neural network has been trained using the following settings:")
    print("Epochs: {}".format(epochs))
    print("Total Steps: {}".format(step_count))
    print("That's a significant number of steps!")

def save_neural_network_checkpoint(filepath='checkpoint.pth', architecture='densenet121', hidden_units=120, dropout_prob=0.5, learning_rate=0.001, num_epochs=12):

    model.class_to_idx = train_data.class_to_idx
    model.cpu()
    torch.save({'architecture': architecture,
                'hidden_units': hidden_units,
                'dropout_prob': dropout_prob,
                'learning_rate': learning_rate,
                'num_epochs': num_epochs,
                'state_dict': model.state_dict(),
                'class_to_idx': model.class_to_idx},
               filepath)

def load_neural_network_checkpoint(filepath='checkpoint.pth'):

    checkpoint = torch.load(filepath)
    architecture = checkpoint['architecture']
    hidden_units = checkpoint['hidden_units']
    dropout_prob = checkpoint['dropout_prob']
    learning_rate = checkpoint['learning_rate']

    model, _, _ = setup_neural_network(architecture, dropout_prob, hidden_units, learning_rate)

    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])



def process_image(image_path):

    for path in image_path:
        image = Image.open(path)  # Open the image

    image_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    processed_image = image_transforms(image)

    return processed_image



def predict(image_path, model, topk=5, power='gpu'):
    
    if torch.cuda.is_available() and power == 'gpu':
        model.to('cuda:0')

    processed_image = process_image(image_path)
    processed_image = processed_image.unsqueeze_(0)
    processed_image = processed_image.float()

    if power == 'gpu':
        with torch.no_grad():
            output = model(processed_image.cuda())
    else:
        with torch.no_grad():
            output = model(processed_image)

    probabilities = F.softmax(output.data, dim=1)

    return probabilities.topk(topk)
