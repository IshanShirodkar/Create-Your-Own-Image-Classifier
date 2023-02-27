import torch
from torch import nn
from torch import optim
import torch.nn.functional as F

import torchvision
from torchvision import datasets, transforms, models

from collections import OrderedDict

import json

import time

from PIL import Image

import numpy as np

import matplotlib.pyplot as plt

from torch.autograd import Variable

import argparse

import functions_train
import functions_predict

parser = argparse.ArgumentParser(description='Train.py')

parser.add_argument('data_dir', nargs='*', action='store', default='./flowers/')
parser.add_argument('--gpu', dest='gpu', action='store', default='gpu')
parser.add_argument('--save_dir', dest='save_dir', action='store', default='./checkpoint.pth')
parser.add_argument('--learning_rate', dest='learning_rate', action='store', default=0.01)
parser.add_argument('--epochs', dest='epochs', action='store', type=int, default=10)
parser.add_argument('--arch', dest='arch', action='store', default='vgg16', type=str)
parser.add_argument('--hidden_units', type=int, dest='hidden_units', action='store', default=120)

parser = parser.parse_args()
data_dir = parser.data_dir
path = parser.save_dir
lr = parser.learning_rate
arch = parser.arch
middle_features = parser.hidden_units
use = parser.gpu
epochs = parser.epochs


def load_data(data_dir='./flowers'):
    data_dir = str(data_dir).strip('[]').strip("'")
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    cropped_size = 224
    resized_size = 255
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]

    train_transforms = transforms.Compose([transforms.RandomResizedCrop(cropped_size),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize(means, stds)])

    validate_transforms = transforms.Compose([transforms.RandomResizedCrop(cropped_size),
                                              transforms.ToTensor(),
                                              transforms.Normalize(means, stds)])

    # resize then crop the images to the appropriate size
    test_transforms = transforms.Compose([transforms.Resize(resized_size),  # Why 255 pixels?
                                          transforms.CenterCrop(cropped_size),
                                          transforms.ToTensor(),
                                          transforms.Normalize(means, stds)])

    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    validate_data = datasets.ImageFolder(valid_dir, transform=validate_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

    image_data = [train_data, validate_data, test_data]

    batch_size = 60

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    validate_loader = torch.utils.data.DataLoader(validate_data, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data)

    dataloaders = [train_loader, validate_loader, test_loader]
    return train_loader, validate_loader, test_loader, train_data


def build_model(arch='vgg16', middle_features=1024, learning_rate=0.01, device='gpu'):
    if arch == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif arch == 'densenet121':
        model = models.densenet121(pretrained=True)
    elif arch == 'alexnet':
        model = models.alexnet(pretrained=True)
    else:
        print("Im sorry but {} is not a valid model. Did you mean vgg16, densenet121 or alexnet?".format(arch))

    for param in model.parameters():
        param.requires_grad = False

    input_features = 25088

    output_number = 102
    dropout_probability = 0.5
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(input_features, middle_features)),
        ('drop', nn.Dropout(p=dropout_probability)),
        ('relu', nn.ReLU()),
        ('fc2', nn.Linear(middle_features, output_number)),
        ('output', nn.LogSoftmax(dim=1))
    ]))

    model.classifier = classifier

    criterion = nn.NLLLoss()
    optimizer = optim.SGD(model.classifier.parameters(), lr=learning_rate)

    if torch.cuda.is_available() and device == 'gpu':
        model.cuda()

    return model, criterion, optimizer


def train_model(model, criterion, optimizer, validate_loader, train_loader, use='gpu', epochs=10):
    running_loss = 0
    steps = 0

    validation = True

    device = torch.device('cuda' if torch.cuda.is_available() and use == 'gpu' else 'cpu')
    model.to(device)

    start_time = time.time()
    print('Training starts')

    for epoch in range(epochs):
        training_loss = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            log_probabilities = model(images)
            loss = criterion(log_probabilities, labels)
            loss.backward()
            optimizer.step()

            training_loss += loss.item()

        print('Epoch {} of {} ///'.format(epoch + 1, epochs),
              'Training loss {:.3f} ///'.format(training_loss / len(train_loader)))

        if validation == True:
            validation_loss = 0
            validation_accuracy = 0
            model.eval()

            with torch.no_grad():
                for images, labels in validate_loader:
                    images, labels = images.to(device), labels.to(device)

                    log_probabilities = model(images)
                    loss = criterion(log_probabilities, labels)

                    validation_loss += loss.item()

                    probabilities = torch.exp(log_probabilities)
                    top_probability, top_class = probabilities.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    validation_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

            model.train()

            print("Validation loss {:.3f} ///".format(validation_loss / len(validate_loader)),
                  "Validation accuracy {:.3f} ///".format(validation_accuracy / len(validate_loader)))

    end_time = time.time()
    print('Training ends')

    training_time = end_time - start_time
    print('Training time {:.0f}m {:.0f}s'.format(training_time / 60, training_time % 60))

    return model


def save_checkpoint(model, optimizer, train_data, arch='vgg16', path='checkpoint.pth', input_features=25088,
                    middle_features=1024, output_number=102, lr=0.01, epochs=10, batch_size=64):
    model.class_to_idx = train_data.class_to_idx

    checkpoint = {'network': 'vgg16',
                  'input': input_features,
                  'output': output_number,
                  'learning_rate': lr,
                  'batch_size': batch_size,
                  'classifier': model.classifier,
                  'epochs': epochs,
                  'optimizer': optimizer.state_dict(),
                  'state_dict': model.state_dict(),
                  'class_to_idx': model.class_to_idx}

    torch.save(checkpoint, path)


train_loader, validate_loader, test_loader, train_data = functions_train.load_data(data_dir)

model, criterion, optimizer = functions_train.build_model(arch, middle_features, lr, use)

functions_train.train_model(model, criterion, optimizer, validate_loader, train_loader, use, epochs)

functions_train.save_checkpoint(model, optimizer, train_data, arch, path, middle_features, lr, epochs)

print("Model is trained") 