from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

from dataloader_image import *
from saveFig import *
#data_dir = 'data/hymenoptera_data'
data_dir = 'data/google taxonomy'
train_dir = data_dir + '/train'
test_dir = data_dir + '/val'

batch_size = 4
epochs = 25
lr = 0.001
momentum = 0.9

step_size = 7
gamma = 0.1

model_name = "vgg11"
my_models = {"resnet18" : models.resnet18,
             "resnet34" : models.resnet34,
             "resnet50" : models.resnet50,
             "alexnet" : models.alexnet,
             "vgg11" : models.vgg11_bn
            }
input_size = {"resnet18" : 224, "resnet34": 224, "resnet50": 224, "alexnet": 224, "vgg11":224}




# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(input_size[model_name]),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(input_size[model_name]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}
train_dataloader, test_dataloader, dataset_sizes, class_names= dataloader(train_dir, test_dir, batch_size, data_transforms)
num_classes = len(class_names)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    accuracies_train = []
    losses_train = []
    accuracies_test = []
    losses_test = []
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
                dataloader = train_dataloader
            else:
                model.eval()   # Set model to evaluate mode
                dataloader = test_dataloader

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            if phase == 'train':
                accuracies_train.append(epoch_acc)
                losses_train.append(epoch_loss)
            else :
                accuracies_test.append(epoch_acc)
                losses_test.append(epoch_loss)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, accuracies_train, accuracies_test, losses_train, losses_test

model = my_models[model_name](pretrained=True)

if model_name in {"resnet18", "resnet34", "resnet50"}:
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
elif model_name in {"alexnet", "vgg11"} :
    num_ftrs = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_ftrs,num_classes)

model = model.to(device)

criterion = nn.CrossEntropyLoss()

optimizer_ft = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=step_size, gamma=gamma)

model,accuracies_train, accuracies_test,losses_train, losses_test = train_model(model, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=epochs)

saveFig(model_name, 'accuracy', accuracies_train, accuracies_test, 1)
saveFig(model_name, 'loss', losses_train, losses_test, 1)



