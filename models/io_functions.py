from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import ConcatDataset
from sklearn.model_selection import KFold
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy


import torch
from torchvision import datasets 

from saveFig import *

def dataloader(train_dir, test_dir, batch_size, data_transforms):
    train_dataset = datasets.ImageFolder(train_dir,data_transforms['train'])
    test_dataset = datasets.ImageFolder(test_dir,data_transforms['val'])

    dataset_sizes = {'train': len(train_dataset), 'val':len(test_dataset)}
    class_names = train_dataset.classes

    if batch_size is None :
        return train_dataset, test_dataset, dataset_sizes, class_names
    else :
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,shuffle=True, num_workers=4)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,shuffle=True, num_workers=4)
        weights_train, weights_test = weights(train_dataloader, test_dataloader)
        return train_dataloader, test_dataloader, dataset_sizes, class_names, weights_train, weights_test

def weights(train_dataloader, test_dataloader):
    weights_train = dict()
    weights_test = dict()

    for _,labels in train_dataloader :
        for label in labels:
            if label.item() not in weights_train.keys():
                weights_train[label.item()] = 0
            weights_train[label.item()] += 1
    for _,labels in test_dataloader :
        for label in labels:
            if label.item() not in weights_test.keys():
                weights_test[label.item()] = 0
            weights_test[label.item()] += 1
    return weights_train, weights_test

def initiate_model(model_name, my_models, lr, momentum, step_size, gamma, num_classes, version):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = my_models[model_name](pretrained=True)
    
    if version == 2:
        for param in model.parameters():
            param.requires_grad = False

    if model_name in {"resnet18", "resnet34", "resnet50"}:
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    elif model_name in {"alexnet", "vgg11"} :
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs,num_classes)

    model = model.to(device)



    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    return model, criterion, optimizer, exp_lr_scheduler, device



