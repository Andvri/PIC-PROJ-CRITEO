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

from dataloader_image import *

#data_dir = 'data/hymenoptera_data'
data_dir = 'data/google taxonomy'
train_dir = data_dir + '/train'
test_dir = data_dir + '/val'

cross_val = True
k_folds = 5

batch_size = 1

model_name = "resnet50"
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

def categories_embedding(model, train_dataloader, num_classes, embedding_size):
    embedding = np.zeros((num_classes,embedding_size))
    label_count = [0]*num_classes
    for inputs, labels in train_dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        outputs = outputs.detach().numpy()
        labels = labels.detach().numpy()
        for i in range(len(labels)):
            label = labels[i]
            embedding[label] += outputs[i]
            label_count[label] +=1
    for i in range(num_classes):
        if label_count[i]:
            embedding[i]/=label_count[i]
    return embedding

def eval(model, test_dataloader, embedding, class_names):
    correct = 0
    for inputs, labels in test_dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        cos = nn.CosineSimilarity(dim=0, eps=1e-6)
        for i in range(len(labels)):
            output = outputs[i]
            label = labels[i]
            label_predict = 0
            similarity_max = 0
            for j in range(len(embedding)):
                category_embedding = torch.LongTensor(embedding[j])
                similarity = cos(output, category_embedding)
                if similarity > similarity_max:
                    similarity_max = similarity
                    label_predict = j
            if label==label_predict:
                correct+=1
            #print("real class : ", class_names[label])
            #print("predicted class : ", class_names[label_predict])
            #print("---------------------------------")
    print("accuracy", correct/(len(test_dataloader)*batch_size))
    return correct/(len(test_dataloader)*batch_size)

model = my_models[model_name](pretrained=True)
if model_name in {"resnet18", "resnet34", "resnet50"}:
    num_features = model.fc.in_features
    model.fc = nn.Sequential(*list(model.fc.children())[:-1])
elif model_name in {"alexnet", "vgg11"} :
    num_features = model.classifier[6].in_features
    model.classifier[6] = nn.Sequential(*list(model.classifier[6].children())[:-1])




model = model.to(device)

model.eval() 

if cross_val :
    print("CROSS VALIDATION with ", k_folds," folds")
    
    train_dataset = datasets.ImageFolder(train_dir,data_transforms['train'])
    test_dataset = datasets.ImageFolder(test_dir,data_transforms['val'])

    dataset = ConcatDataset([train_dataset, test_dataset])
    kfold = KFold(n_splits=k_folds, shuffle=True)
    train_accuracy = 0
    test_accuracy = 0
    for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):

        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)

        train_dataloader = torch.utils.data.DataLoader(
                      dataset, 
                      batch_size=batch_size, sampler=train_subsampler)
        test_dataloader = torch.utils.data.DataLoader(
                      dataset,
                      batch_size=batch_size, sampler=test_subsampler)

        embedding = categories_embedding(model, train_dataloader, num_classes, num_features)
        train_accuracy += eval(model, train_dataloader, embedding, class_names)
        test_accuracy += eval(model, test_dataloader, embedding, class_names)
    train_accuracy /= k_folds
    test_accuracy /= k_folds
    print('average train accuracy = ', train_accuracy)
    print('average test accuracy = ', test_accuracy)
else :
    print("TRAIN")
    embedding = categories_embedding(model, train_dataloader, num_classes, num_features)
    accuracy_train = eval(model, train_dataloader, embedding, class_names)
    print("TEST")
    accuracy_test = eval(model, test_dataloader, embedding, class_names)

